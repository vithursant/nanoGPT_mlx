import os
import math
import time

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from typing import List

import numpy as np
from mlx.utils import tree_flatten, tree_map
from dataclasses import dataclass

import pdb


# model
n_layer = 12
n_head = 12
n_embd = 768
dropout = 0.0 # for pretraining 0 is good, for finetuning try 0.1+
bias = False # do we use bias inside LayerNorm and Linear layers?

# adamw optimizer
learning_rate = 2.6e-5 # max learning rate
max_lr = learning_rate
min_lr = 2.6e-6
num_iters = 600000 # total number of training iterations
warmup_iters = int(num_iters * 0.1)
lr_decay_iters = num_iters - warmup_iters
max_iters = warmup_iters + lr_decay_iters
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
meta_vocab_size = None

# dataset
dataset = 'openwebtext'
batch_size = 1
gradient_accumulation_steps = 512
context_size = 1024

# eval
eval_interval = 10
log_interval = 10
eval_test = False

# -----------------------------------------------------------------------------
config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
exec(open('configurator.py').read()) # overrides from command line or config file
config = {k: globals()[k] for k in config_keys} # will be useful for logging
# -----------------------------------------------------------------------------

# Load vocab and dataset:
# poor man's data loader
data_dir = '/Users/vithursant/Documents/nanoGPT/data/openwebtext'
train_data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
val_data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')


class AdamW(optim.Adam):
    def __init__(
        self,
        learning_rate: float,
        betas: List[float] = [0.9, 0.999],
        eps: float = 1e-8,
        weight_decay: float = 0.01,
    ):
        super().__init__(learning_rate=learning_rate, betas=betas, eps=eps)
        self.weight_decay = weight_decay

    def apply_single(self, gradient: mx.array, parameter: mx.array, state):
        parameter -= self.weight_decay * self.learning_rate * parameter
        return super().apply_single(gradient, parameter, state)

    def set_learning_rate(self, learning_rate: float):
        self.learning_rate = learning_rate

        
class LayerNorm(nn.Module):
    r"""Applies layer normalization [1] on the inputs.

    Computes

    .. math::

        y = \frac{x - E[x]}{\sqrt{Var[x]} + \epsilon} \gamma + \beta,

    where :math:`\gamma` and :math:`\beta` are learned per feature dimension
    parameters initialized at 1 and 0 respectively.

    [1]: https://arxiv.org/abs/1607.06450

    Args:
        dims (int): The feature dimension of the input to normalize over
        eps (float): A small additive constant for numerical stability
        affine (bool): If True learn an affine transform to apply after the
            normalization
    """

    def __init__(self, dims: int, eps: float = 1e-5, affine: bool = True, bias: bool = False):
        super().__init__()
        if affine:
            self.bias = None
            if bias:
                self.bias = mx.zeros((dims,))
            self.weight = mx.ones((dims,))
        self.eps = eps
        self.dims = dims

    def _extra_repr(self):
        return f"{self.dims}, eps={self.eps}, affine={'weight' in self}, bias={self.bias}"

    def __call__(self, x):
        means = mx.mean(x, axis=-1, keepdims=True)
        var = mx.var(x, axis=-1, keepdims=True)
        x = (x - means) * mx.rsqrt(var + self.eps)
        if self.bias:
            return (self.weight * x + self.bias) if "weight" in self else x
        else:
            return (self.weight * x) if "weight" in self else x


class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)

        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout

    def __call__(self, x, mask):
        B, T, C = x.shape # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        query, key, value = mx.split(self.c_attn(x), 3, axis=2)
        key = key.reshape(B, T, self.n_head, C // self.n_head).transpose(0, 2, 1, 3) # (B, nh, T, hs)
        query = query.reshape(B, T, self.n_head, C // self.n_head).transpose(0, 2, 1, 3) # (B, nh, T, hs)
        value = value.reshape(B, T, self.n_head, C // self.n_head).transpose(0, 2, 1, 3) # (B, nh, T, hs)

        # manual implementation of attention
        att = (query @ key.transpose(0, 1, 3, 2)) * (1.0 / math.sqrt(key.shape[3]))
        mask = mask.reshape(1, 1, T, T)
        att = mx.where(mask[:,:,:T,:T] == 0, att, float('-1e9'))
        # y = att @ value # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        att = mx.softmax(att.astype(mx.float32), axis=-1).astype(att.dtype)
        att = self.attn_dropout(att)
        y = (att @ value).transpose(0, 2, 1, 3).reshape(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y

    @staticmethod
    def create_additive_causal_mask(N: int, dtype: mx.Dtype = mx.float32):
        return mx.tril(mx.ones([N, N])).reshape(1, 1, N, N).astype(dtype)

class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu    = nn.GELU()
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def __call__(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def __call__(self, x, mask):
        x = x + self.attn(self.ln_1(x), mask)
        x = x + self.mlp(self.ln_2(x))
        return x


@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50304 # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()

        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config

        self.embedding = nn.Embedding(config.vocab_size, config.n_embd)
        self.transformer = [Block(config) for _ in range(config.n_layer)]
        self.out_proj = nn.Linear(config.n_embd, config.vocab_size, bias=False)

    def __call__(self, x):
        mask = CausalSelfAttention.create_additive_causal_mask(x.shape[1])
        x = self.embedding(x)
        for l in self.transformer:
            x = l(x, mask)
        return self.out_proj(x)

    def loss(self, x, y, reduce=True):
        logits = self(x)
        losses = nn.losses.cross_entropy(logits, y)
        mx.simplify(losses)

        return mx.mean(losses) if reduce else mx.mean(losses, axis=(-1, -2))


def to_samples(context_size, dataset):
    tokens = dataset.size
    window_size = context_size + 1  # include target
    samples = tokens - window_size + 1
    X = np.lib.stride_tricks.as_strided(
        dataset,
        shape=(samples, window_size),
        strides=(dataset.itemsize, dataset.itemsize),
    )
    return X[:, :-1], X[:, 1:]


def iterate_batches(batch_size, context_size, dataset):
    inputs, targets = to_samples(context_size, dataset)
    s = 0
    while True:
        if s == 0:
            # Reset permutation:
            perm = np.random.permutation(inputs.shape[0])
        ids = perm[s : s + batch_size]
        yield inputs[ids], targets[ids]
        s += batch_size
        if s >= inputs.shape[0]:
            s = 0


def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = np.random.randint(len(data) - context_size, size=(batch_size,))
    x = mx.stack([(mx.array(data[i:i+context_size])) for i in ix]).astype(mx.int64)
    y = mx.stack([(mx.array(data[i+1:i+1+context_size])) for i in ix]).astype(mx.int64)
    return x, y


def numel(sizes):
    numel = 1
    for elem in sizes:
        numel *= elem
    return numel


def print_loss(optimizer, iteration_count, average_loss, tic):
    toc = time.perf_counter()
    print(
        f"iter {iteration_count}: train loss {average_loss:.3f}, "
        f"it/sec {1.0 / (toc - tic):.3f}, "
        f"lr {optimizer.learning_rate:.9f}"
    )
    return toc

def main():
    # model init
    model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=context_size,
                    bias=bias, vocab_size=None, dropout=dropout) # start with model_args from command line

    # Initialize model:
    # model = TransformerLM(len(vocab), n_layer, n_embd, n_head)
    if meta_vocab_size is None:
        print("defaulting to vocab_size of GPT-2 to 50304 (50257 rounded up for efficiency)")
    model_args['vocab_size'] = meta_vocab_size if meta_vocab_size is not None else 50304
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
    print(model)

    mx.eval(model.parameters())
    nparams = sum(
        x.size for k, x in tree_flatten(model.parameters()) if "embedding" not in k
    )
    print(f"Training a transformer with {nparams / 1024**2:.3f} M parameters")
    
    # TODO:
    # # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
    # # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
    # param_dict = {}
    # for k, x in tree_flatten(model.trainable_parameters()):
    #     param_dict[k] = x
    # decay_params = [p for n, p in param_dict.items() if len(p.shape) >= 2]
    # nodecay_params = [p for n, p in param_dict.items() if len(p.shape) < 2]
    # num_decay_params = sum(numel(p.shape) for p in decay_params)
    # num_nodecay_params = sum(numel(p.shape) for p in nodecay_params)
    # print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
    # print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
    # optim_groups = [
    #     {'params': decay_params, 'weight_decay': weight_decay},
    #     {'params': nodecay_params, 'weight_decay': 0.0}
    # ]

    optimizer = AdamW(learning_rate=learning_rate, 
                            betas=[beta1, beta2], 
                            weight_decay=weight_decay)
    loss_and_grad_fn = nn.value_and_grad(model, model.loss)


    X, Y = get_batch('train') # fetch the very first batch
    
    tic = time.perf_counter()
    local_iter_num = 0 # number of iterations in the lifetime of this process
    iter_num = 0
    eval_only = False
    while True:
        if iter_num == 0 and eval_only:
            break

        # optimizer = update_learning_rate(iter_num)
        # pdb.set_trace()
        if iter_num < warmup_iters:
            new_lr = max_lr * iter_num / warmup_iters
        elif iter_num > lr_decay_iters:
            new_lr = min_lr
        else:
            decay_ratio = (iter_num - warmup_iters) / (
                lr_decay_iters - warmup_iters
            )
            assert 0 <= decay_ratio <= 1
            coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
            new_lr = min_lr + coeff * (max_lr - min_lr)
        optimizer.set_learning_rate(new_lr)

        # Gradient Accumulation
        accumulated_grads = tree_map(
                    lambda x: mx.zeros_like(x), model.parameters()
                )
        accumulated_loss = 0.0
        for micro_step in range(gradient_accumulation_steps):
            loss, grads = loss_and_grad_fn(X, Y)

            accumulated_grads = tree_map(
                lambda acc, new: acc + new * (1.0 / gradient_accumulation_steps),
                accumulated_grads,
                grads,
            )

            tree_map(
                lambda grad: mx.eval(grad),
                accumulated_grads,
            )

            accumulated_loss += loss.item()

        loss = mx.array(accumulated_loss / gradient_accumulation_steps) # scale the loss to account for gradient accumulation

        tic = print_loss(optimizer, iter_num, loss.item(), tic)
        # immediately async prefetch next batch while model is doing the forward pass on the GPU
        X, Y = get_batch('train')

        mx.simplify(loss, model.parameters())
        mx.eval(loss, model.parameters())
        model.update(
            optimizer.apply_gradients(accumulated_grads, model)
        )
        accumulated_grads = tree_map(
            lambda x: mx.zeros_like(x), model.parameters()
        )

        iter_num += 1
        local_iter_num += 1

        # termination conditions
        if iter_num > max_iters:
            break

if __name__ == "__main__":
    main()
