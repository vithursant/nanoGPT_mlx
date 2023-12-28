import os
import math
import time
import numpy as np
from typing import List

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten, tree_map

from model import GPTConfig, GPT
from optimizer import AdamW

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
eval_only = False

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
