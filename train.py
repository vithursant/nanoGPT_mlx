import os
import math
import time
import json

import numpy as np
from typing import List

import mlx
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten, tree_map

from model import GPTConfig, GPT
from optimizer import AdamW
from tboard_utils import init_tensorboard, get_tensorboard


# model
n_layer = 12
n_head = 12
n_embd = 768
dropout = 0.0 # for pretraining 0 is good, for finetuning try 0.1+
bias = False # do we use bias inside LayerNorm and Linear layers?
d_type = 'float32'

# adamw optimizer
learning_rate = 6.0e-4 # max learning rate
min_lr = 6.0e-5
num_iters = 600000 # total number of training iterations
warmup_pct = 0.1
warmup_iters = 2000
lr_decay_iters = 600000
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
save_interval = 1
eval_interval = 10
log_interval = 10
eval_only = False
out_dir = 'gpt2_openwebtext_pretrain'
save_name = 'gpt2_small_owt'

# -----------------------------------------------------------------------------
config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
exec(open('configurator.py').read()) # overrides from command line or config file
config = {k: globals()[k] for k in config_keys} # will be useful for logging
# -----------------------------------------------------------------------------

# Load vocab and dataset:
# poor man's data loader
data_dir = os.path.join('data', dataset)
train_data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
val_data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')

# model save path
save_model_path = os.path.join(out_dir, save_name + '.npz')
save_model_config_path = os.path.join(out_dir, save_name + '.json')

# initialize tboard logging:
os.makedirs(out_dir, exist_ok=True)
tboard_dir = os.path.join(out_dir, "tboard_log")
init_tensorboard(tboard_dir)


def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = np.random.randint(len(data) - context_size, size=(batch_size,))
    x = mx.stack([(mx.array(data[i:i+context_size])) for i in ix]).astype(mx.int64)
    y = mx.stack([(mx.array(data[i+1:i+1+context_size])) for i in ix]).astype(mx.int64)
    return x, y


def print_loss(optimizer, iteration_count, average_loss, tic):
    toc = time.perf_counter()
    print(
        f"iter {iteration_count}: train loss {average_loss:.3f}, "
        f"it/sec {1.0 / (toc - tic):.3f}, "
        f"lr {optimizer.learning_rate.item():.9f}"
    )
    return toc


def update_learning_rate(it):
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    if it > lr_decay_iters:
        return min_lr
    decay_ratio = (it - warmup_iters) / (
        lr_decay_iters - warmup_iters
    )
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    new_lr = min_lr + coeff * (learning_rate - min_lr)
    return new_lr
    

def log_tboard_dict(log_dict, itr, pre, post=''):
    writer = get_tensorboard()
    for k, v in log_dict.items():
        writer.add_scalar(f'{pre}/{k}{post}', v, itr)


def main():
    # model init
    model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=context_size,
                    bias=bias, vocab_size=None, dropout=dropout) # start with model_args from command line

    # initialize model:
    if meta_vocab_size is None:
        print("defaulting to vocab_size of GPT-2 to 50304 (50257 rounded up for efficiency)")
    model_args['vocab_size'] = meta_vocab_size if meta_vocab_size is not None else 50304
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
    print(model)

    weights = tree_map(lambda p: p.astype(getattr(mx, d_type)), model.parameters())
    model.update(weights)

    mx.eval(model.parameters())
    nparams = sum(
        x.size for k, x in tree_flatten(model.parameters()) if "embedding" not in k
    )
    print(f"Training a transformer with {nparams / 1024**2:.3f} M parameters")


    def loss_fn(model, x, y, reduce=True):
        logits = model(x)
        losses = nn.losses.cross_entropy(
            logits.reshape(-1, logits.shape[-1]), y.reshape(-1)
        )
        return mx.mean(losses) if reduce else mx.mean(losses, axis=(-1, -2))


    # setup optimizer
    optimizer = AdamW(learning_rate=learning_rate, 
                            betas=[beta1, beta2], 
                            weight_decay=weight_decay)
    loss_and_grad_fn = nn.value_and_grad(model, loss_fn)


    def step(inputs, targets, gradient_accumulation_steps):
        # gradient accumulation
        accumulated_grads = tree_map(
                    lambda x: mx.zeros_like(x), model.parameters()
                )
        accumulated_loss = 0.0
        for micro_step in range(gradient_accumulation_steps):
            loss, grads = loss_and_grad_fn(model, X, Y)

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

        # scale the loss to account for gradient accumulation
        loss = mx.array(accumulated_loss / gradient_accumulation_steps) 

        optimizer.update(model, accumulated_grads)

        accumulated_grads = tree_map(
            lambda x: mx.zeros_like(x), model.parameters()
        )
        return loss

    # fetch the first batch of samples.
    X, Y = get_batch('train')
    
    state = [model.state, optimizer.state]

    tic = time.perf_counter()
    local_iter_num = 0 # number of iterations in the lifetime of this process
    iter_num = 0
    
    while True:
        if iter_num == 0 and eval_only:
            break

        # lr schedule
        new_lr = update_learning_rate(iter_num)
        optimizer.set_learning_rate(new_lr)

        # mx.simplify(loss, model.parameters())
        loss = step(X, Y, gradient_accumulation_steps)
        # immediately async prefetch next batch while model is doing the forward pass on the GPU
        X, Y = get_batch('train')

        tic = print_loss(optimizer, iter_num, loss.item(), tic)

        mx.eval(state)

        if iter_num % log_interval == 0:
            log_train_dict = {
                'loss': loss.item(),
                'lr': new_lr
            }
            log_tboard_dict(log_train_dict, iter_num, 'train')
        
        if iter_num % save_interval == 0:
            # save mode weights
            flat_params = tree_flatten(model.parameters())
            mx.savez(save_model_path, **dict(flat_params))
            # save model config
            with open(save_model_config_path, "w") as f:
                json.dump(model.config.__dict__, f)

        iter_num += 1
        local_iter_num += 1

        # termination conditions
        if iter_num > num_iters:
            break

if __name__ == "__main__":
    main()
