import os
import argparse
import tiktoken
import time
import json

import mlx.core as mx
from mlx.utils import tree_unflatten, tree_flatten

from model import GPT, GPTConfig

import pdb

# -----------------------------------------------------------------------------
init_from = 'resume' # either 'resume' (from an out_dir) or a gpt2 variant (e.g. 'gpt2-xl')
out_dir = 'out' # ignored if init_from is not 'resume'
start = "\n" # or "<|endoftext|>" or etc. Can also specify a file, use as: "FILE:prompt.txt"
num_samples = 10 # number of samples to draw
max_new_tokens = 256 # number of tokens generated in each sample
temperature = 0.8 # 1.0 = no change, < 1.0 = less random, > 1.0 = more random, in predictions
top_k = 200 # retain only the top_k most likely tokens, clamp others to have 0 probability
seed = 1337
exec(open('configurator.py').read()) # overrides from command line or config file
# -----------------------------------------------------------------------------

model_weights_path = os.path.join(out_dir, out_dir + '.npz')
model_config_path = os.path.join(out_dir, out_dir + '.json')

# model
if init_from == 'resume':
    # init from a model saved in a specific directory
    with open(model_config_path, "r") as f:
        config_args = json.load(f)

    config = GPTConfig(**config_args)
    model = GPT(config)

    weights = mx.load(model_weights_path)
    model.update(tree_unflatten(list(weights.items())))
    mx.eval(model.parameters())

    nparams = sum(x.size for k, x in tree_flatten(model.parameters()))
    print(f"Loaded GPT-2 with {nparams / 1e6:.3f} M parameters")

elif init_from.startswith('gpt2'):
    # TODO
    print("Only mlx pre-trained models supported currently.")

# ok let's assume gpt-2 encodings by default
enc = tiktoken.get_encoding("gpt2")
encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
decode = lambda l: enc.decode(l)

# encode the beginning of the prompt
if start.startswith('FILE:'):
    with open(start[5:], 'r', encoding='utf-8') as f:
        start = f.read()
start_ids = encode(start)
x = (mx.array([start_ids], dtype=mx.uint32))

# run generation
start = time.time()
for k in range(num_samples):
    y = model.generate(x, max_new_tokens, temperature=temperature, top_k=top_k)
    print(decode(y[0].tolist()))
end = time.time()
