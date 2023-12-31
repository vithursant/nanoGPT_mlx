# config for training GPT-2 (124M) on OWT

# these make the total batch size be ~0.5M
# 1 batch size * 1024 block size * 512 gradaccum * 1 M3 Pro = 524,288
batch_size = 1
block_size = 1024
gradient_accumulation_steps = 512

# this makes total number of tokens be 300B
num_iters = 4496

# eval stuff
eval_interval = 250
eval_iters = 200
log_interval = 10

# weight decay
weight_decay = 1e-1
