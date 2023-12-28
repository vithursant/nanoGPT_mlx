# train a miniature character-level shakespeare model
# good for debugging and playing on macbooks and such
dataset = 'shakespeare'
gradient_accumulation_steps = 1
batch_size = 4
block_size = 256 # context of up to 256 previous characters

learning_rate = 2e-3 # with baby networks can afford to go a bit higher
num_iters = 5000
min_lr = 1e-4 # learning_rate / 10 usually
beta2 = 0.99 # make a bit bigger because number of tokens per iter is small

# baby GPT model :)
n_layer = 6
n_head = 6
n_embd = 384
dropout = 0.2

# eval stuff
eval_interval = 250 # keep frequent because we'll overfit
eval_iters = 200
log_interval = 10 # don't print too too often

