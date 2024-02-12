import math

import mlx.core as mx
import mlx.nn as nn

from dataclasses import dataclass

import pdb

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
            self.bias = bias
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
        f"""
        Initializes the Causal Self-Attention layer.

        Args:
            config (GPTConfig): An instance of the configuration class
                specifying the hyperparameters for the Causal Self-Attention layer.

        Attributes:
            c_attn (nn.Linear): Linear layer for key, query, and value projections.
            c_proj (nn.Linear): Linear layer for output projection.
            attn_dropout (nn.Dropout): Dropout layer for attention weights.
            resid_dropout (nn.Dropout): Dropout layer for residual connections.
            n_head (int): Number of attention heads.
            n_embd (int): Dimensionality of the embedding.
            dropout (float): Dropout probability.

        Notes:
            - The configuration class should contain the necessary hyperparameters for
              configuring the Causal Self-Attention layer.
            - The `c_attn` layer combines key, query, and value projections in a batch.
            - The `c_proj` layer handles the output projection.
            - The `attn_dropout` and `resid_dropout` layers apply dropout regularization.
            - The `n_head`, `n_embd`, and `dropout` attributes store hyperparameter values.
        """
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

    def __call__(self, x, mask, cache=None):
        B, T, C = x.shape # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        query, key, value = mx.split(self.c_attn(x), 3, axis=2)
        key = key.reshape(B, T, self.n_head, C // self.n_head).transpose(0, 2, 1, 3) # (B, nh, T, hs)
        query = query.reshape(B, T, self.n_head, C // self.n_head).transpose(0, 2, 1, 3) # (B, nh, T, hs)
        value = value.reshape(B, T, self.n_head, C // self.n_head).transpose(0, 2, 1, 3) # (B, nh, T, hs)

        if cache is not None:
            key_cache, value_cache = cache
            key = mx.concatenate([key_cache, key], axis=2)
            value = mx.concatenate([value_cache, value], axis=2)

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
        return y, (key, value)

    @staticmethod
    def create_additive_causal_mask(N: int, dtype: mx.Dtype = mx.float32):
        return mx.tril(mx.ones([N, N])).reshape(1, 1, N, N).astype(dtype)


class MLP(nn.Module):

    def __init__(self, config):
        f"""
        Initializes the Multi-Layer Perceptron (MLP) layer.

        Args:
            config (GPTConfig): An instance of the configuration class
                specifying the hyperparameters for the MLP layer.

        Attributes:
            c_fc (nn.Linear): Linear layer for fully connected transformations.
            gelu (nn.GELU): GELU activation function.
            c_proj (nn.Linear): Linear layer for output projection.
            dropout (nn.Dropout): Dropout layer for regularization.

        Notes:
            - Ensure that the `config` parameter is an instance of `MLPConfig`.
            - The configuration class should contain the necessary hyperparameters for
              configuring the MLP layer.
            - The `c_fc` layer performs a fully connected transformation.
            - The `gelu` layer applies the GELU activation function.
            - The `c_proj` layer handles the output projection.
            - The `dropout` layer applies dropout regularization.
        """
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
        """
        Initializes a block in a transformer architecture.

        Args:
            config (BlockConfig): An instance of the configuration class
                specifying the hyperparameters for the block.

        Attributes:
            ln_1 (LayerNorm): Layer normalization for the first sub-block.
            attn (CausalSelfAttention): Causal Self-Attention sub-block.
            ln_2 (LayerNorm): Layer normalization for the second sub-block.
            mlp (MLP): Multi-Layer Perceptron sub-block.

        Notes:
            - Ensure that the `config` parameter is an instance of `BlockConfig`.
            - The configuration class should contain the necessary hyperparameters for
              configuring the block.
            - The `ln_1` layer performs layer normalization for the first sub-block.
            - The `attn` layer represents the Causal Self-Attention sub-block.
            - The `ln_2` layer performs layer normalization for the second sub-block.
            - The `mlp` layer represents the Multi-Layer Perceptron sub-block.
        """
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def __call__(self, x, mask, cache=None):
        att, cache = self.attn(self.ln_1(x), mask, cache)
        x = x + att
        x = x + self.mlp(self.ln_2(x))
        return x, cache


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
        """
        Initializes a GPT (Generative Pre-trained Transformer) model.

        Args:
            config (GPTConfig): An instance of the configuration class
                specifying the hyperparameters for the GPT model.

        Attributes:
            config (GPTConfig): Configuration instance containing model hyperparameters.
            embedding (nn.Embedding): Embedding layer for input tokens.
            transformer (List[Block]): List of transformer blocks.
            out_proj (nn.Linear): Linear layer for output projection.
        """
        super().__init__()

        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config

        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.wpe = nn.Embedding(config.block_size, config.n_embd)
        self.drop = nn.Dropout(config.dropout)
        self.transformer = [Block(config) for _ in range(config.n_layer)]
        self.ln_f = LayerNorm(config.n_embd, bias=config.bias)
        self.out_proj = nn.Linear(config.n_embd, config.vocab_size, bias=False)

    def _sample_next_token(self, x, temperature):
        logits = mx.expand_dims(x[:, -1], axis=0) @ self.wte.weight.T
        y = logits[:, -1, :]
        y = mx.random.categorical(y * (1 / temperature))
        return y

    def generate(self, idx: mx.array, max_new_tokens=256, temperature=1.0, top_k=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        # Initialize the initial sequence context (idx)
        idx = mx.zeros((1, 1), dtype=mx.int64)
        
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            # idx_cond = idx if idx[0].shape[1] <= self.config.block_size else idx[:, -self.config.block_size:]
            idx_cond = idx if idx.shape[1] <= self.config.block_size else idx[:, -self.config.block_size:]

            # forward the model to get the logits for the index in the sequence
            logits = self(idx_cond)

            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature

            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = custom_topk(logits, min(top_k, logits.shape[-1]))

                v_shape = v.shape

                # Compute the index of the last element along the second dimension of v
                last_index = v_shape[1] - 1

                # Use MLX.take to extract the last element along the second dimension of v
                last_element = mx.take(v, mx.array([last_index]))

                # Expand the last element to match the shape of logits for broadcasting
                v_last_expanded = mx.expand_dims(last_element, axis=1)

                # Replace values with -1e9 where mask is True
                mask = logits < v_last_expanded
                inf_tensor = mx.ones_like(logits) * float('-1e9')
                logits = (mask * logits) + ((1 - mask) * inf_tensor)

            # apply softmax to convert logits to (normalized) probabilities
            probs = mx.softmax(logits)

            # Sample from the distribution
            idx_next = mx.random.categorical(probs, 1)

            # Append sampled index to the running sequence
            idx = mx.concatenate([idx, mx.expand_dims(idx_next, axis=0)], axis=1)

        return idx

    def _forward_transformer(
        self, x: mx.array, pos: mx.array, mask=None, cache=None, build_cache=False
    ):
        tok_emb = self.wte(x)
        pos_emb = self.wpe(pos)
        x = self.drop(tok_emb + pos_emb)
        kv_cache = []

        if cache is not None:
            for i in range(len(cache)):
                x, cache[i] = self.transformer[i](x, mask=None, cache=cache[i])
        else:
            for block in self.transformer:
                x, curr_cache = block(x, mask=mask)
                if build_cache:
                    kv_cache.append(curr_cache)

        x = self.ln_f(x)
        return x, kv_cache if build_cache else cache

    def __call__(self, x):
        b, t = x.shape
        assert (
            t <= self.config.block_size
        ), f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        pos = mx.arange(0, t, 1, dtype=x.dtype)
        mask = CausalSelfAttention.create_additive_causal_mask(x.shape[1])

        x, _ = self._forward_transformer(x, pos, mask=mask)
        return self.out_proj(x)
        

def custom_topk(input, k):
    """
    Custom implementation of top-k function in MLX.
    :param input: The input tensor.
    :param k: The number of elements to keep.
    :return: A tuple containing the top-k values and their indices.
    """
    # Flatten the input tensor along the last dimension
    flat_input = mx.reshape(input, (-1,))

    # Sort the flattened input tensor in descending order
    sorted_indices = mx.argsort(flat_input)
    sorted_indices = mx.take(sorted_indices, mx.arange(sorted_indices.size - 1, -1, -1))

    # Slice the sorted indices to get the top-k indices
    topk_indices = custom_slice(sorted_indices, start=0, end=k)
    
    # Gather the top-k values using the top-k indices
    topk_values = mx.take(flat_input, topk_indices)
    
    return mx.expand_dims(topk_values, axis=0), mx.expand_dims(topk_indices,  axis=0)


def custom_slice(indices, start, end):
    """
    Custom implementation of slice operation for indices.
    :param indices: The sorted indices tensor.
    :param start: The starting index for slicing.
    :param end: The ending index for slicing.
    :return: The sliced indices tensor.
    """
    # Take the range of indices from start to end
    sliced_indices = mx.take(indices, mx.arange(start, end))
    return sliced_indices