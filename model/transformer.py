import torch
import torch.nn as nn
import torch.nn.functional as F
from parallel.tensor_parallel import ColumnParallelLinear, RowParallelLinear
from parallel.process_groups import get_model_parallel


class Attention(nn.Module):
    """Multi-head self-attention with optional causal mask."""

    def __init__(self, hidden_size, num_heads):
        super().__init__()
        mpu = get_model_parallel()
        tp_size = mpu["tp_size"] if mpu else 1
        self.num_heads = num_heads // tp_size
        self.head_dim = hidden_size // num_heads
        self.hidden_size_per_partition = hidden_size // tp_size

        self.qkv = ColumnParallelLinear(hidden_size, 3 * hidden_size, bias=False)
        self.o = RowParallelLinear(hidden_size, hidden_size, bias=False)

    def forward(self, x):
        B, S, _ = x.shape
        hp = self.hidden_size_per_partition
        qkv = self.qkv(x).reshape(B, S, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(2)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        out = out.transpose(1, 2).reshape(B, S, hp)
        return self.o(out)


class MLP(nn.Module):
    def __init__(self, hidden_size, ffn_hidden_size):
        super().__init__()
        mpu = get_model_parallel()
        tp_size = mpu["tp_size"] if mpu else 1
        self.fc1 = ColumnParallelLinear(hidden_size, ffn_hidden_size, bias=False)
        self.fc2 = RowParallelLinear(ffn_hidden_size, hidden_size, bias=False)

    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x)))


class DecoderLayer(nn.Module):
    """Transformer decoder layer: attention + MLP with residual connections."""

    def __init__(self, hidden_size, num_heads, ffn_hidden_size):
        super().__init__()
        self.ln1 = nn.LayerNorm(hidden_size)
        self.attn = Attention(hidden_size, num_heads)
        self.ln2 = nn.LayerNorm(hidden_size)
        self.mlp = MLP(hidden_size, ffn_hidden_size)

    def _forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

    def forward(self, x, use_checkpoint=False):
        if use_checkpoint:
            return torch.utils.checkpoint.checkpoint(self._forward, x, use_reentrant=False)
        return self._forward(x)


class Decoder(nn.Module):
    """Stack of DecoderLayer blocks."""

    def __init__(self, hidden_size, num_heads, ffn_hidden_size, num_layers, layer_ids=None):
        super().__init__()
        if layer_ids is None:
            self.layer_ids = list(range(num_layers))
        else:
            self.layer_ids = layer_ids
        self.layers = nn.ModuleList([
            DecoderLayer(hidden_size, num_heads, ffn_hidden_size)
            for _ in self.layer_ids
        ])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class GPT(nn.Module):
    """Decoder with embedding, optional LM head, and optional loss computation."""

    def __init__(self, embedding, decoder, ln_f, lm_head, loss_fn=None):
        super().__init__()
        self.embedding = embedding
        self.decoder = decoder
        self.ln_f = ln_f
        self.lm_head = lm_head
        self.loss_fn = loss_fn

    def forward(self, input_ids, labels=None, loss_mask=None):
        x = self.embedding(input_ids)
        x = self.decoder(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        if labels is not None and loss_mask is not None and self.loss_fn is not None:
            loss = self.loss_fn(logits, labels, loss_mask)
            return logits, loss
        return logits
