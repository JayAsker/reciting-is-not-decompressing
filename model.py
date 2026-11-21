"""字节级 Transformer：压缩流(1024 槽) -> 明文(1024 槽 x 256 类) + 长度辅助头。

`copy=True` 时加 pointer-generator 拷贝头：每个输出位置可以「直接生成字节」或
「从输入流某个位置搬运字节」，两者由门控混合。
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from data import MAX_LEN, PAD, VOCAB


class Block(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.n_heads = n_heads
        self.ln1 = nn.LayerNorm(d_model)
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model), nn.GELU(), nn.Linear(4 * d_model, d_model)
        )

    def forward(self, x, mask):
        b, t, d = x.shape
        h = self.ln1(x)
        q, k, v = self.qkv(h).split(d, dim=2)
        q, k, v = (
            z.view(b, t, self.n_heads, d // self.n_heads).transpose(1, 2) for z in (q, k, v)
        )
        a = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        a = a.transpose(1, 2).reshape(b, t, d)
        x = x + self.proj(a)
        return x + self.mlp(self.ln2(x))


class ByteTransformer(nn.Module):
    def __init__(
        self, d_model=256, n_layers=3, n_heads=8, max_len=MAX_LEN, copy=False, copy_rel_only=False
    ):
        super().__init__()
        self.max_len = max_len
        self.copy = copy
        self.copy_rel_only = copy_rel_only
        self.tok = nn.Embedding(VOCAB, d_model, padding_idx=PAD)
        self.pos = nn.Embedding(max_len, d_model)
        self.blocks = nn.ModuleList(Block(d_model, n_heads) for _ in range(n_layers))
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, 256)
        self.len_head = nn.Sequential(
            nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, 1)
        )
        if copy:
            self.copy_gate = nn.Linear(d_model, 1)
            self.rel_window = 32
            self.rel_bias = nn.Parameter(torch.zeros(2 * self.rel_window + 1))
            if not copy_rel_only:
                self.copy_q = nn.Linear(d_model, d_model)
                self.copy_k = nn.Linear(d_model, d_model)

    def forward(self, inp, in_len):
        b, t = inp.shape
        pos = torch.arange(t, device=inp.device)
        x = self.tok(inp) + self.pos(pos)[None]
        keep_in = torch.arange(t, device=inp.device)[None] < in_len[:, None]
        keep = keep_in[:, None, None, :]
        for blk in self.blocks:
            x = blk(x, keep)
        x = self.norm(x)
        logp = F.log_softmax(self.head(x), dim=-1)
        if self.copy:
            ar = torch.arange(t, device=inp.device)
            rel_idx = (ar[:, None] - ar[None, :] + self.rel_window).clamp_(0, 2 * self.rel_window)
            if self.copy_rel_only:
                scores = self.rel_bias[rel_idx][None].expand(b, t, t)
            else:
                q = self.copy_q(x)
                k = self.copy_k(self.tok(inp))
                scores = q @ k.transpose(1, 2) / math.sqrt(q.shape[-1]) + self.rel_bias[rel_idx]
            scores = scores.masked_fill(~keep_in[:, None, :], float("-inf"))
            attn = scores.softmax(-1)
            idx = inp.clamp(max=255)[:, None, :].expand(b, t, -1)
            copy_p = torch.zeros(b, t, 256, device=inp.device, dtype=attn.dtype)
            copy_p.scatter_add_(2, idx, attn)
            gate = torch.sigmoid(self.copy_gate(x))
            mixed = gate * logp.exp() + (1.0 - gate) * copy_p
            logp = mixed.clamp_min(1e-9).log()
        pooled = (x * keep.squeeze(2).transpose(1, 2).to(x.dtype)).sum(1) / in_len[:, None].to(
            x.dtype
        )
        length = torch.sigmoid(self.len_head(pooled)).squeeze(-1) * self.max_len
        return logp, length


def n_params(model):
    return sum(p.numel() for p in model.parameters())
