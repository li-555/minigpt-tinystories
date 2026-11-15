"""Small GPT, adapted from nanoGPT's MIT-licensed architecture (see LICENSE).

See README.md for training and evaluation commands.
"""
import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class GPTConfig:
    vocab_size: int = 8192
    block_size: int = 256
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 384
    dropout: float = 0.1
    attention_backend: str = "manual"

    def __post_init__(self):
        for name in ("vocab_size", "block_size", "n_layer", "n_head", "n_embd"):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.n_embd % self.n_head:
            raise ValueError("n_embd must be divisible by n_head")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in [0, 1)")
        if self.attention_backend != "manual":
            raise ValueError("unknown attention backend")


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.register_buffer("mask", torch.ones(config.block_size, config.block_size,
                                               dtype=torch.bool).tril(), persistent=False)

    def forward(self, x):
        """Apply explicit causal multi-head attention to [batch, time, channels]."""
        batch, length, channels = x.shape
        heads = self.config.n_head
        head_dim = channels // heads
        query, key, value = self.c_attn(x).split(channels, dim=-1)
        query = query.reshape(batch, length, heads, head_dim).transpose(1, 2)
        key = key.reshape(batch, length, heads, head_dim).transpose(1, 2)
        value = value.reshape(batch, length, heads, head_dim).transpose(1, 2)

        scores = (query @ key.transpose(-2, -1)) * (head_dim ** -0.5)
        scores = scores.masked_fill(~self.mask[:length, :length], float("-inf"))
        weights = self.attn_dropout(F.softmax(scores, dim=-1))
        attended = (weights @ value).transpose(1, 2).reshape(batch, length, channels)
        return self.resid_dropout(self.c_proj(attended))


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        """Expand and transform each token, then project back to model width."""
        hidden = F.gelu(self.c_fc(x))
        return self.dropout(self.c_proj(hidden))


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.RMSNorm(config.n_embd, eps=1e-5)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.RMSNorm(config.n_embd, eps=1e-5)
        self.mlp = MLP(config)

    def forward(self, x):
        """Apply attention and MLP in sequence, with pre-normalization."""
        x = x + self.attn(self.ln_1(x))
        return x + self.mlp(self.ln_2(x))


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict({
            "wte": nn.Embedding(config.vocab_size, config.n_embd),
            "wpe": nn.Embedding(config.block_size, config.n_embd),
            "drop": nn.Dropout(config.dropout),
            "h": nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            "ln_f": nn.RMSNorm(config.n_embd, eps=1e-5),
        })
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.lm_head.weight = self.transformer.wte.weight
        self.apply(self._init_weights)
        for name, parameter in self.named_parameters():
            if name.endswith("c_proj.weight"):
                nn.init.normal_(parameter, std=0.02 / math.sqrt(2 * config.n_layer))

    @staticmethod
    def _init_weights(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def parameter_count(self):
        return sum(p.numel() for p in self.parameters())

    def forward(self, idx, targets=None):
        if idx.ndim != 2 or not 0 < idx.shape[1] <= self.config.block_size:
            raise ValueError("input must be [batch, sequence] within block_size")
        pos = torch.arange(idx.shape[1], device=idx.device)
        x = self.transformer.drop(self.transformer.wte(idx) + self.transformer.wpe(pos))
        for block in self.transformer.h:
            x = block(x)
        logits = self.lm_head(self.transformer.ln_f(x))
        loss = None if targets is None else F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]).float(), targets.reshape(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None,
                 eos_id=None, generator=None):
        """Sample continuations while preserving the prompt and original model mode."""
        if idx.ndim != 2 or idx.numel() == 0:
            raise ValueError("idx must be a nonempty [batch, sequence] tensor")
        if type(max_new_tokens) is not int or max_new_tokens < 0:
            raise ValueError("max_new_tokens must be a nonnegative integer")
        if not math.isfinite(temperature) or temperature <= 0:
            raise ValueError("temperature must be finite and positive")
        if top_k is not None and (type(top_k) is not int or top_k <= 0):
            raise ValueError("top_k must be a positive integer")
        temperature = max(temperature, torch.finfo(torch.float32).tiny)

        was_training = self.training
        self.eval()
        try:
            finished = torch.zeros(idx.shape[0], dtype=torch.bool, device=idx.device)
            for _ in range(max_new_tokens):
                logits, _ = self(idx[:, -self.config.block_size:])
                next_logits = logits[:, -1, :].float()
                # Center before scaling so tiny temperatures cannot overflow upward.
                next_logits = next_logits - next_logits.amax(dim=-1, keepdim=True)
                if top_k is None:
                    probabilities = F.softmax(next_logits / temperature, dim=-1)
                    next_token = torch.multinomial(probabilities, 1, generator=generator)
                else:
                    values, ids = torch.topk(next_logits, min(top_k, next_logits.shape[-1]))
                    probabilities = F.softmax(values / temperature, dim=-1)
                    sampled = torch.multinomial(probabilities, 1, generator=generator)
                    next_token = ids.gather(1, sampled)
                if eos_id is not None:
                    next_token = next_token.masked_fill(finished[:, None], eos_id)
                    finished = next_token[:, 0].eq(eos_id)
                idx = torch.cat((idx, next_token), dim=1)
                if eos_id is not None and finished.all():
                    break
            return idx
        finally:
            self.train(was_training)
