#!/usr/bin/env python3
"""TRM and STRM on Sudoku-Extreme, Maze-Hard and PPBench -- one file.

TRM (Tiny Recursive Model, Jolicoeur-Martineau 2025) keeps an answer state z_H and a latent state z_L. One
deep-supervision segment runs H_cycles cycles of L_cycles latent updates and one answer update,

    z_L <- net(z_L, z_H + x)   (L_cycles times),    z_H <- net(z_H, z_L),

the first H_cycles - 1 cycles without gradient; the heads read z_H.

STRM (state-only multi-timescale model) keeps a single state s, a bank of leaky integrators with one fixed rate
a_d per channel, log-spaced from 1 (fast) to rate_min (slow). One network call is

    s <- (1 - a) * s + a * net(rms_norm(s) + x),

a segment runs `cycles` calls, the last `grad_cycles` with gradient. The heads read only the slow channels of s
(nominal rate <= readout_max_rate), rms-normalized among themselves. In training, every trajectory draws a
stretch factor c ~ logU[1/J, J] and uses the rates a_d ** c (rate jitter, J = rate_jitter); at evaluation c = 1.

Both models share everything else: the network (two post-norm blocks: token mixing -- an MLP over the sequence
for Sudoku, self-attention with RoPE otherwise -- then a SwiGLU MLP), 16 register tokens, deep supervision with
ACT halting (up to 16 segments, 10% exploration), stablemax cross-entropy + 0.5 * halting BCE, AdamW, EMA 0.999.
At evaluation every puzzle runs all 16 segments.

Usage:
    python train.py --task sudoku --model strm --data-dir data/sudoku-extreme-1k-aug-1000
    python train.py --task maze --model trm --data-dir data/maze-30x30-hard-1k-aug-8-relabeled --seed 1
    python train.py --task ppbench --model strm --data-dir data/ppbench-6type-10x10-aug-10 --eval-only \
        --init-from checkpoints/ppbench_strm/step_242300.pt
Every setting of the presets below can be overridden on the command line (e.g. --lr 1e-4).
"""
from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import time
from dataclasses import dataclass, fields
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

IGNORE_LABEL_ID = -100
torch.backends.cuda.enable_flash_sdp(False)  # as the TRM code (errors for some sequence lengths)


# ---------------------------------------------------------------------------------------
# Config and the paper's settings
# ---------------------------------------------------------------------------------------
@dataclass
class Config:
    task: str = "sudoku"           # sudoku | maze | ppbench
    model: str = "strm"            # trm | strm
    data_dir: str = ""

    # Network
    mixer: str = "attn"            # token mixing: mlp (over the sequence axis) | attn (self-attention + RoPE)
    hidden_size: int = 512
    expansion: float = 4.0
    num_layers: int = 2
    num_heads: int = 8
    rope_theta: float = 10000.0
    puzzle_emb_len: int = 16       # register tokens (one learned vector in register 0, the rest zero)
    rms_norm_eps: float = 1e-5

    # TRM recursion
    H_cycles: int = 3
    L_cycles: int = 4

    # STRM recursion
    cycles: int = 15               # net calls per segment (= H_cycles * (L_cycles + 1) of TRM)
    grad_cycles: int = 5           # last calls with gradient (= L_cycles + 1 of TRM)
    rate_min: float = 1 / 13       # slowest rate; rates are log-spaced from 1 to rate_min over the channels
    readout_max_rate: float = 13 ** -0.5  # the heads read the channels with rate <= this (sqrt(rate_min))
    rate_jitter: float = 1.5       # J: per-trajectory rate exponent c ~ logU[1/J, J] in training (1 = off)

    # Deep supervision / ACT
    halt_max_steps: int = 16
    halt_exploration_prob: float = 0.1

    # Optimization
    steps: int = 65100
    global_batch_size: int = 768
    micro_batch_size: int = 384    # gradients accumulate over global / micro batches (one ACT carry each)
    lr: float = 1e-4
    lr_warmup_steps: int = 2000
    lr_min_ratio: float = 1.0      # after warmup: cosine decay to lr_min_ratio * lr (1 = constant)
    beta1: float = 0.9
    beta2: float = 0.95
    weight_decay: float = 1.0      # on all parameters with >= 2 dims
    grad_clip: float = 1.0
    ema_rate: float = 0.999

    # Evaluation / run
    eval_every: int = 3255         # optimizer steps between evaluations (each saves an EMA checkpoint)
    eval_examples: int = 0         # puzzles per split at intermediate evaluations (0 = all; the last one: all)
    seed: int = 0
    checkpoint_dir: str = ""
    init_from: str = ""            # load these (EMA) weights before training / evaluation
    eval_only: bool = False        # with --init-from: evaluate on every split and exit
    log_every: int = 100
    compile: bool = True

    # Filled in from the dataset
    vocab_size: int = 0
    seq_len: int = 0

    @property
    def seq_total(self) -> int:
        return self.seq_len + self.puzzle_emb_len


# Settings of the paper. TRM: H 3 x L 6 (Sudoku) / H 3 x L 4 (attention tasks); STRM: the same number of network
# calls per segment and of calls with gradient.
PRESETS = {
    ("sudoku", "trm"): dict(mixer="mlp", H_cycles=3, L_cycles=6, lr=1e-4, lr_min_ratio=0.0, weight_decay=1.0,
                            steps=65104, eval_every=3255, eval_examples=10000),
    ("sudoku", "strm"): dict(mixer="mlp", cycles=21, grad_cycles=7, rate_min=1 / 49, readout_max_rate=1 / 7,
                             lr=1.5e-4, lr_min_ratio=0.1, weight_decay=0.7,
                             steps=65104, eval_every=3255, eval_examples=10000),
    ("maze", "trm"): dict(mixer="attn", H_cycles=3, L_cycles=4, lr=1e-4, weight_decay=1.0,
                          steps=65100, eval_every=3255, micro_batch_size=64),
    ("maze", "strm"): dict(mixer="attn", cycles=15, grad_cycles=5, lr=1.5e-4, weight_decay=0.7,
                           steps=65100, eval_every=3255, micro_batch_size=64),
    ("ppbench", "trm"): dict(mixer="attn", H_cycles=3, L_cycles=4, lr=1e-4, weight_decay=1.0,
                             steps=242300, eval_every=12115),
    ("ppbench", "strm"): dict(mixer="attn", cycles=15, grad_cycles=5, lr=1.5e-4, weight_decay=0.7,
                              steps=242300, eval_every=12115),
}


def parse_args() -> Config:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for f in fields(Config):
        if f.name in ("vocab_size", "seq_len"):
            continue
        flag = "--" + f.name.replace("_", "-")
        if isinstance(f.default, bool):
            p.add_argument(flag, dest=f.name, default=None, action=argparse.BooleanOptionalAction)
        else:
            p.add_argument(flag, dest=f.name, type=type(f.default), default=None)
    args = {k: v for k, v in vars(p.parse_args()).items() if v is not None}
    task, model = args.get("task", Config.task), args.get("model", Config.model)
    assert (task, model) in PRESETS, f"unknown task/model {task}/{model}"
    cfg = Config(**{**PRESETS[(task, model)], **args})
    cfg.data_dir = cfg.data_dir or f"data/{task}"
    cfg.checkpoint_dir = cfg.checkpoint_dir or f"checkpoints/{task}_{model}_seed{cfg.seed}"
    assert cfg.global_batch_size % cfg.micro_batch_size == 0
    assert 1 <= cfg.grad_cycles <= cfg.cycles and cfg.rate_jitter >= 1.0
    assert 0 < cfg.rate_min <= cfg.readout_max_rate <= 1.0
    return cfg


# ---------------------------------------------------------------------------------------
# Data: the TRM PuzzleDataset layout (<split>/all__{inputs,labels,group_indices}.npy, dataset.json)
# ---------------------------------------------------------------------------------------
def load_split(data_dir: str, split: str) -> Dict[str, np.ndarray]:
    return {k: np.load(os.path.join(data_dir, split, f"all__{k}.npy"), mmap_mode="r")
            for k in ("inputs", "labels", "group_indices")}


def to_batch(inputs: np.ndarray, labels: np.ndarray, device) -> Dict[str, torch.Tensor]:
    x = torch.from_numpy(np.ascontiguousarray(inputs, dtype=np.int32)).to(device, non_blocking=True)
    y = torch.from_numpy(np.ascontiguousarray(labels, dtype=np.int32)).to(device, non_blocking=True)
    return {"inputs": x, "labels": torch.where(y == 0, IGNORE_LABEL_ID, y)}  # label 0 is PAD


def train_batches(data: Dict[str, np.ndarray], batch_size: int, seed: int, device):
    """Infinite stream: the groups (a puzzle and its augmentations) are shuffled once per epoch and one random
    member of each is drawn."""
    gi = np.asarray(data["group_indices"])
    rng = np.random.default_rng(seed)
    buf = np.empty(0, dtype=np.int64)
    while True:
        while buf.size < batch_size:
            buf = np.concatenate([buf, rng.permutation(gi.size - 1)])
        gids, buf = buf[:batch_size], buf[batch_size:]
        idx = gi[gids] + (rng.random(batch_size) * (gi[gids + 1] - gi[gids])).astype(np.int64)
        idx.sort()  # a sorted gather is friendlier to the memmap
        yield to_batch(data["inputs"][idx], data["labels"][idx], device)


def eval_batches(data: Dict[str, np.ndarray], batch_size: int, device, max_examples: int = 0):
    """Fixed-size chunks over a split (the first max_examples rows, rounded up to whole batches; 0 = all);
    the last chunk is padded with all-PAD rows, which are not scored."""
    n = data["inputs"].shape[0]
    if max_examples:
        n = min(n, -(-max_examples // batch_size) * batch_size)
    for start in range(0, n, batch_size):
        x = np.array(data["inputs"][start:start + batch_size], dtype=np.int32)
        y = np.array(data["labels"][start:start + batch_size], dtype=np.int32)
        pad = ((0, batch_size - x.shape[0]), (0, 0))
        yield to_batch(np.pad(x, pad), np.pad(y, pad), device)


def token_types(data_dir: str, vocab_size: int) -> Tuple[List[str], np.ndarray]:
    """PPBench: puzzle type of every token id ("<type>:<cell>" in vocab.json; 1-based, 0 = PAD)."""
    path = os.path.join(data_dir, "vocab.json")
    if not os.path.exists(path):
        return [], np.zeros(vocab_size, dtype=np.int64)
    with open(path) as f:
        vocab = json.load(f)
    names = sorted({tok.split(":")[0] for tok in vocab})
    types = np.zeros(vocab_size, dtype=np.int64)
    for tok, tid in vocab.items():
        types[tid] = names.index(tok.split(":")[0]) + 1
    return names, types


# ---------------------------------------------------------------------------------------
# Network
# ---------------------------------------------------------------------------------------
def trunc_normal_init_(tensor: torch.Tensor, std: float = 1.0, lower: float = -2.0, upper: float = 2.0):
    """JAX-style truncated normal (PyTorch's version does not preserve the std)."""
    with torch.no_grad():
        if std == 0:
            return tensor.zero_()
        sqrt2 = math.sqrt(2)
        a, b = math.erf(lower / sqrt2), math.erf(upper / sqrt2)
        z = (b - a) / 2
        c = (2 * math.pi) ** -0.5
        pdf_u, pdf_l = c * math.exp(-0.5 * lower ** 2), c * math.exp(-0.5 * upper ** 2)
        comp_std = std / math.sqrt(1 - (upper * pdf_u - lower * pdf_l) / z - ((pdf_u - pdf_l) / z) ** 2)
        tensor.uniform_(a, b).erfinv_().mul_(sqrt2 * comp_std).clip_(lower * comp_std, upper * comp_std)
    return tensor


class CastedLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = False):
        super().__init__()
        self.weight = nn.Parameter(trunc_normal_init_(torch.empty(out_features, in_features), std=in_features ** -0.5))
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None

    def forward(self, x):
        return F.linear(x, self.weight.to(x.dtype), None if self.bias is None else self.bias.to(x.dtype))


class CastedEmbedding(nn.Module):
    def __init__(self, num: int, dim: int, init_std: float):
        super().__init__()
        self.embedding_weight = nn.Parameter(trunc_normal_init_(torch.empty(num, dim), std=init_std))

    def forward(self, x):
        return F.embedding(x, self.embedding_weight.to(torch.bfloat16))


def rms_norm(x: torch.Tensor, eps: float) -> torch.Tensor:
    return (x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + eps)).to(x.dtype)


class SwiGLU(nn.Module):
    def __init__(self, dim: int, expansion: float):
        super().__init__()
        inter = -(round(expansion * dim * 2 / 3) // -256) * 256  # rounded up to a multiple of 256
        self.gate_up_proj = CastedLinear(dim, 2 * inter)
        self.down_proj = CastedLinear(inter, dim)

    def forward(self, x):
        gate, up = self.gate_up_proj(x).chunk(2, dim=-1)
        return self.down_proj(F.silu(gate) * up)


def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


class Attention(nn.Module):
    """Non-causal multi-head self-attention, 1D RoPE (applied in float32), no bias."""

    def __init__(self, cfg: Config):
        super().__init__()
        self.num_heads, self.head_dim = cfg.num_heads, cfg.hidden_size // cfg.num_heads
        self.qkv_proj = CastedLinear(cfg.hidden_size, 3 * cfg.hidden_size)
        self.o_proj = CastedLinear(cfg.hidden_size, cfg.hidden_size)

    def forward(self, h, cos, sin):
        B, L, D = h.shape
        q, k, v = self.qkv_proj(h).view(B, L, 3 * self.num_heads, self.head_dim).split(self.num_heads, dim=2)
        q, k = ((t.float() * cos[:, None] + rotate_half(t.float()) * sin[:, None]).to(h.dtype) for t in (q, k))
        out = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2))
        return self.o_proj(out.transpose(1, 2).reshape(B, L, D))


class Block(nn.Module):
    """Post-norm: h = rms(h + mix(h)), h = rms(h + mlp(h)); mix over the sequence axis (MLP) or attention."""

    def __init__(self, cfg: Config):
        super().__init__()
        if cfg.mixer == "mlp":
            self.mlp_t = SwiGLU(cfg.seq_total, cfg.expansion)
        else:
            self.self_attn = Attention(cfg)
        self.mlp = SwiGLU(cfg.hidden_size, cfg.expansion)
        self.eps = cfg.rms_norm_eps

    def forward(self, h, cos_sin):
        if hasattr(self, "mlp_t"):
            h = h.transpose(1, 2)
            h = rms_norm(h + self.mlp_t(h), self.eps).transpose(1, 2)
        else:
            h = rms_norm(h + self.self_attn(h, *cos_sin), self.eps)
        return rms_norm(h + self.mlp(h), self.eps)


class Net(nn.Module):
    """The single shared network: net(h, u) = blocks(h + u)."""

    def __init__(self, cfg: Config):
        super().__init__()
        self.layers = nn.ModuleList([Block(cfg) for _ in range(cfg.num_layers)])
        dim = cfg.hidden_size // cfg.num_heads
        inv_freq = 1.0 / (cfg.rope_theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        freqs = torch.outer(torch.arange(cfg.seq_total, dtype=torch.float32), inv_freq)
        self.cos = nn.Buffer(torch.cat((freqs, freqs), -1).cos(), persistent=False)
        self.sin = nn.Buffer(torch.cat((freqs, freqs), -1).sin(), persistent=False)

    def forward(self, h, u=None):
        h = h if u is None else h + u
        for layer in self.layers:
            h = layer(h, (self.cos, self.sin))
        return h


# ---------------------------------------------------------------------------------------
# Models: one deep-supervision segment each, with ACT around it
# ---------------------------------------------------------------------------------------
class RecursiveModel(nn.Module):
    """Embeddings, heads and ACT; subclasses define the state, its reset and one segment."""

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        D = cfg.hidden_size
        self.embed_tokens = CastedEmbedding(cfg.vocab_size, D, init_std=D ** -0.5)
        self.lm_head = CastedLinear(D, cfg.vocab_size)
        self.q_head = CastedLinear(D, 1, bias=True)
        self.puzzle_emb = CastedEmbedding(1, D, init_std=0)  # one learned vector (all puzzles share it)
        self.net = Net(cfg)
        with torch.no_grad():  # (almost) zero halting head, for faster bootstrapping
            self.q_head.weight.zero_()
            self.q_head.bias.fill_(-5)

    def embed(self, inputs):
        emb = self.embed_tokens(inputs)
        pe = F.pad(self.puzzle_emb.embedding_weight.to(torch.bfloat16)[0],
                   (0, (self.cfg.puzzle_emb_len - 1) * self.cfg.hidden_size))
        pe = pe.view(1, self.cfg.puzzle_emb_len, -1).expand(emb.shape[0], -1, -1)
        return math.sqrt(self.cfg.hidden_size) * torch.cat((pe, emb), dim=-2)

    def heads(self, r):
        return self.lm_head(r)[:, self.cfg.puzzle_emb_len:], self.q_head(r[:, 0]).float()[..., 0]

    def initial_carry(self, batch):
        B = batch["inputs"].shape[0]
        dev = batch["inputs"].device
        return dict(state=self.empty_state(B, dev), steps=torch.zeros(B, dtype=torch.int32, device=dev),
                    halted=torch.ones(B, dtype=torch.bool, device=dev),  # every slot takes a puzzle first
                    data={k: torch.empty_like(v) for k, v in batch.items()})

    def forward(self, carry, batch):
        halted = carry["halted"]
        state = self.reset_state(carry["state"], halted)
        steps = torch.where(halted, 0, carry["steps"])
        data = {k: torch.where(halted.view(-1, 1), batch[k], v) for k, v in carry["data"].items()}

        state, logits, q_halt = self.segment(state, self.embed(data["inputs"]))

        with torch.no_grad():
            steps = steps + 1
            halted = steps >= self.cfg.halt_max_steps
            if self.training:  # halt when the head predicts a correct answer, with random minimum segments
                halted = halted | (q_halt > 0)
                min_steps = ((torch.rand_like(q_halt) < self.cfg.halt_exploration_prob)
                             * torch.randint_like(steps, low=2, high=self.cfg.halt_max_steps + 1))
                halted = halted & (steps >= min_steps)
        return dict(state=state, steps=steps, halted=halted, data=data), logits, q_halt


class TRM(RecursiveModel):
    def __init__(self, cfg: Config):
        super().__init__(cfg)
        self.H_init = nn.Buffer(trunc_normal_init_(torch.empty(cfg.hidden_size, dtype=torch.bfloat16)))
        self.L_init = nn.Buffer(trunc_normal_init_(torch.empty(cfg.hidden_size, dtype=torch.bfloat16)))

    def empty_state(self, B, dev):
        shape = (B, self.cfg.seq_total, self.cfg.hidden_size)
        return torch.empty(shape, dtype=torch.bfloat16, device=dev), torch.empty(shape, dtype=torch.bfloat16, device=dev)

    def reset_state(self, state, halted):
        r = halted.view(-1, 1, 1)
        return torch.where(r, self.H_init, state[0]), torch.where(r, self.L_init, state[1])

    def segment(self, state, x):
        z_H, z_L = state
        with torch.no_grad():
            for _ in range(self.cfg.H_cycles - 1):
                for _ in range(self.cfg.L_cycles):
                    z_L = self.net(z_L, z_H + x)
                z_H = self.net(z_H, z_L)
        for _ in range(self.cfg.L_cycles):
            z_L = self.net(z_L, z_H + x)
        z_H = self.net(z_H, z_L)
        return (z_H.detach(), z_L.detach()), *self.heads(z_H)


class STRM(RecursiveModel):
    def __init__(self, cfg: Config):
        super().__init__(cfg)
        self.s_init = nn.Buffer(trunc_normal_init_(torch.empty(cfg.hidden_size, dtype=torch.bfloat16)))
        rates = torch.logspace(0, math.log10(cfg.rate_min), cfg.hidden_size, dtype=torch.float64, device="cpu")
        self.rates = nn.Buffer(rates.float().to(self.s_init.device))                  # fast -> slow over channels
        self.readout_mask = nn.Buffer((self.rates <= cfg.readout_max_rate * (1 + 1e-6)).float())

    def empty_state(self, B, dev):
        return (torch.empty(B, self.cfg.seq_total, self.cfg.hidden_size, device=dev),  # s, float32
                torch.ones(B, device=dev))                                            # rate exponent c

    def reset_state(self, state, halted):
        s, c = state
        s = torch.where(halted.view(-1, 1, 1), self.s_init.float(), s)
        if self.training and self.cfg.rate_jitter > 1:  # a new trajectory draws c ~ logU[1/J, J]
            c_new = torch.exp((2 * torch.rand_like(c) - 1) * math.log(self.cfg.rate_jitter))
        else:
            c_new = torch.ones_like(c)
        return s, torch.where(halted, c_new, c)

    def segment(self, state, x):
        s, c = state
        a = self.rates.view(1, 1, -1) ** c.view(-1, 1, 1)

        def calls(s, n):
            for _ in range(n):
                z = self.net(rms_norm(s, self.cfg.rms_norm_eps).to(torch.bfloat16) + x)
                s = (1 - a) * s + a * z.float()
            return s

        with torch.no_grad():
            s = calls(s, self.cfg.cycles - self.cfg.grad_cycles)
        s = calls(s, self.cfg.grad_cycles)

        m = self.readout_mask  # slow-only readout: rms over the slow channels, the fast ones zeroed
        r = s * m * torch.rsqrt((s.square() * m).sum(-1, keepdim=True) / m.sum() + self.cfg.rms_norm_eps)
        return (s.detach(), c), *self.heads(r.to(torch.bfloat16))


# ---------------------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------------------
def stablemax_cross_entropy(logits, labels, mask):
    x = logits.double()
    s = torch.where(x < 0, 1 / (1 - x + 1e-30), x + 1)
    logprobs = torch.log(s / s.sum(-1, keepdim=True))
    picked = torch.gather(logprobs, -1, torch.where(mask, labels, 0).long().unsqueeze(-1)).squeeze(-1)
    return -torch.where(mask, picked, 0)


class Loss(nn.Module):
    """Per segment: stablemax CE of the answer (mean over a puzzle's cells, summed over puzzles) + 0.5 * BCE of the
    halting logit against exact correctness. Metrics count the puzzles that halted in this segment."""

    def __init__(self, model: RecursiveModel):
        super().__init__()
        self.model = model

    def initial_carry(self, batch):
        return self.model.initial_carry(batch)

    def forward(self, carry, batch):
        carry, logits, q_halt = self.model(carry, batch)
        labels = carry["data"]["labels"]
        with torch.no_grad():
            mask = labels != IGNORE_LABEL_ID
            counts = mask.sum(-1)
            is_correct = mask & (logits.argmax(-1) == labels)
            seq_correct = is_correct.sum(-1) == counts
            valid = carry["halted"] & (counts > 0)
            metrics = {"count": valid.sum(),
                       "accuracy": torch.where(valid, is_correct.sum(-1) / counts.clamp_min(1), 0).sum(),
                       "exact_accuracy": (valid & seq_correct).sum(),
                       "q_halt_accuracy": (valid & ((q_halt >= 0) == seq_correct)).sum(),
                       "steps": torch.where(valid, carry["steps"], 0).sum()}
        lm_loss = (stablemax_cross_entropy(logits, labels, mask) / counts.clamp_min(1).unsqueeze(-1)).sum()
        q_loss = F.binary_cross_entropy_with_logits(q_halt, seq_correct.to(q_halt.dtype), reduction="sum")
        metrics.update(lm_loss=lm_loss.detach(), q_halt_loss=q_loss.detach())
        return carry, lm_loss + 0.5 * q_loss, metrics, valid & seq_correct


# ---------------------------------------------------------------------------------------
# Optimization
# ---------------------------------------------------------------------------------------
class EMA:
    def __init__(self, model: nn.Module, rate: float):
        self.rate = rate
        self.shadow = {n: p.detach().clone() for n, p in model.named_parameters()}

    @torch.no_grad()
    def update(self, model):
        for n, p in model.named_parameters():
            self.shadow[n].lerp_(p, 1 - self.rate)

    @contextlib.contextmanager
    def applied(self, model):
        backup = {n: p.detach().clone() for n, p in model.named_parameters()}
        with torch.no_grad():
            for n, p in model.named_parameters():
                p.copy_(self.shadow[n])
        try:
            yield
        finally:
            with torch.no_grad():
                for n, p in model.named_parameters():
                    p.copy_(backup[n])


def lr_at(step: int, cfg: Config) -> float:
    """Linear warmup, then cosine decay to lr_min_ratio * lr."""
    if step < cfg.lr_warmup_steps:
        return cfg.lr * step / cfg.lr_warmup_steps
    progress = (step - cfg.lr_warmup_steps) / max(1, cfg.steps - cfg.lr_warmup_steps)
    return cfg.lr * (cfg.lr_min_ratio + (1 - cfg.lr_min_ratio) * 0.5 * (1 + math.cos(math.pi * progress)))


# ---------------------------------------------------------------------------------------
# Train / eval
# ---------------------------------------------------------------------------------------
@torch.inference_mode()
def evaluate(model, data, cfg: Config, device, type_of_token, type_names, max_examples=0) -> Dict[str, float]:
    """Exact accuracy (and the other metrics) after all halt_max_steps segments; per puzzle type for PPBench."""
    model.eval()
    totals: Dict[str, float] = {}
    per_type = torch.zeros(2, len(type_names) + 1, device=device)
    for batch in eval_batches(data, cfg.micro_batch_size, device, max_examples):
        carry = model.initial_carry(batch)
        for _ in range(cfg.halt_max_steps):
            carry, _, metrics, correct = model(carry, batch)
        for k, v in metrics.items():
            totals[k] = totals.get(k, 0.0) + float(v)
        types = type_of_token[batch["inputs"].long()].amax(-1)  # 0 for padded rows
        per_type[0] += torch.bincount(types, weights=correct.float(), minlength=per_type.shape[1])
        per_type[1] += torch.bincount(types, minlength=per_type.shape[1]).float()
    model.train()
    count = max(totals.pop("count"), 1)
    out = {k: v / count for k, v in totals.items()} | {"puzzles": count}
    for i, name in enumerate(type_names, start=1):
        out[f"exact_{name}"] = float(per_type[0, i] / per_type[1, i].clamp_min(1))
    return out


def main():
    cfg = parse_args()
    device = torch.device("cuda")
    torch.manual_seed(cfg.seed)

    with open(os.path.join(cfg.data_dir, "train", "dataset.json")) as f:
        meta = json.load(f)
    cfg.vocab_size, cfg.seq_len = int(meta["vocab_size"]), int(meta["seq_len"])
    train_data = load_split(cfg.data_dir, "train")
    eval_sets = {s: load_split(cfg.data_dir, s) for s in ("val", "test", "golden")
                 if os.path.exists(os.path.join(cfg.data_dir, s, "dataset.json"))}
    type_names, types = token_types(cfg.data_dir, cfg.vocab_size)
    type_of_token = torch.from_numpy(types).to(device)

    with torch.device(device):
        raw = Loss((TRM if cfg.model == "trm" else STRM)(cfg))
    if cfg.init_from:
        raw.load_state_dict(torch.load(cfg.init_from, map_location=device))
    model = torch.compile(raw) if cfg.compile else raw
    print(f"[{cfg.task} / {cfg.model}] {sum(p.numel() for p in raw.parameters()) / 1e6:.2f}M params | {cfg}", flush=True)

    def run_eval(step, final):
        for split, data in eval_sets.items():
            m = evaluate(model, data, cfg, device, type_of_token, type_names, 0 if final else cfg.eval_examples)
            print(f"[eval {split} @ step {step}] " + " | ".join(f"{k} {v:.4f}" for k, v in m.items()), flush=True)

    if cfg.eval_only:
        return run_eval(0, final=True)

    params = list(raw.parameters())
    optimizer = torch.optim.AdamW([{"params": [p for p in params if p.dim() >= 2], "weight_decay": cfg.weight_decay},
                                   {"params": [p for p in params if p.dim() < 2], "weight_decay": 0.0}],
                                  lr=cfg.lr, betas=(cfg.beta1, cfg.beta2))
    ema = EMA(raw, cfg.ema_rate)
    accum = cfg.global_batch_size // cfg.micro_batch_size
    batches = train_batches(train_data, cfg.micro_batch_size, cfg.seed, device)
    carries = [None] * accum  # one persistent ACT stream per micro-batch slot
    running: Dict[str, torch.Tensor] = {}
    t0 = time.time()
    model.train()

    for step in range(1, cfg.steps + 1):
        for slot in range(accum):
            batch = next(batches)
            carries[slot] = carries[slot] or model.initial_carry(batch)
            carries[slot], loss, metrics, _ = model(carries[slot], batch)
            ((1 / cfg.global_batch_size) * loss).backward()
            for k, v in metrics.items():
                running[k] = running.get(k, 0) + v.float()
        grad_norm = torch.nn.utils.clip_grad_norm_(params, cfg.grad_clip, error_if_nonfinite=True)
        for group in optimizer.param_groups:
            group["lr"] = lr_at(step, cfg)
        optimizer.step()
        optimizer.zero_grad()
        ema.update(raw)

        if step % cfg.log_every == 0:
            count = max(float(running.pop("count")), 1)
            logs = {k: float(v) / (cfg.global_batch_size * cfg.log_every if k.endswith("loss") else count)
                    for k, v in running.items()}
            print(f"step {step}/{cfg.steps} | " + " | ".join(f"{k} {v:.4f}" for k, v in logs.items())
                  + f" | lr {lr_at(step, cfg):.2e} | grad_norm {float(grad_norm):.3f}"
                  + f" | {(time.time() - t0) / step:.3f}s/step", flush=True)
            running = {}

        if step % cfg.eval_every == 0 or step == cfg.steps:
            with ema.applied(raw):  # evaluate and save the EMA weights
                run_eval(step, final=step == cfg.steps)
                os.makedirs(cfg.checkpoint_dir, exist_ok=True)
                torch.save(raw.state_dict(), os.path.join(cfg.checkpoint_dir, f"step_{step}.pt"))


if __name__ == "__main__":
    main()
