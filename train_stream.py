#!/usr/bin/env python3
"""Segment-free STRM on Sudoku-Extreme -- one file on top of train.py (network, data, EMA).

The model is STRM's single state s with one fixed rate a_d per channel (log-spaced from 1 to rate_min), without
rate jitter, and with Gaussian noise on what the network sees. One network call is

    s <- (1 - a) * s + a * net(rms_norm(s) + noise * N(0, I) + x),

and the answer is decoded from the slow channels of s (rate <= readout_max_rate). There are no segments, no
halting head and no fixed number of calls:

  * Residual. The change of a call is the RMS, over tokens and channels, of the difference of rms_norm(s) after
    and before it; the residual is the mean change of the last `conv_calls` calls. A trajectory has converged
    when its residual is below a threshold.
  * Training. Every puzzle runs `trajectories` (K) trajectories, which differ only by their noise. An optimizer
    micro-step advances all of them by a random number of calls without gradient (uniform in [0, skip_max]) and
    then by a random number of calls with gradient (uniform in [grad_min, grad_max]); after that the state is
    detached. The answer is decoded after every call with gradient. The call j calls before the last one has
    weight call_loss_decay ** j, and only the puzzle's winner (the trajectory with the lowest answer loss at the
    last call) gets the answer loss. The convergence loss, on the winner as well, is the BCE of the logit
    (conv_delta - residual) / conv_kappa against "the decoded answer is exactly right"; it has the same
    per-call weights, normalized to sum to 1. A puzzle is replaced by a new one once the residual of one of its
    trajectories was below conv_delta after some call of the micro-step, or after max_calls calls.
  * Inference. The residual is checked after every call. A puzzle stops at the first call at which a
    trajectory's residual is below halt_delta and returns that trajectory's answer; a puzzle that never
    converges stops after eval_max_calls calls with the answer of its lowest-residual trajectory. No labels are
    used. halt_delta is a little below conv_delta: the convergence loss puts its decision boundary at
    conv_delta, so wrong trajectories sit just above it.

Usage:
    python train_stream.py --data-dir data/sudoku-extreme-1k-aug-1000
    python train_stream.py --data-dir data/sudoku-extreme-1k-aug-1000 --eval-only \
        --init-from pretrained/sudoku_strm_stream_step10000.pt
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from dataclasses import dataclass, fields
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from train import (EMA, IGNORE_LABEL_ID, CastedEmbedding, CastedLinear, Net, eval_batches, load_split, lr_at,
                   rms_norm, stablemax_cross_entropy, train_batches, trunc_normal_init_)


@dataclass
class Config:
    data_dir: str = "data/sudoku-extreme-1k-aug-1000"

    # Network (as STRM on Sudoku in train.py)
    mixer: str = "mlp"
    hidden_size: int = 512
    expansion: float = 4.0
    num_layers: int = 2
    num_heads: int = 8             # only sets the (unused) RoPE table of train.Net
    rope_theta: float = 10000.0
    puzzle_emb_len: int = 16       # register tokens
    rms_norm_eps: float = 1e-5

    # State
    rate_min: float = 1 / 49       # slowest rate; rates are log-spaced from 1 to rate_min over the channels
    readout_max_rate: float = 1 / 7  # the head reads the channels with rate <= this
    noise: float = 0.3             # std of the noise added to rms_norm(s) at every call, training and inference

    # Stream of calls
    trajectories: int = 4          # K trajectories per puzzle in training (winner-take-all)
    skip_max: int = 14             # calls without gradient per micro-step: uniform in [0, skip_max]
    grad_min: int = 3              # calls with gradient per micro-step: uniform in [grad_min, grad_max]
    grad_max: int = 11
    max_calls: int = 336           # training: a puzzle is replaced after this many calls at the latest
    call_loss_decay: float = 0.5   # the call j calls before the last one of a micro-step has weight decay ** j

    # Convergence loss and halting
    conv_weight: float = 0.1
    conv_delta: float = 0.09       # residual threshold of the convergence loss and of replacement in training
    conv_kappa: float = 0.02
    conv_calls: int = 2            # calls the residual is averaged over
    halt_delta: float = 0.07       # residual threshold at inference
    eval_max_calls: int = 1344

    # Optimization
    steps: int = 25000
    global_batch_size: int = 768   # puzzles per optimizer step
    micro_batch_size: int = 96     # puzzles per micro-step (x K trajectories); gradients accumulate over
    lr: float = 1.5e-4             # global_batch_size / micro_batch_size micro-steps (one stream each)
    lr_warmup_steps: int = 2000
    lr_min_ratio: float = 0.1      # after warmup: cosine decay to lr_min_ratio * lr
    beta1: float = 0.9
    beta2: float = 0.95
    weight_decay: float = 0.7      # on all parameters with >= 2 dims
    grad_clip: float = 1.0
    ema_rate: float = 0.999

    # Evaluation / run
    eval_every: int = 2500         # optimizer steps between evaluations (each saves an EMA checkpoint)
    eval_examples: int = 10368     # test puzzles at intermediate evaluations (0 = all; the last one: all)
    eval_trajectories: str = "1,4"  # trajectories per puzzle at inference; one evaluation per entry
    seed: int = 0
    checkpoint_dir: str = ""
    init_from: str = ""            # load these (EMA) weights before training / evaluation
    eval_only: bool = False        # with --init-from: evaluate on the test set and exit
    log_every: int = 100
    compile: bool = True

    # Filled in from the dataset
    vocab_size: int = 0
    seq_len: int = 0

    @property
    def seq_total(self) -> int:
        return self.seq_len + self.puzzle_emb_len


def parse_args() -> Config:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for f in fields(Config):
        if f.name in ("vocab_size", "seq_len"):
            continue
        flag = "--" + f.name.replace("_", "-")
        if isinstance(f.default, bool):
            p.add_argument(flag, dest=f.name, default=f.default, action=argparse.BooleanOptionalAction)
        else:
            p.add_argument(flag, dest=f.name, type=type(f.default), default=f.default)
    cfg = Config(**vars(p.parse_args()))
    cfg.checkpoint_dir = cfg.checkpoint_dir or f"checkpoints/sudoku_strm_stream_seed{cfg.seed}"
    assert cfg.global_batch_size % cfg.micro_batch_size == 0 and cfg.trajectories >= 1
    assert cfg.skip_max >= 0 and 1 <= cfg.conv_calls <= cfg.grad_min <= cfg.grad_max
    assert 0 < cfg.rate_min <= cfg.readout_max_rate <= 1.0 and cfg.eval_max_calls >= cfg.conv_calls
    return cfg


# ---------------------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------------------
class StreamSTRM(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        D = cfg.hidden_size
        self.embed_tokens = CastedEmbedding(cfg.vocab_size, D, init_std=D ** -0.5)
        self.lm_head = CastedLinear(D, cfg.vocab_size)
        self.puzzle_emb = nn.Parameter(torch.zeros(D))  # one learned vector in register 0 (all puzzles share it)
        self.net = Net(cfg)
        self.s_init = nn.Buffer(trunc_normal_init_(torch.empty(D, dtype=torch.bfloat16)))
        rates = torch.logspace(0, math.log10(cfg.rate_min), D, dtype=torch.float64, device="cpu")
        self.rates = nn.Buffer(rates.float().to(self.s_init.device))                  # fast -> slow over channels
        self.readout_mask = nn.Buffer((self.rates <= cfg.readout_max_rate * (1 + 1e-6)).float())
        self.rng = random.Random(cfg.seed)  # the numbers of calls of a micro-step

    def embed(self, inputs):
        emb = self.embed_tokens(inputs)
        pe = F.pad(self.puzzle_emb.to(torch.bfloat16), (0, (self.cfg.puzzle_emb_len - 1) * self.cfg.hidden_size))
        pe = pe.view(1, self.cfg.puzzle_emb_len, -1).expand(emb.shape[0], -1, -1)
        return math.sqrt(self.cfg.hidden_size) * torch.cat((pe, emb), dim=-2)

    def decode(self, s):
        """Answer logits from the slow channels of s, rms-normalized among themselves."""
        m = self.readout_mask
        r = s * m * torch.rsqrt((s.square() * m).sum(-1, keepdim=True) / m.sum() + self.cfg.rms_norm_eps)
        return self.lm_head(r.to(torch.bfloat16))[:, self.cfg.puzzle_emb_len:]

    def calls(self, s, x, n: int, decode: bool = False) -> Tuple[torch.Tensor, List[torch.Tensor], List[torch.Tensor]]:
        """n calls on the state s (B, L, D). Returns s, every call's change (B,) and, with decode, its logits."""
        a = self.rates.view(1, 1, -1)
        changes, logits = [], []
        normed = rms_norm(s, self.cfg.rms_norm_eps)
        for _ in range(n):
            seen = normed.to(torch.bfloat16)
            seen = seen + self.cfg.noise * torch.randn_like(seen)
            s = (1 - a) * s + a * self.net(seen + x).float()
            before, normed = normed, rms_norm(s, self.cfg.rms_norm_eps)
            changes.append(((normed - before).square().mean(dim=(-1, -2)) + 1e-12).sqrt())
            if decode:
                logits.append(self.decode(s))
        return s, changes, logits

    def initial_carry(self, batch):
        B = batch["inputs"].shape[0] * self.cfg.trajectories
        dev = batch["inputs"].device
        return dict(s=torch.empty(B, self.cfg.seq_total, self.cfg.hidden_size, device=dev),
                    calls=torch.zeros(B, dtype=torch.int32, device=dev),
                    halted=torch.ones(B, dtype=torch.bool, device=dev),  # every slot takes a puzzle first
                    data={k: torch.empty_like(v).repeat_interleave(self.cfg.trajectories, dim=0)
                          for k, v in batch.items()})

    def forward(self, carry, batch):
        """One training micro-step. Rows i * K ... i * K + K - 1 are the K trajectories of puzzle i."""
        cfg, K = self.cfg, self.cfg.trajectories
        halted = carry["halted"]
        s = torch.where(halted.view(-1, 1, 1), self.s_init.float(), carry["s"])
        calls = torch.where(halted, 0, carry["calls"])
        data = {k: torch.where(halted.view(-1, 1), batch[k].repeat_interleave(K, dim=0), v)
                for k, v in carry["data"].items()}
        x, labels = self.embed(data["inputs"]), data["labels"]

        n_free, n_grad = self.rng.randint(0, cfg.skip_max), self.rng.randint(cfg.grad_min, cfg.grad_max)
        with torch.no_grad():
            s, free_changes, _ = self.calls(s, x, n_free)
        s, changes, logits = self.calls(s, x, n_grad, decode=True)

        mask = labels != IGNORE_LABEL_ID
        counts = mask.sum(-1)
        answer = [stablemax_cross_entropy(lg, labels, mask).sum(-1) / counts.clamp_min(1) for lg in logits]
        with torch.no_grad():
            cells = [mask & (lg.argmax(-1) == labels) for lg in logits]
            correct = [c.sum(-1) == counts for c in cells]
            winner = F.one_hot(answer[-1].view(-1, K).argmin(1), K).reshape(-1).bool()

        # Per call with gradient: the answer loss and (once conv_calls calls with gradient lie behind it) the
        # convergence loss, both on the winner, the call j calls before the last one with weight decay ** j.
        lm_loss, conv_loss, conv_norm = 0.0, 0.0, 0.0
        for j in range(n_grad):
            weight = cfg.call_loss_decay ** (n_grad - 1 - j)
            lm_loss = lm_loss + weight * (answer[j] * winner).sum()
            if j + 1 >= cfg.conv_calls:
                residual = torch.stack(changes[j + 1 - cfg.conv_calls:j + 1], 0).mean(0)
                bce = F.binary_cross_entropy_with_logits((cfg.conv_delta - residual) / cfg.conv_kappa,
                                                         correct[j].to(residual.dtype), reduction="none")
                conv_loss = conv_loss + weight * (bce * winner).sum()
                conv_norm += weight
        conv_loss = conv_loss / conv_norm
        loss = lm_loss + cfg.conv_weight * conv_loss

        with torch.no_grad():
            # the residual after every call of this micro-step: a puzzle is replaced once one trajectory converged
            all_changes = torch.stack(free_changes + [c.detach() for c in changes], 0)
            lowest = all_changes.unfold(0, cfg.conv_calls, 1).mean(-1).amin(0)
            calls = calls + n_free + n_grad
            halted = (lowest < cfg.conv_delta).view(-1, K).any(1).repeat_interleave(K) | (calls >= cfg.max_calls)
            # metrics of the puzzles that leave, on the trajectory inference would pick (lowest residual)
            picked = F.one_hot(residual.view(-1, K).argmin(1), K).reshape(-1).bool()
            valid = halted & picked & (counts > 0)
            metrics = {"count": valid.sum(),
                       "accuracy": torch.where(valid, cells[-1].sum(-1) / counts.clamp_min(1), 0).sum(),
                       "exact_accuracy": (valid & correct[-1]).sum(),
                       "conv_accuracy": (valid & ((residual < cfg.conv_delta) == correct[-1])).sum(),
                       "calls": torch.where(valid, calls, 0).sum(),
                       "lm_loss": lm_loss.detach(), "conv_loss": conv_loss.detach()}
        return dict(s=s.detach(), calls=calls, halted=halted, data=data), loss, metrics


# ---------------------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------------------
@torch.inference_mode()
def evaluate(model: StreamSTRM, data, cfg: Config, device, K: int, max_examples: int = 0, chunk: int = 7):
    """Inference with K trajectories per puzzle: the residual is checked after every call, and a puzzle stops at
    the first call at which one trajectory is below halt_delta (or at eval_max_calls) with the answer of its
    lowest-residual trajectory. The calls run in chunks, after which the stopped puzzles leave the batch."""
    compiled = model.net.__dict__.pop("forward", None)  # run the net eagerly here: the batch size changes
    names = ["puzzles", "accuracy", "exact_accuracy", "converged", "conv_accuracy", "calls"]
    totals = torch.zeros(len(names), dtype=torch.float64, device=device)
    for batch in eval_batches(data, cfg.micro_batch_size, device, max_examples):
        real = (batch["labels"] != IGNORE_LABEL_ID).any(-1)  # the last chunk is padded
        inputs = batch["inputs"][real].repeat_interleave(K, dim=0)
        labels = batch["labels"][real].repeat_interleave(K, dim=0)
        s = model.s_init.float().expand(inputs.shape[0], cfg.seq_total, cfg.hidden_size).clone()
        totals[0] += inputs.shape[0] // K
        recent: List[torch.Tensor] = []  # the changes of the last conv_calls calls
        call = 0
        while call < cfg.eval_max_calls:
            s, changes, logits = model.calls(s, model.embed(inputs), min(chunk, cfg.eval_max_calls - call), decode=True)
            done = torch.zeros(inputs.shape[0] // K, dtype=torch.bool, device=device)  # puzzles stopped in this chunk
            for change, call_logits in zip(changes, logits):
                call += 1
                recent = (recent + [change])[-cfg.conv_calls:]
                residual, best = torch.stack(recent, 0).mean(0).view(-1, K).min(1)
                cells = ((call_logits.argmax(-1) == labels) | (labels == IGNORE_LABEL_ID)).view(len(best), K, -1)
                cells = cells.gather(1, best.view(-1, 1, 1).expand(-1, 1, cells.shape[-1]))[:, 0]
                mask = (labels != IGNORE_LABEL_ID)[::K]
                converged = (residual < cfg.halt_delta) & (call >= cfg.conv_calls)
                stop = (converged if call < cfg.eval_max_calls else torch.ones_like(converged)) & ~done
                exact = cells.all(-1)
                totals[1] += ((cells & mask).sum(-1) / mask.sum(-1))[stop].sum()
                totals[2] += exact[stop].sum()
                totals[3] += converged[stop].sum()
                totals[4] += (converged == exact)[stop].sum()
                totals[5] += call * stop.sum()
                done = done | stop
            keep = (~done).repeat_interleave(K)
            if not keep.any():
                break
            s, inputs, labels = s[keep], inputs[keep], labels[keep]
            recent = [c[keep] for c in recent]
    if compiled is not None:
        model.net.forward = compiled
    out = dict(zip(names, totals.cpu().tolist()))
    return {k: v if k == "puzzles" else v / max(out["puzzles"], 1) for k, v in out.items()}


# ---------------------------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------------------------
def main():
    cfg = parse_args()
    device = torch.device("cuda")
    torch.manual_seed(cfg.seed)

    with open(os.path.join(cfg.data_dir, "train", "dataset.json")) as f:
        meta = json.load(f)
    cfg.vocab_size, cfg.seq_len = int(meta["vocab_size"]), int(meta["seq_len"])
    train_data, test_data = load_split(cfg.data_dir, "train"), load_split(cfg.data_dir, "test")

    with torch.device(device):
        model = StreamSTRM(cfg)
    if cfg.init_from:
        model.load_state_dict(torch.load(cfg.init_from, map_location=device))
    if cfg.compile:  # only the net: the number of calls changes from micro-step to micro-step
        model.net.forward = torch.compile(model.net.forward)
    print(f"[sudoku / segment-free strm] {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M params | {cfg}", flush=True)

    def run_eval(step, final):
        for K in (int(k) for k in cfg.eval_trajectories.split(",")):
            m = evaluate(model, test_data, cfg, device, K, 0 if final else cfg.eval_examples)
            print(f"[eval test @ step {step} | {K} trajectories] " + " | ".join(f"{k} {v:.4f}" for k, v in m.items()),
                  flush=True)

    if cfg.eval_only:
        return run_eval(0, final=cfg.eval_examples == 0)

    params = list(model.parameters())
    optimizer = torch.optim.AdamW([{"params": [p for p in params if p.dim() >= 2], "weight_decay": cfg.weight_decay},
                                   {"params": [p for p in params if p.dim() < 2], "weight_decay": 0.0}],
                                  lr=cfg.lr, betas=(cfg.beta1, cfg.beta2))
    ema = EMA(model, cfg.ema_rate)
    accum = cfg.global_batch_size // cfg.micro_batch_size
    batches = train_batches(train_data, cfg.micro_batch_size, cfg.seed, device)
    carries = [None] * accum  # one persistent stream of puzzles per micro-batch slot
    running: Dict[str, torch.Tensor] = {}
    t0 = time.time()

    for step in range(1, cfg.steps + 1):
        for slot in range(accum):
            batch = next(batches)
            carries[slot] = carries[slot] or model.initial_carry(batch)
            carries[slot], loss, metrics = model(carries[slot], batch)
            ((1 / cfg.global_batch_size) * loss).backward()
            for k, v in metrics.items():
                running[k] = running.get(k, 0) + v.float()
        grad_norm = torch.nn.utils.clip_grad_norm_(params, cfg.grad_clip, error_if_nonfinite=True)
        for group in optimizer.param_groups:
            group["lr"] = lr_at(step, cfg)
        optimizer.step()
        optimizer.zero_grad()
        ema.update(model)

        if step % cfg.log_every == 0:
            count = max(float(running.pop("count")), 1)
            logs = {k: float(v) / (cfg.global_batch_size * cfg.log_every if k.endswith("loss") else count)
                    for k, v in running.items()}
            print(f"step {step}/{cfg.steps} | " + " | ".join(f"{k} {v:.4f}" for k, v in logs.items())
                  + f" | lr {lr_at(step, cfg):.2e} | grad_norm {float(grad_norm):.3f}"
                  + f" | {(time.time() - t0) / step:.3f}s/step", flush=True)
            running = {}

        if step % cfg.eval_every == 0 or step == cfg.steps:
            with ema.applied(model):  # evaluate and save the EMA weights
                run_eval(step, final=step == cfg.steps)
                os.makedirs(cfg.checkpoint_dir, exist_ok=True)
                torch.save(model.state_dict(), os.path.join(cfg.checkpoint_dir, f"step_{step}.pt"))


if __name__ == "__main__":
    main()
