# STRM: a state-only, multi-timescale Tiny Recursive Model

Minimal code for the paper *Flat Recursive Reasoning: A Single-State Tiny Recursive Model with a Multi-Timescale
Readout*. It trains and evaluates **STRM**, the state-only model of the paper, and the **TRM** baseline
([Jolicoeur-Martineau, 2025](https://arxiv.org/abs/2510.04871)) on Sudoku-Extreme, Maze-Hard and PPBench, with
the paper's settings.

```
train.py                          both models, all three tasks: data loading, network, training, evaluation
train_stream.py                   segment-free STRM on Sudoku-Extreme (not in the paper; see below)
dataset/build_sudoku_dataset.py   Sudoku-Extreme, 1k training puzzles x 1k augmentations
dataset/build_maze_aug_dataset.py Maze-Hard, 8 dihedral copies per training maze with recomputed labels
dataset/build_ppbench_dataset.py  PPBench, 6 puzzle types at 10x10, 10 augmentations per training puzzle
```

## The models

Both models use the same network, `net(h, u) = B2(B1(h + u))`, made of two post-norm blocks. Each block applies
token mixing and then a SwiGLU MLP. Token mixing is an MLP over the sequence for Sudoku and self-attention with RoPE
otherwise. The input `x` is the embedded puzzle with 16 register tokens. Training uses deep supervision for up to
16 segments with ACT halting, and the models are evaluated with an EMA of the weights after all 16 segments.

**TRM** keeps an answer state `z_H` and a latent state `z_L`:

```
per segment:  repeat H times:  repeat n times: z_L = net(z_L, z_H + x);  z_H = net(z_H, z_L)
              (only the last of the H cycles with gradient)          heads read z_H
```

**STRM** keeps a single state `s`, a bank of leaky integrators with one fixed rate per channel. The rates
`a_d` are log-spaced from 1 to `a_min`:

```
per segment:  repeat K times:  s = (1 - a^c) * s + a^c * net(rms_norm(s) + x)
              (only the last G calls with gradient)
              heads read only the slow channels of s (a_d <= R), rms-normalized among themselves
```

During training, each trajectory draws its own rate exponent `c ~ logU[1/J, J]` (rate jitter). At evaluation
`c = 1`. STRM has exactly the parameters of TRM, and the calls per segment and calls with gradient are matched:
`K = H (n + 1)` and `G = n + 1`.

## Setup

```bash
pip install -r requirements.txt     # PyTorch >= 2.5 with CUDA
python dataset/build_sudoku_dataset.py --output-dir data/sudoku-extreme-1k-aug-1000 --subsample-size 1000 --num-aug 1000
python dataset/build_maze_aug_dataset.py --output-dir data/maze-30x30-hard-1k-aug-8-relabeled
pip install git+https://github.com/approximatelabs/pencil-puzzle-bench   # only for building PPBench
python dataset/build_ppbench_dataset.py --output-dir data/ppbench-6type-10x10-aug-10
```

## Training

```bash
python train.py --task sudoku  --model strm --data-dir data/sudoku-extreme-1k-aug-1000 --seed 0
python train.py --task sudoku  --model trm  --data-dir data/sudoku-extreme-1k-aug-1000 --seed 0
python train.py --task maze    --model strm --data-dir data/maze-30x30-hard-1k-aug-8-relabeled --seed 0
python train.py --task maze    --model trm  --data-dir data/maze-30x30-hard-1k-aug-8-relabeled --seed 0
python train.py --task ppbench --model strm --data-dir data/ppbench-6type-10x10-aug-10 --seed 0
python train.py --task ppbench --model trm  --data-dir data/ppbench-6type-10x10-aug-10 --seed 0
```

Each command runs on one GPU; all runs in the paper used an RTX 5090 or RTX 4090. The presets in `train.py`
(`PRESETS`) hold the paper's settings. Any of them can be overridden from the command line, for example
`--lr 1e-4` or `--micro-batch-size 192`. The micro-batch size only changes memory use: gradients are
accumulated up to the global batch of 768.

Settings shared by all runs: hidden size 512, 2 blocks, 16 registers, AdamW (betas 0.9/0.95), weight decay on
matrices only, gradient clipping at 1.0, 2,000 warm-up steps, global batch 768, and EMA 0.999. The table lists
the settings that differ between runs. "Constant" means the learning rate stays flat after warm-up.

| task | model | recursion | lr (after warm-up) | weight decay | steps |
|---|---|---|---|---|---|
| Sudoku-Extreme | TRM | H 3, n 6 | 1e-4, cosine to 0 | 1.0 | 65,104 |
| Sudoku-Extreme | STRM | K 21, G 7, a_min 1/49, R 1/7, J 1.5 | 1.5e-4, cosine to 0.1x | 0.7 | 65,104 |
| Maze-Hard | TRM | H 3, n 4 | 1e-4, constant | 1.0 | 65,100 |
| Maze-Hard | STRM | K 15, G 5, a_min 1/13, R 1/sqrt(13), J 1.5 | 1.5e-4, constant | 0.7 | 65,100 |
| PPBench | TRM | H 3, n 4 | 1e-4, constant | 1.0 | 242,300 |
| PPBench | STRM | K 15, G 5, a_min 1/13, R 1/sqrt(13), J 1.5 | 1.5e-4, constant | 0.7 | 242,300 |

## Evaluation

Every `--eval-every` steps, the script evaluates the EMA weights and saves them to
`checkpoints/<task>_<model>_seed<seed>/step_<N>.pt`. It evaluates each split the data directory has, out of
`val`, `test` and `golden` (PPBench). For PPBench it also reports exact accuracy per puzzle type.

On Sudoku-Extreme the intermediate evaluations score only the first 10,000 puzzles of each split, and the final
one scores the whole test set (422,786 puzzles). To score a saved checkpoint on every split:

```bash
python train.py --task sudoku --model strm --data-dir data/sudoku-extreme-1k-aug-1000 --eval-only \
    --init-from checkpoints/sudoku_strm_seed0/step_65104.pt
```

## Pretrained checkpoints

`pretrained/` holds the EMA weights of two STRM runs on Sudoku-Extreme with the paper's settings (seeds 0 and 1).
For each run there is the final checkpoint and the best one, chosen by the intermediate evaluations on the first
10,368 test puzzles.

| file | seed | step | exact accuracy |
|---|---|---|---|
| `sudoku_strm_seed0_best.pt` | 0 | 39,060 | 91.3% |
| `sudoku_strm_seed0_final.pt` | 0 | 65,104 | 90.3% |
| `sudoku_strm_seed1_best.pt` | 1 | 52,080 | 93.0% (first 10,368 test puzzles) |
| `sudoku_strm_seed1_final.pt` | 1 | 65,104 | 92.1% |

Unless noted, the accuracy is on the whole test set (422,786 puzzles). To score a checkpoint:

```bash
python train.py --task sudoku --model strm --data-dir data/sudoku-extreme-1k-aug-1000 --eval-only \
    --init-from pretrained/sudoku_strm_seed1_final.pt
```

## Segment-free STRM (Sudoku-Extreme)

`train_stream.py` is a later variant of STRM that is not in the paper. It keeps the state, the rates and the
slow-channel readout, and drops segments, the halting head and rate jitter:

- **Noise.** Every call adds Gaussian noise (std 0.3) to what the network sees, in training and at inference:
  `s = (1 - a) * s + a * net(rms_norm(s) + noise + x)`.
- **Residual.** The change of a call is the RMS difference of `rms_norm(s)` after and before it. The residual
  is the mean change of the last 2 calls. A trajectory has converged when its residual is below a threshold.
- **Training.** Each puzzle runs 4 trajectories, which differ only by their noise. An optimizer micro-step
  advances them by a random number of calls without gradient (0 to 14) and then a random number with gradient
  (3 to 11), and detaches the state. The answer is decoded after every call with gradient; the call `j` calls
  before the last has weight `0.5^j`. Only the trajectory with the lowest answer loss gets the answer loss
  (winner-take-all). A convergence loss on the same trajectory trains the residual to be below 0.09 exactly
  when the decoded answer is right. A puzzle is replaced when one of its trajectories has converged, or after
  336 calls.
- **Inference.** The residual is checked after every call, and a puzzle stops at the first call at which it is
  below 0.07 (`--halt-delta`), or after 1,344 calls. With several trajectories per puzzle, the first one to
  converge gives the answer. No labels are used.

```bash
python train_stream.py --data-dir data/sudoku-extreme-1k-aug-1000 --seed 0
python train_stream.py --data-dir data/sudoku-extreme-1k-aug-1000 --eval-only \
    --init-from pretrained/sudoku_strm_stream_step10000.pt
```

Training takes 25,000 steps (about 11 hours on an RTX 5090, 27 GB with the default micro-batch of 96 puzzles).
Every 2,500 steps the script evaluates the EMA weights on the first 10,368 test puzzles, with 1 and with 4
trajectories per puzzle, and saves them. `--eval-examples 0` scores the whole test set.

`pretrained/sudoku_strm_stream_step10000.pt` holds the EMA weights of one run (seed 0) at step 10,000. On the
first 10,368 test puzzles it scores:

| trajectories per puzzle | exact accuracy | mean calls per trajectory |
|---|---|---|
| 1 | 99.29% | 83.2 |
| 4 | 99.82% | 36.6 |

The checkpoint was chosen by these intermediate evaluations, so the numbers are optimistic. Accuracy of this
run peaks between steps 10,000 and 15,000 and falls afterwards, so the last checkpoint is not the best one.

## Credits

- The network, the deep-supervision and ACT training, and `dataset/build_sudoku_dataset.py` and
  `dataset/common.py` follow the TRM code
  ([SamsungSAILMontreal/TinyRecursiveModels](https://github.com/SamsungSAILMontreal/TinyRecursiveModels), MIT).
- Torben Glass wrote the relabeled Maze-Hard data, the PPBench pipeline and its training recipe
  (`dataset/build_maze_aug_dataset.py`, `dataset/build_ppbench_dataset.py`).
- PPBench: [approximatelabs/pencil-puzzle-bench](https://github.com/approximatelabs/pencil-puzzle-bench).
  Sudoku-Extreme and Maze-Hard: [sapientinc](https://huggingface.co/sapientinc).

## License

MIT
