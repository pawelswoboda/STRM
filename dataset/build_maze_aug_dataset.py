# By Torben Glass (MIT). Maze-Hard with the 8 dihedral transforms of every training maze, each label recomputed with
# the tie-breaking rule of the generator (transforming the published labels makes them inconsistent):
#   python dataset/build_maze_aug_dataset.py --output-dir data/maze-30x30-hard-1k-aug-8-relabeled
from collections.abc import Iterator
from collections import deque
import math
import os
import csv
import json
import numpy as np

from argdantic import ArgParser
from pydantic import BaseModel
from tqdm import tqdm
from huggingface_hub import hf_hub_download

from common import PuzzleDatasetMetadata, dihedral_transform


CHARSET = "# SGo"
WALL, EMPTY, START, GOAL, PATH = map(ord, CHARSET)

# Tie-breaking rule of the A* implementation the source dataset was generated with
# (facebookresearch/searchformer): equal-cost nodes are ordered by
# GridPos.__hash__ = x + 100000 * y, which in grid coordinates means
# down, then left, then right, then up.
ORDER = ((1, 0), (0, -1), (0, 1), (-1, 0))


cli = ArgParser()


class DataProcessConfig(BaseModel):
    source_repo: str = "sapientinc/maze-30x30-hard-1k"
    output_dir: str = "data/processed/maze-30x30-hard-1k-aug-8-relabeled"

    subsample_size: int | None = None
    aug: int = 8
    verify: bool = True


def neighbors(grid: np.ndarray, p: tuple[int, int]) -> Iterator[tuple[int, int]]:
    """Walkable neighbours of `p`, yielded in tie-breaking order."""
    for dr, dc in ORDER:
        r, c = p[0] + dr, p[1] + dc
        if 0 <= r < grid.shape[0] and 0 <= c < grid.shape[1] and grid[r, c] != WALL:
            yield r, c


def canonical_label(grid: np.ndarray) -> np.ndarray:
    """Solve `grid` and return its label, picking the same shortest path as the generator.

    A dihedral transform of a maze is a different maze, and the transformed label is in
    general no longer the path the tie-breaking rule selects. The label therefore has to be
    recomputed from the transformed maze instead of being transformed along with it.
    """
    goal = tuple(np.argwhere(grid == GOAL)[0])
    dist, queue = {goal: 0}, deque([goal])
    while queue:
        p = queue.popleft()
        for n in neighbors(grid, p):
            if n not in dist:
                dist[n] = dist[p] + 1
                queue.append(n)

    p = tuple(np.argwhere(grid == START)[0])
    assert p in dist, "maze has no path from start to goal"

    label = grid.copy()
    while dist[p] > 0:
        d = dist[p]
        p = next(n for n in neighbors(grid, p) if dist.get(n) == d - 1)
        if label[p] == EMPTY:
            label[p] = PATH
    return label


def read_subset(set_name: str, config: DataProcessConfig) -> tuple[list[np.ndarray], list[np.ndarray]]:
    inputs = []
    labels = []

    with open(hf_hub_download(config.source_repo, f"{set_name}.csv", repo_type="dataset"), newline="") as csvfile:  # type: ignore
        reader = csv.reader(csvfile)
        next(reader)  # Skip header
        for source, q, a, rating in reader:
            n = int(len(q) ** 0.5)
            inputs.append(np.frombuffer(q.encode(), dtype=np.uint8).reshape(n, n))
            labels.append(np.frombuffer(a.encode(), dtype=np.uint8).reshape(n, n))

    return inputs, labels


def convert_subset(set_name: str, config: DataProcessConfig) -> None:
    inputs, labels = read_subset(set_name, config)
    grid_size = inputs[0].shape

    # If subsample_size is specified for the training set,
    # randomly sample the desired number of examples.
    if set_name == "train" and config.subsample_size is not None:
        total_samples = len(inputs)
        if config.subsample_size < total_samples:
            indices = np.random.choice(total_samples, size=config.subsample_size, replace=False)
            inputs = [inputs[i] for i in indices]
            labels = [labels[i] for i in indices]

    # Only the training set is augmented; the test set stays as published.
    num_aug = config.aug if set_name == "train" else 1

    # Generate dataset
    results = {k: [] for k in ["inputs", "labels", "puzzle_identifiers", "puzzle_indices", "group_indices"]}
    puzzle_id = 0
    example_id = 0
    not_reproduced = 0

    results["puzzle_indices"].append(0)
    results["group_indices"].append(0)

    for inp, out in zip(tqdm(inputs), labels):
        for aug_idx in range(num_aug):
            aug_inp = dihedral_transform(inp, aug_idx)
            aug_out = canonical_label(aug_inp) if num_aug > 1 else out

            # The untransformed maze must reproduce the published label, otherwise the
            # source dataset was not generated with the tie-breaking rule assumed here.
            if config.verify and num_aug > 1 and aug_idx == 0 and not np.array_equal(aug_out, out):
                not_reproduced += 1

            results["inputs"].append(aug_inp)
            results["labels"].append(aug_out)
            example_id += 1
            puzzle_id += 1

            results["puzzle_indices"].append(example_id)
            results["puzzle_identifiers"].append(0)

        # Push group
        results["group_indices"].append(puzzle_id)

    if not_reproduced:
        raise ValueError(
            f"{not_reproduced}/{len(inputs)} published labels are not reproduced by the "
            f"tie-breaking rule {ORDER}. The source dataset appears to use a different or "
            f"non-deterministic generator, so the recomputed labels would be inconsistent "
            f"with it. Re-run with verify=False to build anyway."
        )

    # Char mappings
    char2id = np.zeros(256, np.uint8)
    char2id[np.array(list(map(ord, CHARSET)))] = np.arange(len(CHARSET)) + 1

    # To Numpy
    def _seq_to_numpy(seq: list[np.ndarray]) -> np.ndarray:
        arr = np.vstack([char2id[s.reshape(-1)] for s in seq])

        return arr

    results = {
        "inputs": _seq_to_numpy(results["inputs"]),
        "labels": _seq_to_numpy(results["labels"]),

        "group_indices": np.array(results["group_indices"], dtype=np.int32),
        "puzzle_indices": np.array(results["puzzle_indices"], dtype=np.int32),
        "puzzle_identifiers": np.array(results["puzzle_identifiers"], dtype=np.int32),
    }

    # Metadata
    # Training samples one puzzle per group per epoch, so total_groups and
    # mean_puzzle_examples stay the same as in the unaugmented dataset.
    metadata = PuzzleDatasetMetadata(
        seq_len=int(math.prod(grid_size)),
        vocab_size=len(CHARSET) + 1,  # PAD + Charset
        pad_id=0,
        ignore_label_id=0,
        blank_identifier_id=0,
        num_puzzle_identifiers=1,
        total_groups=len(results["group_indices"]) - 1,
        mean_puzzle_examples=1,
        total_puzzles=len(results["group_indices"]) - 1,
        sets=["all"]
    )

    # Save metadata as JSON.
    save_dir = os.path.join(config.output_dir, set_name)
    os.makedirs(save_dir, exist_ok=True)

    with open(os.path.join(save_dir, "dataset.json"), "w") as f:
        json.dump(metadata.model_dump(), f)

    # Save data
    for k, v in results.items():
        np.save(os.path.join(save_dir, f"all__{k}.npy"), v)

    # Save IDs mapping (for visualization only)
    with open(os.path.join(config.output_dir, "identifiers.json"), "w") as f:
        json.dump(["<blank>"], f)


@cli.command(singleton=True)
def preprocess_data(config: DataProcessConfig) -> None:
    convert_subset("train", config)
    convert_subset("test", config)


if __name__ == "__main__":
    cli()
