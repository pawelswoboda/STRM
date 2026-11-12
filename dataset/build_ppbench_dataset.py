# By Torben Glass (MIT). PPBench (Pencil Puzzle Bench) at six puzzle types and grid size 10x10, 10 augmentations per
# training puzzle; needs the ppbench package (pip install git+https://github.com/approximatelabs/pencil-puzzle-bench):
#   python dataset/build_ppbench_dataset.py --output-dir data/ppbench-6type-10x10-aug-10
from typing import Any
import base64
import json
import os
import random

import numpy as np
from argdantic import ArgParser
from pydantic import BaseModel
from tqdm import tqdm
from huggingface_hub import hf_hub_download
from ppbench import Puzzle

from common import PuzzleDatasetMetadata, dihedral_transform


# Puzzle types kept from PPBench, with the single grid size retained for each.
PUZZLE_TYPES = {
    "sudoku": 9,
    "lightup": 10,
    "nurikabe": 10,
    "shakashaka": 10,
    "heyawake": 10,
    "tapa": 10,
}
GRID_SIZE = 10

# Shakashaka answer symbols are triangles, named by the corner holding the right angle.
# A dihedral transform moves that corner, so the symbols have to be relabelled too.
SHAKASHAKA_CORNER = {"5": "NW", "4": "NE", "3": "SE", "2": "SW"}
SHAKASHAKA_SYMBOL = {v: k for k, v in SHAKASHAKA_CORNER.items()}
CORNER_MAPS = [
    {"NW": "NW", "NE": "NE", "SE": "SE", "SW": "SW"},
    {"NW": "SW", "SW": "SE", "SE": "NE", "NE": "NW"},
    {"NW": "SE", "NE": "SW", "SE": "NW", "SW": "NE"},
    {"NW": "NE", "NE": "SE", "SE": "SW", "SW": "NW"},
    {"NW": "NE", "NE": "NW", "SW": "SE", "SE": "SW"},
    {"NW": "SW", "SW": "NW", "NE": "SE", "SE": "NE"},
    {"NW": "NW", "SE": "SE", "NE": "SW", "SW": "NE"},
    {"NW": "SE", "SE": "NW", "NE": "NE", "SW": "SW"},
]

_XOR_KEY = b"ppbench"

Record = dict[str, Any]  # one decoded PPBench JSON record
Example = tuple[np.ndarray, np.ndarray]  # (input, label) token grids, [size, size] each


cli = ArgParser()


class DataProcessConfig(BaseModel):
    source_repo: str = "bluecoconut/pencil-puzzle-bench"
    output_dir: str = "data/ppbench-6type-10x10-aug-10"

    num_aug: int = 10
    val_per_type: int = 100
    val_per_type_tapa: int = 50
    seed: int = 37


def load_records(config: DataProcessConfig, filename: str) -> list[Record]:
    path = hf_hub_download(config.source_repo, filename, repo_type="dataset")

    records = []
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            enc = base64.b64decode(record.pop("solution_enc"))
            record["solution"] = json.loads(bytes(b ^ _XOR_KEY[i % len(_XOR_KEY)] for i, b in enumerate(enc)))
            records.append(record)

    return records


def split_records(config: DataProcessConfig) -> dict[str, list[Record]]:
    """Hold out the golden set, then a fixed-size validation set per type, then filter to the kept types."""
    rng = random.Random(config.seed)

    golden_urls = {r["puzzlink_url"] for r in load_records(config, "golden_300.jsonl")}
    records = load_records(config, "full_dataset.jsonl")

    golden = [r for r in records if r["puzzlink_url"] in golden_urls]
    remaining = [r for r in records if r["puzzlink_url"] not in golden_urls]

    by_type: dict[str, list[Record]] = {}
    for r in remaining:
        by_type.setdefault(r["pid"], []).append(r)

    val_urls = set()
    for pid, group in by_type.items():
        n = config.val_per_type_tapa if pid == "tapa" else config.val_per_type
        val_urls.update(r["puzzlink_url"] for r in rng.sample(group, min(n, len(group))))

    def keep(r: Record) -> bool:
        return r["pid"] in PUZZLE_TYPES and r["width"] == r["height"] == PUZZLE_TYPES[r["pid"]]

    return {
        "train": [r for r in remaining if keep(r) and r["puzzlink_url"] not in val_urls],
        "test": [r for r in remaining if keep(r) and r["puzzlink_url"] in val_urls],
        "golden": [r for r in golden if keep(r)],
    }


def read_layers(state: str, size: int) -> list[np.ndarray]:
    """Split a pzprv3 board dump into its stacked size x size layers."""
    rows = [line.split() for line in state.splitlines()[2:]]
    rows = [r for r in rows if len(r) == size]
    assert len(rows) % size == 0

    return [np.array(rows[i:i + size], dtype=object) for i in range(0, len(rows), size)]


def room_borders(rooms: np.ndarray) -> np.ndarray:
    """Replace heyawake room ids by a per-cell bitmask of which sides are room boundaries."""
    size = rooms.shape[0]
    out = np.empty_like(rooms)
    for r in range(size):
        for c in range(size):
            bits = 0
            for i, (dr, dc) in enumerate(((0, 1), (1, 0), (0, -1), (-1, 0))):
                rr, cc = r + dr, c + dc
                if not (0 <= rr < size and 0 <= cc < size) or rooms[rr, cc] != rooms[r, c]:
                    bits |= 1 << i
            out[r, c] = str(bits)
    return out


def tapa_clue(s: str) -> str:
    """Sort a tapa clue: its numbers are an unordered set, so "1,2,1" and "1,1,2" are one clue."""
    if "," not in s:
        return s
    return ",".join(sorted(s.split(","), key=lambda p: (p == "-", 0 if p == "-" else int(p))))


def encode(pid: str, state: str, size: int, tid: int) -> np.ndarray:
    """Turn a board dump into a size x size grid of one string token per cell."""
    layers = [dihedral_transform(layer, tid) for layer in read_layers(state, size)]

    if pid == "shakashaka":
        corners = CORNER_MAPS[tid]
        layers[-1] = np.vectorize(
            lambda s: SHAKASHAKA_SYMBOL[corners[SHAKASHAKA_CORNER[s]]] if s in SHAKASHAKA_CORNER else s,
            otypes=[object],
        )(layers[-1])
    elif pid == "heyawake":
        layers[0] = room_borders(layers[0])
    elif pid == "tapa":
        layers[0] = np.vectorize(tapa_clue, otypes=[object])(layers[0])

    grid = layers[0]
    for layer in layers[1:]:
        grid = np.char.add(np.char.add(grid.astype(str), "|"), layer.astype(str)).astype(object)

    return np.vectorize(lambda s: f"{pid}:{s}", otypes=[object])(grid)


def build_examples(record: Record, num_aug: int, rng: random.Random) -> list[Example] | None:
    """Return (input, label) token grids: the untouched puzzle first, then num_aug augmented copies.

    Augmentation combines trajectory sampling (the input is a random intermediate solve state)
    with a random dihedral transform applied to input and label alike.
    """
    pid, size = record["pid"], PUZZLE_TYPES[record["pid"]]
    moves = list(record["solution"]["moves_full"])

    try:
        puzzle = Puzzle.from_url(record["puzzlink_url"])
    except Exception:
        return None

    initial = puzzle.get_state()

    # One puzz.link sudoku blanks out the nine box centres (pzpr's "no number" cells, dumped as
    # "-"). This is not a valid sudoku puzzle, so we skip it.
    if pid == "sudoku" and any(tok == "-" for line in initial.splitlines()[2:] for tok in line.split()):
        return None

    for move in moves:
        puzzle.send_move(move)
    if not puzzle.is_complete():
        return None
    solved = puzzle.get_state()

    examples = [(encode(pid, initial, size, 0), encode(pid, solved, size, 0))]

    for _ in range(num_aug):
        order = rng.sample(moves, len(moves))
        puzzle.reset()
        for move in order[:rng.randrange(len(moves))]:
            puzzle.send_move(move)

        tid = rng.randrange(8)
        examples.append((encode(pid, puzzle.get_state(), size, tid), encode(pid, solved, size, tid)))

    return examples


def to_padded_ids(grid: np.ndarray, vocab: dict[str, int]) -> np.ndarray:
    """Map tokens to ids and pad the grid out to GRID_SIZE x GRID_SIZE."""
    out = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.int32)
    ids = np.vectorize(vocab.__getitem__, otypes=[np.int32])(grid)
    out[:ids.shape[0], :ids.shape[1]] = ids
    return out.reshape(-1)


def save_subset(set_name: str, examples: list[list[Example]], vocab: dict[str, int], config: DataProcessConfig) -> None:
    results = {k: [] for k in ["inputs", "labels", "puzzle_identifiers", "puzzle_indices", "group_indices"]}
    puzzle_id = 0
    example_id = 0

    results["puzzle_indices"].append(0)
    results["group_indices"].append(0)

    for group in examples:
        for inp, out in group:
            results["inputs"].append(to_padded_ids(inp, vocab))
            results["labels"].append(to_padded_ids(out, vocab))
            example_id += 1
            puzzle_id += 1

            results["puzzle_indices"].append(example_id)
            results["puzzle_identifiers"].append(0)

        results["group_indices"].append(puzzle_id)

    results = {
        "inputs": np.stack(results["inputs"]),
        "labels": np.stack(results["labels"]),

        "group_indices": np.array(results["group_indices"], dtype=np.int32),
        "puzzle_indices": np.array(results["puzzle_indices"], dtype=np.int32),
        "puzzle_identifiers": np.array(results["puzzle_identifiers"], dtype=np.int32),
    }

    metadata = PuzzleDatasetMetadata(
        seq_len=GRID_SIZE * GRID_SIZE,
        vocab_size=len(vocab) + 1,  # PAD + tokens
        pad_id=0,
        ignore_label_id=0,
        blank_identifier_id=0,
        num_puzzle_identifiers=1,
        total_groups=len(results["group_indices"]) - 1,
        mean_puzzle_examples=1,
        total_puzzles=len(results["group_indices"]) - 1,
        sets=["all"]
    )

    save_dir = os.path.join(config.output_dir, set_name)
    os.makedirs(save_dir, exist_ok=True)

    with open(os.path.join(save_dir, "dataset.json"), "w") as f:
        json.dump(metadata.model_dump(), f)

    for k, v in results.items():
        np.save(os.path.join(save_dir, f"all__{k}.npy"), v)


@cli.command(singleton=True)
def preprocess_data(config: DataProcessConfig) -> None:
    rng = random.Random(config.seed)
    splits = split_records(config)

    # All splits share one vocabulary, so they are encoded before any of them is written.
    encoded = {}
    for set_name, records in splits.items():
        num_aug = config.num_aug if set_name == "train" else 0

        groups = []
        for record in tqdm(records, desc=set_name):
            examples = build_examples(record, num_aug, rng)
            if examples is not None:
                groups.append(examples)

        encoded[set_name] = groups

    tokens = {t for groups in encoded.values() for g in groups for pair in g for grid in pair for t in grid.ravel()}
    vocab = {t: i + 1 for i, t in enumerate(sorted(tokens))}

    os.makedirs(config.output_dir, exist_ok=True)
    with open(os.path.join(config.output_dir, "vocab.json"), "w") as f:
        json.dump(vocab, f)
    with open(os.path.join(config.output_dir, "identifiers.json"), "w") as f:
        json.dump(["<blank>"], f)

    for set_name, groups in encoded.items():
        save_subset(set_name, groups, vocab, config)


if __name__ == "__main__":
    cli()
