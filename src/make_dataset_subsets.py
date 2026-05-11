import argparse
from pathlib import Path
import random

import torch


def load_dataset(path: Path):
    dataset = torch.load(path, map_location="cpu")
    if not isinstance(dataset, list):
        raise TypeError(
            f"Expected a list dataset, but got {type(dataset).__name__}."
        )
    return dataset


def sample_subset(dataset, fraction: float, seed: int):
    subset_size = max(1, int(len(dataset) * fraction))
    rng = random.Random(seed)
    indices = rng.sample(range(len(dataset)), subset_size)
    return [dataset[i] for i in indices]


def save_subset(dataset, output_path: Path):
    torch.save(dataset, output_path)


def main():
    parser = argparse.ArgumentParser(
        description="Create smaller training subsets from expert_dataset.pt."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("expert_dataset.pt"),
        help="Path to the source .pt dataset file.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used for subset sampling.",
    )
    args = parser.parse_args()

    dataset = load_dataset(args.input)
    dataset_dir = args.input.parent
    dataset_stem = args.input.stem

    fractions = {
        "50": 0.50,
        "25": 0.25,
    }

    print(f"Loaded {len(dataset)} samples from {args.input}")

    for label, fraction in fractions.items():
        subset = sample_subset(dataset, fraction, args.seed)
        output_path = dataset_dir / f"{dataset_stem}_{label}.pt"
        save_subset(subset, output_path)
        print(
            f"Saved {len(subset)} samples "
            f"({int(fraction * 100)}%) to {output_path}"
        )


if __name__ == "__main__":
    main()
