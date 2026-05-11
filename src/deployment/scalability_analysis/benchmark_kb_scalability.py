from __future__ import annotations

import argparse
import csv
import random
import re
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[2]
DEFAULT_CHECKPOINTS_DIR = PROJECT_ROOT / "datasets" / "checkpoints" / "full"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "outputs"
DEFAULT_OUTPUT_CSV = DEFAULT_OUTPUT_DIR / "era_kb_scalability.csv"
DEFAULT_SUMMARY_MD = DEFAULT_OUTPUT_DIR / "era_kb_scalability_summary.md"


@dataclass(frozen=True)
class SnapshotSpec:
    label: str
    step: int
    path: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark ERA knowledge-bank scalability from checkpoint snapshots."
    )
    parser.add_argument(
        "--checkpoints-dir",
        type=Path,
        default=DEFAULT_CHECKPOINTS_DIR,
        help="Directory containing knowledge_bank_snapshot*.pt files.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=DEFAULT_OUTPUT_CSV,
        help="Destination CSV for table-ready metrics.",
    )
    parser.add_argument(
        "--output-summary",
        type=Path,
        default=DEFAULT_SUMMARY_MD,
        help="Destination markdown summary.",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda", "gpu"),
        default="auto",
        help="Benchmark device. Use 'gpu' or 'cuda' for Jetson GPU. Defaults to CUDA when available, else CPU.",
    )
    parser.add_argument(
        "--num-queries",
        type=int,
        default=256,
        help="Number of single-query retrieval calls benchmarked per snapshot.",
    )
    parser.add_argument(
        "--warmup-queries",
        type=int,
        default=32,
        help="Number of warmup retrieval calls before timing each snapshot.",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=5,
        help="Top-k used by the retrieval benchmark.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=7,
        help="Random seed for query selection.",
    )
    return parser.parse_args()


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "gpu":
        device_arg = "cuda"
    if device_arg == "cpu":
        return torch.device("cpu")
    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested, but no CUDA device is available.")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def sync_if_needed(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def checkpoint_sort_key(path: Path) -> tuple[int, int]:
    stem = path.stem
    if stem.endswith("pretrained"):
        return (0, -1)
    match = re.search(r"_(\d+)$", stem)
    step = int(match.group(1)) if match else 10**9
    return (1, step)


def discover_snapshots(checkpoints_dir: Path) -> list[SnapshotSpec]:
    paths = sorted(
        checkpoints_dir.glob("knowledge_bank_snapshot*.pt"),
        key=checkpoint_sort_key,
    )
    if not paths:
        raise FileNotFoundError(
            f"No knowledge bank snapshots found in {checkpoints_dir}."
        )

    snapshots: list[SnapshotSpec] = []
    for path in paths:
        stem = path.stem
        if stem.endswith("pretrained"):
            snapshots.append(SnapshotSpec(label="pretrained", step=-1, path=path))
            continue
        match = re.search(r"_(\d+)$", stem)
        if not match:
            continue
        step = int(match.group(1))
        snapshots.append(SnapshotSpec(label=str(step), step=step, path=path))
    return snapshots


def load_snapshot(path: Path) -> dict[str, torch.Tensor]:
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def tensor_payload_megabytes(tensors: Iterable[torch.Tensor]) -> float:
    payload_bytes = sum(t.numel() * t.element_size() for t in tensors)
    return payload_bytes / (1024.0 * 1024.0)


def build_query_indices(num_entries: int, num_queries: int, seed: int) -> torch.Tensor:
    query_count = min(num_entries, max(1, num_queries))
    rng = random.Random(seed + num_entries)
    indices = list(range(num_entries))
    rng.shuffle(indices)
    return torch.tensor(indices[:query_count], dtype=torch.long)


def benchmark_snapshot(
    latents: torch.Tensor,
    actions: torch.Tensor,
    device: torch.device,
    num_queries: int,
    warmup_queries: int,
    k: int,
    seed: int,
) -> dict[str, float]:
    latents = latents.to(device=device, dtype=torch.float32)
    actions = actions.to(device=device, dtype=torch.float32)
    query_indices = build_query_indices(latents.shape[0], num_queries, seed).to(device)
    actual_k = min(k, latents.shape[0])

    def retrieve_once(query: torch.Tensor) -> torch.Tensor:
        distances = torch.cdist(query.view(1, -1), latents, p=2).squeeze(0)
        topk_values, topk_indices = torch.topk(distances, actual_k, largest=False)
        weights = 1.0 / (topk_values + 1e-8)
        weights = weights / weights.sum()
        retrieved_actions = actions.index_select(0, topk_indices)
        return torch.sum(weights.unsqueeze(1) * retrieved_actions, dim=0)

    with torch.inference_mode():
        for idx in query_indices[: min(warmup_queries, query_indices.numel())]:
            _ = retrieve_once(latents[idx])
        sync_if_needed(device)

        timings_ms: list[float] = []
        for idx in query_indices:
            query = latents[idx]
            sync_if_needed(device)
            start = time.perf_counter()
            _ = retrieve_once(query)
            sync_if_needed(device)
            timings_ms.append((time.perf_counter() - start) * 1000.0)

    timings_ms_sorted = sorted(timings_ms)
    p95_index = min(len(timings_ms_sorted) - 1, max(0, int(len(timings_ms_sorted) * 0.95) - 1))

    return {
        "latency_mean_ms": statistics.mean(timings_ms),
        "latency_median_ms": statistics.median(timings_ms),
        "latency_std_ms": statistics.pstdev(timings_ms) if len(timings_ms) > 1 else 0.0,
        "latency_p95_ms": timings_ms_sorted[p95_index],
        "queries_per_second": 1000.0 / statistics.mean(timings_ms),
    }


def write_csv(rows: list[dict[str, object]], output_csv: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "checkpoint_label",
        "checkpoint_step",
        "snapshot_file",
        "bank_size",
        "latent_space_dimensions",
        "action_dimensions",
        "reliability_dimensions",
        "payload_memory_mb",
        "checkpoint_file_mb",
        "bytes_per_entry",
        "latency_mean_ms",
        "latency_median_ms",
        "latency_std_ms",
        "latency_p95_ms",
        "queries_per_second",
        "benchmark_device",
        "topk",
        "num_queries",
    ]
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_summary(rows: list[dict[str, object]], output_summary: Path, device: torch.device) -> None:
    output_summary.parent.mkdir(parents=True, exist_ok=True)
    first = rows[0]
    last = rows[-1]
    growth = float(last["bank_size"]) / float(first["bank_size"])
    latency_growth = float(last["latency_mean_ms"]) / float(first["latency_mean_ms"])
    memory_growth = float(last["payload_memory_mb"]) / float(first["payload_memory_mb"])

    lines = [
        "# ERA Knowledge Bank Scalability Summary",
        "",
        f"- Benchmark device: `{device.type}`",
        f"- Snapshots analyzed: `{len(rows)}`",
        f"- Latent dimensionality: `{int(first['latent_space_dimensions'])}`",
        f"- Smallest bank: `{int(first['bank_size']):,}` entries",
        f"- Largest bank: `{int(last['bank_size']):,}` entries",
        f"- Bank growth factor: `{growth:.2f}x`",
        f"- Mean retrieval latency growth factor: `{latency_growth:.2f}x`",
        f"- Payload memory growth factor: `{memory_growth:.2f}x`",
        "",
        "## Table Columns",
        "",
        "- `bank_size`: number of stored event-centric memory entries.",
        "- `latent_space_dimensions`: compact latent width of each entry.",
        "- `payload_memory_mb`: in-memory tensor footprint for `latents`, `actions`, and `reliability`.",
        "- `latency_mean_ms` / `latency_p95_ms`: single-query top-k retrieval time using the ERA retrieval kernel.",
        "",
        "## Suggested Reading of the Plot",
        "",
        "- Memory scales close to linearly with bank size because each entry stores a fixed 128-D latent plus action and reliability tensors.",
        "- Retrieval latency should be interpreted empirically for the tested device, not as a theoretical asymptotic claim.",
    ]
    output_summary.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    snapshots = discover_snapshots(args.checkpoints_dir)

    rows: list[dict[str, object]] = []

    for spec in snapshots:
        snapshot = load_snapshot(spec.path)
        latents = snapshot["latents"].detach().cpu()
        actions = snapshot["actions"].detach().cpu()
        reliability = snapshot["reliability"].detach().cpu()

        metrics = benchmark_snapshot(
            latents=latents,
            actions=actions,
            device=device,
            num_queries=args.num_queries,
            warmup_queries=args.warmup_queries,
            k=args.k,
            seed=args.seed,
        )

        payload_memory_mb = tensor_payload_megabytes((latents, actions, reliability))
        row = {
            "checkpoint_label": spec.label,
            "checkpoint_step": spec.step,
            "snapshot_file": spec.path.name,
            "bank_size": int(latents.shape[0]),
            "latent_space_dimensions": int(latents.shape[1]),
            "action_dimensions": int(actions.shape[1]),
            "reliability_dimensions": int(reliability.shape[1]),
            "payload_memory_mb": round(payload_memory_mb, 6),
            "checkpoint_file_mb": round(spec.path.stat().st_size / (1024.0 * 1024.0), 6),
            "bytes_per_entry": round(
                (
                    latents[0].numel() * latents.element_size()
                    + actions[0].numel() * actions.element_size()
                    + reliability[0].numel() * reliability.element_size()
                ),
                2,
            ),
            "benchmark_device": device.type,
            "topk": args.k,
            "num_queries": min(int(args.num_queries), int(latents.shape[0])),
        }
        for key, value in metrics.items():
            row[key] = round(value, 6)
        rows.append(row)
        print(
            f"[done] {spec.path.name}: "
            f"{row['bank_size']:,} entries, "
            f"{row['payload_memory_mb']:.3f} MB, "
            f"{row['latency_mean_ms']:.3f} ms/query"
        )

    write_csv(rows, args.output_csv)
    write_summary(rows, args.output_summary, device)
    print(f"[saved] CSV -> {args.output_csv}")
    print(f"[saved] Summary -> {args.output_summary}")


if __name__ == "__main__":
    main()
