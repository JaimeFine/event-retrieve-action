# ERA Knowledge Bank Scalability

This folder is an additive workflow for preparing the paper section:

- `Scalability Analysis of the ERA Knowledge Bank`

It does not modify any existing ERA source files. It benchmarks the real stored
knowledge-bank snapshots in:

- `datasets/checkpoints/full`

## Files

- `benchmark_kb_scalability.py`
  Generates a table-ready CSV with bank size, latent-space dimensionality,
  memory footprint, and single-query retrieval latency.
- `plot_kb_scalability.py`
  Builds a dual-axis figure from that CSV.
- `outputs/`
  Generated artifacts live here after running the scripts.

## Default Outputs

- `outputs/era_kb_scalability.csv`
- `outputs/era_kb_scalability_summary.md`
- `outputs/era_kb_scalability_dual_axis.pdf`
- `outputs/era_kb_scalability_dual_axis.png`

## Run

From `Projects/ERA/src/deployment/scalability_analysis`:

```powershell
python benchmark_kb_scalability.py
python plot_kb_scalability.py
```

For Jetson Orin Nano Super GPU explicitly:

```powershell
python benchmark_kb_scalability.py --device gpu
python plot_kb_scalability.py
```

Or from anywhere:

```powershell
python "C:\Users\13647\OneDrive\Desktop\MiMundo\Projects\ERA\src\deployment\scalability_analysis\benchmark_kb_scalability.py"
python "C:\Users\13647\OneDrive\Desktop\MiMundo\Projects\ERA\src\deployment\scalability_analysis\plot_kb_scalability.py"
```

With explicit GPU selection:

```powershell
python "C:\Users\13647\OneDrive\Desktop\MiMundo\Projects\ERA\src\deployment\scalability_analysis\benchmark_kb_scalability.py" --device gpu
python "C:\Users\13647\OneDrive\Desktop\MiMundo\Projects\ERA\src\deployment\scalability_analysis\plot_kb_scalability.py"
```

## Notes

- The benchmark measures the ERA retrieval kernel directly: single-query
  `torch.cdist` nearest-neighbor retrieval with top-`k=5`.
- `payload_memory_mb` is the in-memory tensor footprint of `latents`,
  `actions`, and `reliability`, which is the most useful quantity for your
  table and systems discussion.
- The current default is to benchmark on CUDA if available, otherwise CPU.
- On Jetson, `--device gpu` and `--device cuda` mean the same thing.
- The benchmark defaults to `datasets/checkpoints/full`, so it stays aligned
  with the deployment-facing checkpoint bundle after the move to
  `src/deployment/`.
