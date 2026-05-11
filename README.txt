ERA-only evaluation bundle

Folder purpose:
- Run the same online-evaluation idea as the original comparison flow
- But evaluate ERA only
- Without re-running PPO / BC / Acados

Main script:
- bruce_code\scripts\run_era_only_comparison.py

Included:
- local ERA bridge modules
- shared simulation modules
- external ERA package copy
- agent_finetuned.pt
- knowledge_bank_snapshot.pt

Default outputs:
- outputs\formal_<difficulty>_seed<seed>_e<episodes>_s<steps>.json
- outputs\formal_<difficulty>_seed<seed>_e<episodes>_s<steps>.csv
- outputs\*_trajectory.json
- outputs\*_trajectory_points.csv

Example command pattern:
- use Isaac Sim Python to run bruce_code\scripts\run_era_only_comparison.py

Trajectory export:
- enabled by default with --save-trajectories true
- ERA helpers: bruce_code\methods\era\trajectory.py
- VPF helpers: bruce_code\methods\vpf\trajectory.py
