# Acados Baseline Status

Current status in this repository:
- `AcadosNMPCController` now prefers a real acados OCP solver runtime (`code/methods/acados/solver.py`).
- If solver initialization fails, it automatically falls back to an improved grid-search controller.

Local `acados/` folder usage:
- The `acados/` directory under `autodl-tmp` is the upstream acados source tree.
- It provides Python templates (`acados_template`) and C solver generation/build tooling.
- This project reuses that installation/runtime through Python imports (`acados_template`, `casadi`).

Implementation files in this project:
- `code/methods/acados/model.py`  (point-mass dynamics model)
- `code/methods/acados/ocp.py`    (OCP cost/constraints/horizon)
- `code/methods/acados/solver.py` (acados solver wrapper + runtime solve)
- `code/baselines.py`             (`AcadosNMPCController` with acados->grid fallback)

Environment prerequisites (typical):
1. `acados_template` importable in Python
2. `casadi` importable in Python
3. acados shared libraries discoverable (`LD_LIBRARY_PATH` includes acados lib directory)
4. `ACADOS_SOURCE_DIR` points to your acados source root

How to verify backend:
- Run online comparison with `--acados-backend auto` or `--acados-backend acados`.
- Check output JSON field `acados.backend_used`:
  - `acados` => full solver path used
  - `grid`   => fallback path used
