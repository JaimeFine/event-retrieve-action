from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from .ocp import build_point_mass_ocp


class AcadosPointMassPolicy:
    def __init__(
        self,
        *,
        horizon: int,
        dt: float,
        max_speed: float,
        safety_radius: float,
        max_obstacles: int,
        max_accel: float,
    ):
        self.horizon = int(horizon)
        self.dt = float(dt)
        self.max_speed = float(max_speed)
        self.safety_radius = float(safety_radius)
        self.max_obstacles = int(max_obstacles)
        self.max_accel = float(max_accel)

        self._solver = None
        self._ready = False
        self._last_error = ""

        self._initialize_solver()

    @property
    def is_ready(self) -> bool:
        return self._ready

    @property
    def last_error(self) -> str:
        return self._last_error

    def _initialize_solver(self) -> None:
        try:
            import casadi as ca
            import acados_template
            from acados_template import AcadosOcpSolver

            ocp = build_point_mass_ocp(
                acados_template,
                ca,
                horizon=self.horizon,
                dt=self.dt,
                max_speed=self.max_speed,
                max_accel=self.max_accel,
                safety_radius=self.safety_radius,
                max_obstacles=self.max_obstacles,
            )

            build_dir = Path("code") / "methods" / "acados" / "generated"
            build_dir.mkdir(parents=True, exist_ok=True)
            json_path = build_dir / "point_mass_ocp.json"

            cwd = Path.cwd()
            os.chdir(build_dir)
            try:
                self._solver = AcadosOcpSolver(ocp, json_file=str(json_path.name))
            finally:
                os.chdir(cwd)

            self._ready = True
        except Exception as exc:
            self._ready = False
            self._last_error = str(exc)

    def _extract_state_goal_obstacles(self, event_list) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if event_list is None or event_list.numel() == 0:
            state = np.zeros((6,), dtype=np.float64)
            goal = np.array([5.0, 0.0, 0.0], dtype=np.float64)
            obstacles = np.full((self.max_obstacles, 3), 1e3, dtype=np.float64)
            return state, goal, obstacles

        events = event_list.detach().cpu().numpy()
        if events.ndim == 1:
            events = events.reshape(1, -1)

        ego_vel = events[0, 7:10] if events.shape[1] >= 10 else np.zeros((3,), dtype=np.float64)
        rel_goal = events[0, 10:13] if events.shape[1] >= 13 else np.array([5.0, 0.0, 0.0], dtype=np.float64)

        state = np.concatenate([np.zeros((3,), dtype=np.float64), ego_vel.astype(np.float64)], axis=0)

        obstacles = np.full((self.max_obstacles, 3), 1e3, dtype=np.float64)
        obs_count = min(self.max_obstacles, events.shape[0])
        if obs_count > 0:
            obstacles[:obs_count, :] = events[:obs_count, 1:4].astype(np.float64)

        return state, rel_goal.astype(np.float64), obstacles

    def act(self, event_list) -> np.ndarray:
        if not self._ready or self._solver is None:
            raise RuntimeError(self._last_error or "acados solver is not ready")

        state, goal, obstacles = self._extract_state_goal_obstacles(event_list)

        self._solver.set(0, "lbx", state)
        self._solver.set(0, "ubx", state)

        params = obstacles.reshape(-1)
        for i in range(self.horizon):
            self._solver.set(i, "p", params)

            yref = np.zeros((9,), dtype=np.float64)
            yref[:3] = goal
            yref[3:6] = 0.0
            self._solver.set(i, "yref", yref)

        yref_e = np.zeros((6,), dtype=np.float64)
        yref_e[:3] = goal
        self._solver.set(self.horizon, "yref", yref_e)

        status = int(self._solver.solve())
        if status != 0:
            raise RuntimeError(f"acados solve failed with status {status}")

        u0 = np.array(self._solver.get(0, "u"), dtype=np.float64).reshape(-1)
        delta_v = u0 * self.dt
        speed = np.linalg.norm(delta_v)
        if speed > self.max_speed:
            delta_v = delta_v / (speed + 1e-8) * self.max_speed
        return delta_v.astype(np.float32)
