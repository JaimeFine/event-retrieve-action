from __future__ import annotations

import importlib

import torch


class AcadosNMPCController:
    def __init__(
        self,
        horizon: int = 20,
        dt: float = 0.05,
        max_speed: float = 5.0,
        safety_radius: float = 1.0,
        backend: str = "auto",
        max_obstacles: int = 8,
        max_accel: float = 6.0,
    ):
        self.horizon = horizon
        self.dt = dt
        self.max_speed = max_speed
        self.safety_radius = safety_radius
        self.max_obstacles = max_obstacles
        self.max_accel = max_accel
        self.backend = self._resolve_backend(backend)

        self._acados_runtime = None
        self._acados_init_error = ""
        if self.backend == "acados":
            self._initialize_acados_runtime()

    def _resolve_backend(self, backend: str) -> str:
        if backend not in {"auto", "grid", "acados"}:
            raise ValueError(f"Unsupported backend: {backend!r}")
        if backend == "grid":
            return "grid"

        try:
            importlib.import_module("acados_template")
            importlib.import_module("casadi")
            return "acados"
        except Exception:
            return "grid"

    def _initialize_acados_runtime(self) -> None:
        try:
            module = importlib.import_module("bruce_code.methods.acados.solver")
            policy_cls = getattr(module, "AcadosPointMassPolicy")
            runtime = policy_cls(
                horizon=self.horizon,
                dt=self.dt,
                max_speed=self.max_speed,
                safety_radius=self.safety_radius,
                max_obstacles=self.max_obstacles,
                max_accel=self.max_accel,
            )
            if not runtime.is_ready:
                raise RuntimeError(runtime.last_error)
            self._acados_runtime = runtime
        except Exception as exc:
            self._acados_runtime = None
            self._acados_init_error = str(exc)
            self.backend = "grid"

    def _goal_vector(self, event_list: torch.Tensor | None) -> torch.Tensor:
        if event_list is not None and event_list.numel() > 0 and event_list.shape[-1] >= 13:
            goal = event_list[0, 10:13]
            if torch.norm(goal) > 1e-6:
                return goal
            return torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32, device=event_list.device)
        return torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32)

    def _grid_candidates(self, goal_vec: torch.Tensor) -> torch.Tensor:
        goal_dir = goal_vec / torch.norm(goal_vec).clamp_min(1e-6)
        lateral = torch.tensor(
            [-goal_dir[1], goal_dir[0], 0.0],
            dtype=goal_dir.dtype,
            device=goal_dir.device,
        )
        if torch.norm(lateral) < 1e-6:
            lateral = torch.tensor([0.0, 1.0, 0.0], dtype=goal_dir.dtype, device=goal_dir.device)
        lateral = lateral / torch.norm(lateral).clamp_min(1e-6)

        candidates = []
        for speed_scale in (0.25, 0.4, 0.55, 0.7, 0.85, 1.0):
            for lateral_scale in (-1.2, -0.8, -0.4, 0.0, 0.4, 0.8, 1.2):
                candidate = goal_dir + 0.85 * lateral_scale * lateral
                candidate = candidate / torch.norm(candidate).clamp_min(1e-6)
                candidates.append(candidate * (speed_scale * self.max_speed))
        return torch.stack(candidates, dim=0)

    def _score_candidate(self, candidate: torch.Tensor, event_list: torch.Tensor | None) -> torch.Tensor:
        goal_alignment = torch.dot(candidate, self._goal_vector(event_list))
        if event_list is None or event_list.numel() == 0:
            return goal_alignment

        rel_pos = event_list[:, 1:4]
        rel_vel = event_list[:, 4:7] if event_list.shape[-1] >= 7 else torch.zeros_like(rel_pos)
        ego_vel = event_list[0, 7:10] if event_list.shape[-1] >= 10 else torch.zeros(3, dtype=candidate.dtype)

        horizon_times = torch.linspace(
            self.dt,
            self.dt * self.horizon,
            steps=max(2, self.horizon // 2),
            dtype=rel_pos.dtype,
            device=rel_pos.device,
        )

        min_clearance = None
        for t_h in horizon_times:
            future_rel = rel_pos - (candidate.view(1, 3) - rel_vel) * t_h
            clearance = torch.norm(future_rel, dim=1).min()
            min_clearance = clearance if min_clearance is None else torch.minimum(min_clearance, clearance)

        smooth_penalty = torch.norm(candidate - ego_vel) * 0.15
        collision_penalty = torch.relu(
            torch.tensor(self.safety_radius, dtype=candidate.dtype, device=candidate.device) - min_clearance
        ) * 60.0

        return goal_alignment - collision_penalty - smooth_penalty

    @torch.no_grad()
    def act(self, event_list: torch.Tensor | None) -> torch.Tensor:
        if self.backend == "acados" and self._acados_runtime is not None:
            try:
                action = self._acados_runtime.act(event_list)
                if isinstance(action, torch.Tensor):
                    return action.view(-1).float()
                return torch.tensor(action, dtype=torch.float32).view(-1)
            except Exception:
                self.backend = "grid"

        goal_vec = self._goal_vector(event_list).float()
        candidates = self._grid_candidates(goal_vec)
        scores = torch.stack([self._score_candidate(candidate, event_list) for candidate in candidates])
        return candidates[int(torch.argmax(scores).item())].view(-1)
