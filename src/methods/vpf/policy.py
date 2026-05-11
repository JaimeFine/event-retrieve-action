from __future__ import annotations

import torch


class VPFPolicy:
    """
    Stateless virtual potential field controller implemented directly from the
    event-list representation used by the shared Isaac environment.

    Event layout per intruder:
    [type_id, rel_pos(3), rel_vel(3), ego_vel(3), rel_goal(3)]
    """

    def __init__(
        self,
        device_name: str,
        detection_threshold: float = 5.0,
        attractive_gain: float = 3.0,
        repulsive_gain: float = 15.0,
        max_speed: float = 5.0,
        stuck_threshold: float = 1.0,
        goal_far_threshold: float = 2.0,
        tangential_gain: float = 2.0,
    ):
        self._device = torch.device(device_name)
        self.detection_threshold = float(detection_threshold)
        self.attractive_gain = float(attractive_gain)
        self.repulsive_gain = float(repulsive_gain)
        self.max_speed = float(max_speed)
        self.stuck_threshold = float(stuck_threshold)
        self.goal_far_threshold = float(goal_far_threshold)
        self.tangential_gain = float(tangential_gain)

    def reset(self) -> None:
        return

    def predict(self, event_list: torch.Tensor | None) -> torch.Tensor:
        if event_list is None or event_list.numel() == 0:
            return torch.zeros(3, device=self._device)

        events = event_list.detach().to(self._device, dtype=torch.float32)
        if events.dim() == 1:
            events = events.unsqueeze(0)

        rel_goal = events[0, 10:13]
        dist_to_goal = torch.norm(rel_goal)

        if dist_to_goal > 0.5:
            f_attr = (rel_goal / (dist_to_goal + 1e-8)) * self.attractive_gain
        else:
            f_attr = torch.zeros(3, device=self._device)

        rel_positions = events[:, 1:4]
        dists = torch.norm(rel_positions, dim=1)
        mask = dists < self.detection_threshold

        f_rep = torch.zeros(3, device=self._device)
        if torch.any(mask):
            safe_rel_positions = rel_positions[mask]
            safe_dists = dists[mask].unsqueeze(1)
            repulsion_mag = self.repulsive_gain / (safe_dists.pow(2) + 1e-6)
            unit_away = (-safe_rel_positions) / (safe_dists + 1e-8)
            f_rep = torch.sum(unit_away * repulsion_mag, dim=0)

        action = f_attr + f_rep

        if torch.norm(action) < self.stuck_threshold and dist_to_goal > self.goal_far_threshold:
            perpendicular = torch.stack(
                (-rel_goal[1], rel_goal[0], torch.tensor(0.0, device=self._device))
            ).to(dtype=torch.float32)
            perpendicular_norm = torch.norm(perpendicular)
            if perpendicular_norm > 1e-8:
                action = action + (perpendicular / perpendicular_norm) * self.tangential_gain

        speed = torch.norm(action)
        if speed > self.max_speed:
            action = (action / (speed + 1e-8)) * self.max_speed

        return action.view(-1)
