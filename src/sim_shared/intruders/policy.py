from __future__ import annotations

import numpy as np
import torch

from bruce_code.sim_shared.constants import SAFETY_THRESHOLD


class IntruderPolicy:
    def __init__(self, response_gain=0.35, max_delta=1.0):
        self.response_gain = float(response_gain)
        self.max_delta = float(max_delta)

    def select_action(self, state):
        if torch.is_tensor(state):
            state_np = state.detach().cpu().numpy()
            device = state.device
        else:
            state_np = np.asarray(state, dtype=np.float32)
            device = None

        rel_pos = state_np[:3]
        intr_vel = state_np[3:6]
        dist = np.linalg.norm(rel_pos) + 1e-6
        to_ego = -rel_pos / dist

        lateral = np.array([-to_ego[1], to_ego[0], 0.0], dtype=np.float32)
        lateral_norm = np.linalg.norm(lateral)
        if lateral_norm < 1e-6:
            lateral = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        else:
            lateral = lateral / lateral_norm

        desired_vel = 2.0 * to_ego + 0.35 * lateral
        if dist < SAFETY_THRESHOLD * 0.75:
            desired_vel *= 0.5

        action = self.response_gain * (desired_vel - intr_vel)
        action_norm = np.linalg.norm(action)
        if action_norm > self.max_delta:
            action = action / action_norm * self.max_delta

        if device is not None:
            return torch.tensor(action, dtype=torch.float32, device=device)
        return action.astype(np.float32)


class MultiAgentIntruderController:
    def __init__(self, lr=1e-3, gamma=0.99, max_speed=4.0):
        self.policy = IntruderPolicy()
        self.gamma = gamma
        self.max_speed = float(max_speed)
        # These defaults act as a medium-difficulty fallback before the current
        # DifficultyProfile is applied. They were chosen to produce visibly
        # reactive intruders without causing unrealistic oscillation when the
        # controller is instantiated before profile configuration.
        self.response_gain = 0.35
        self.separation_gain = 0.28
        self.bird_wiggle_amp = 0.2
        self.bird_wiggle_freq = 0.05
        self.step_count = 0
        self._last_metric = 0.0

    def configure_for_profile(self, profile) -> None:
        self.response_gain = float(profile.response_gain)
        self.separation_gain = float(profile.separation_gain)
        self.bird_wiggle_amp = float(profile.bird_wiggle_amp)
        self.max_speed = float(profile.max_intruder_speed)

    def get_state(self, ego_pos, intruder_pos, intruder_vel):
        rel_pos = intruder_pos - ego_pos
        return np.concatenate([rel_pos, intruder_vel]).astype(np.float32)

    def select_action(self, state):
        action = self.policy.select_action(state)
        log_prob = torch.tensor(0.0, device=action.device) if torch.is_tensor(action) else 0.0
        return action, log_prob

    def compute_multiagent_reward(self, ego_pos, intruder_positions):
        rewards = []
        for pos in intruder_positions:
            dist = np.linalg.norm(pos - ego_pos)
            if dist < 0.5:
                rewards.append(10.0)
            elif dist < SAFETY_THRESHOLD:
                rewards.append(1.0 / (dist + 1e-6))
            else:
                rewards.append(-0.05 * dist)
        return rewards

    def store(self, log_probs, rewards):
        return

    def update(self):
        metric = float(self._last_metric)
        self._last_metric = 0.0
        return metric


def _clamp_speed(velocity, max_speed):
    speed = np.linalg.norm(velocity)
    if speed <= max_speed:
        return velocity
    return velocity / (speed + 1e-6) * max_speed


def _compute_separation(current_pos, positions, index, threshold=1.4):
    repulse = np.zeros(3, dtype=np.float32)
    for j, other_pos in enumerate(positions):
        if j == index:
            continue
        diff = current_pos - other_pos
        dist = np.linalg.norm(diff)
        if 1e-6 < dist < threshold:
            repulse += (diff / dist) * (threshold - dist)
    return repulse


def apply_multiagent_intruder_behavior(controller, ego, intruders):
    if len(intruders) == 0:
        controller._last_metric = 0.0
        return 0.0

    ego_pos, _ = ego.get_world_pose()
    positions = []
    velocities = []
    for intr in intruders:
        pos, vel = intr.get_state()
        positions.append(np.array(pos, dtype=np.float32))
        velocities.append(np.array(vel, dtype=np.float32))

    avg_delta = 0.0
    phase = controller.step_count * controller.bird_wiggle_freq

    for i, intr in enumerate(intruders):
        type_name = intr.__class__.__name__
        current_pos = positions[i]
        current_vel = velocities[i]

        if type_name == "StaticObstacle":
            intr.prim.set_linear_velocity(np.zeros(3, dtype=np.float32))
            continue

        rel_pos = current_pos - ego_pos
        dist = np.linalg.norm(rel_pos) + 1e-6
        chase_dir = -rel_pos / dist

        lateral = np.array([-chase_dir[1], chase_dir[0], 0.0], dtype=np.float32)
        lateral_norm = np.linalg.norm(lateral)
        if lateral_norm < 1e-6:
            lateral = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        else:
            lateral = lateral / lateral_norm

        separation = _compute_separation(current_pos, positions, i)

        if type_name == "BirdIntruder":
            # Birds keep a lower forward gain and a sinusoidal lateral term so
            # they remain less ballistic than drones but still become harder to
            # predict as bird_wiggle_amp increases with difficulty.
            wiggle = np.sin(phase + i * 0.5)
            desired_vel = 1.35 * chase_dir + (0.2 + controller.bird_wiggle_amp * wiggle) * lateral
            max_speed = min(controller.max_speed, 2.3)
        else:
            # Drones are intentionally more direct and faster than birds, which
            # makes the profile-controlled response_gain and max_speed the main
            # knobs for increasing adversarial pressure across difficulty levels.
            desired_vel = 2.05 * chase_dir + 0.18 * lateral
            max_speed = controller.max_speed

        desired_vel += controller.separation_gain * separation
        if dist < SAFETY_THRESHOLD * 0.75:
            desired_vel *= 0.6

        new_vel = current_vel + controller.response_gain * (desired_vel - current_vel)
        new_vel = _clamp_speed(new_vel, max_speed)

        intr.prim.set_linear_velocity(new_vel.astype(np.float32))
        avg_delta += float(np.linalg.norm(new_vel - current_vel))

    controller.step_count += 1
    controller._last_metric = avg_delta / max(len(intruders), 1)
    return controller._last_metric
