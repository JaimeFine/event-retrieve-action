# pyright: reportMissingImports=false
from __future__ import annotations

import numpy as np
import torch
from isaacsim.core.api.objects import DynamicSphere
from isaacsim.core.prims import RigidPrim
from omni.isaac.core import World

from bruce_code.sim_shared.challenger.situation import RuleBasedAdversarialSpawner
from bruce_code.sim_shared.constants import (
    NUM_INTRUDERS,
    SAFETY_THRESHOLD,
    detection_threshold,
    device,
    ego,
)
from bruce_code.sim_shared.difficulty import RuleBasedDifficultyScheduler
from bruce_code.sim_shared.intruders import (
    BirdIntruder,
    DroneIntruder,
    MultiAgentIntruderController,
    StaticObstacle,
    apply_multiagent_intruder_behavior,
)


class SharedIsaacEnvironment:
    def __init__(
        self,
        agent,
        seed: int,
        difficulty_mode: str = "curriculum",
        difficulty_level: str = "medium",
        total_curriculum_steps: int = 500,
        num_intruders: int = NUM_INTRUDERS,
    ):
        self.agent = agent
        self.dt = 0.05
        self.world = World(stage_units_in_meters=1.0, physics_dt=self.dt)

        self.d_threshold = detection_threshold
        self.active_scenario_intruders = []
        self.ego_start = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        self.ego_goal = np.array([100.0, 0.0, 5.0], dtype=np.float32)

        self.rng = np.random.RandomState(seed)
        self.scheduler = RuleBasedDifficultyScheduler(
            total_steps=total_curriculum_steps,
            mode=difficulty_mode,
            fixed_level=difficulty_level,
        )
        self.spawner = RuleBasedAdversarialSpawner(self.rng)

        self.num_intruders = num_intruders
        self.intruder_controller = MultiAgentIntruderController(lr=1e-3)
        self.last_difficulty_profile = self.scheduler.get_profile(0)
        self.last_episode_summary = {}
        self.last_episode_trajectory: list[dict] = []
        self.intruder_retire_distance = 12.0
        self.intruder_retire_below_z = -5.0

    def setup_environment(self):
        self.world.scene.add_default_ground_plane()

        self.ego = DynamicSphere(
            prim_path="/World/ego_drone",
            name="ego_drone",
            position=np.array([0.0, 0.0, 1.5]),
            radius=0.25,
            color=ego,
        )
        self.world.scene.add(self.ego)

        self.intruders = []
        hidden_pos = [0.0, 0.0, -100.0]

        for i in range(self.num_intruders):
            self.intruders.append(DroneIntruder(f"pool_drone_{i}", hidden_pos))
            self.intruders.append(BirdIntruder(f"pool_bird_{i}", hidden_pos))
            self.intruders.append(StaticObstacle(f"pool_static_{i}", hidden_pos))

        for intruder in self.intruders:
            self.world.scene.add(intruder.prim)

        self.ego_view = RigidPrim(
            prim_paths_expr="/World/ego_drone",
            name="ego_view",
            track_contact_forces=False,
        )
        self.world.scene.add(self.ego_view)

    def load_scenario(self, seed):
        self.rng = np.random.RandomState(seed)
        self.spawner = RuleBasedAdversarialSpawner(self.rng)
        self.active_scenario_intruders = []
        self.ego.set_world_pose(position=self.ego_start)

        for intruder in self.intruders:
            intruder.set_state(np.array([0.0, 0.0, -100.0]), np.zeros(3))

        self.ego.set_world_pose(position=self.ego_start)
        self.ego.set_linear_velocity(np.zeros(3))
        self.intruder_controller.step_count = 0

    def detection(self):
        ego_pos, _ = self.ego.get_world_pose()
        ego_vel = self.ego.get_linear_velocity()

        event_list = []
        radii_list = []

        for intruder in self.active_scenario_intruders:
            pos, vel = intruder.get_state()
            rel_pos = pos - ego_pos
            dist = np.linalg.norm(rel_pos)

            if dist <= self.d_threshold:
                rel_vel = vel - ego_vel

                if isinstance(intruder, DroneIntruder):
                    type_id = 0.0
                elif isinstance(intruder, BirdIntruder):
                    type_id = 1.0
                else:
                    type_id = 2.0

                rel_goal = self.ego_goal - ego_pos
                event = np.concatenate([[type_id], rel_pos, rel_vel, ego_vel, rel_goal])

                event_list.append(event)
                radii_list.append(intruder.radius)

        if len(event_list) == 0:
            return None, None

        return (
            torch.from_numpy(np.array(event_list)).float().to(device),
            torch.from_numpy(np.array(radii_list)).float().to(device),
        )

    def manage_intruders(self, current_step):
        profile = self.scheduler.get_profile(current_step)
        self.last_difficulty_profile = profile
        self.intruder_controller.configure_for_profile(profile)

        ego_pos, _ = self.ego.get_world_pose()
        ego_vel = self.ego.get_linear_velocity()
        self._retire_stale_intruders(ego_pos, ego_vel)
        target_active = min(self.num_intruders, profile.active_intruders)

        while len(self.active_scenario_intruders) < target_active:
            cases = self.spawner.spawn_event(ego_pos, ego_vel, profile)

            for spawn_pos, velocity in cases:
                available = [inst for inst in self.intruders if inst not in self.active_scenario_intruders]
                if not available or len(self.active_scenario_intruders) >= target_active:
                    break

                new_intruder = available[0]
                new_intruder.set_state(spawn_pos, velocity)
                self.active_scenario_intruders.append(new_intruder)

    def _retire_stale_intruders(self, ego_pos, ego_vel) -> None:
        retained_intruders = []
        ego_speed = float(np.linalg.norm(ego_vel))
        ego_dir = ego_vel / (ego_speed + 1e-6) if ego_speed > 1e-6 else np.array([1.0, 0.0, 0.0], dtype=np.float32)

        for intruder in self.active_scenario_intruders:
            intruder_pos, intruder_vel = intruder.get_state()
            intruder_pos = np.array(intruder_pos, dtype=np.float32)
            intruder_vel = np.array(intruder_vel, dtype=np.float32)
            rel_pos = intruder_pos - ego_pos
            dist = float(np.linalg.norm(rel_pos))
            forward_offset = float(np.dot(rel_pos, ego_dir))
            relative_speed = float(np.linalg.norm(intruder_vel - ego_vel))

            is_hidden = intruder_pos[2] < self.intruder_retire_below_z
            is_far_from_ego = dist > self.intruder_retire_distance
            is_far_behind = forward_offset < -6.0 and dist > self.d_threshold * 1.5
            is_static_and_cleared = intruder.__class__.__name__ == "StaticObstacle" and dist > self.d_threshold * 1.75
            is_receding_fast = forward_offset < -2.0 and relative_speed > 1.0 and dist > self.d_threshold * 1.25

            if is_hidden or is_far_from_ego or is_far_behind or is_static_and_cleared or is_receding_fast:
                intruder.set_state(np.array([0.0, 0.0, -100.0], dtype=np.float32), np.zeros(3, dtype=np.float32))
                continue

            retained_intruders.append(intruder)

        self.active_scenario_intruders = retained_intruders

    def _extract_proximity_metrics(self, event_list, radii):
        if event_list is None or radii is None:
            return 0, None, None

        events = event_list
        if events.dim() == 1:
            events = events.unsqueeze(0)

        rel_positions = events[:, 1:4]
        dists = torch.norm(rel_positions, dim=1)
        surface_dists = dists - radii.view(-1) - 0.25

        return (
            int(events.shape[0]),
            float(torch.min(dists).item()),
            float(torch.min(surface_dists).item()),
        )

    def _record_trajectory_point(
        self,
        *,
        step_index: int,
        decision_step_count: int,
        ego_pos,
        ego_vel,
        base_vel,
        policy_action,
        final_action,
        goal_distance: float,
        detected_intruders: int,
        min_center_distance: float | None,
        min_surface_distance: float | None,
        warning_flag: int,
        collision_flag: int,
        success_flag: int,
        termination_reason: str,
    ) -> None:
        point = {
            "step_index": int(step_index),
            "decision_step_count": int(decision_step_count),
            "simulation_time_s": float((step_index + 1) * self.dt),
            "ego_x": float(ego_pos[0]),
            "ego_y": float(ego_pos[1]),
            "ego_z": float(ego_pos[2]),
            "ego_vx": float(ego_vel[0]),
            "ego_vy": float(ego_vel[1]),
            "ego_vz": float(ego_vel[2]),
            "goal_x": float(self.ego_goal[0]),
            "goal_y": float(self.ego_goal[1]),
            "goal_z": float(self.ego_goal[2]),
            "goal_distance": float(goal_distance),
            "base_vx": float(base_vel[0]),
            "base_vy": float(base_vel[1]),
            "base_vz": float(base_vel[2]),
            "policy_vx": float(policy_action[0]),
            "policy_vy": float(policy_action[1]),
            "policy_vz": float(policy_action[2]),
            "command_vx": float(final_action[0]),
            "command_vy": float(final_action[1]),
            "command_vz": float(final_action[2]),
            "active_intruders": int(len(self.active_scenario_intruders)),
            "detected_intruders": int(detected_intruders),
            "min_intruder_center_distance": (
                None if min_center_distance is None else float(min_center_distance)
            ),
            "min_intruder_surface_distance": (
                None if min_surface_distance is None else float(min_surface_distance)
            ),
            "warning_flag": int(warning_flag),
            "collision_flag": int(collision_flag),
            "success_flag": int(success_flag),
            "difficulty_name": str(self.last_difficulty_profile.name),
            "difficulty_scalar": float(self.last_difficulty_profile.scalar),
            "termination_reason": str(termination_reason),
        }
        self.last_episode_trajectory.append(point)

    def run(self, steps, episode_seed):
        WARNING, COLLISION = 0, 0

        self.load_scenario(episode_seed)
        self.world.reset()
        self.last_episode_trajectory = []

        ego_pos, _ = self.ego.get_world_pose()
        prev_dist = np.linalg.norm(self.ego_goal - ego_pos)
        dist_to_goal = float(prev_dist)

        index = None
        z_t = None

        success = 0
        total_step = 0
        reward_trace = []

        for i in range(steps):
            self.manage_intruders(current_step=i)

            intruder_loss = 0.0
            if len(self.active_scenario_intruders) > 0:
                intruder_loss = apply_multiagent_intruder_behavior(
                    self.intruder_controller,
                    self.ego,
                    self.active_scenario_intruders,
                )

            event_list, radii = self.detection()
            ego_pos, _ = self.ego.get_world_pose()

            dir_to_goal = self.ego_goal - ego_pos
            dir_to_goal /= np.linalg.norm(dir_to_goal) + 1e-6
            base_vel = torch.from_numpy(dir_to_goal).float().to(device) * 3.0

            if event_list is None:
                final_action = base_vel
                policy_action = torch.zeros_like(base_vel)
                z_t = None
            else:
                total_step += 1
                selection = self.agent.select_action(event_list, k=5)
                if (
                    isinstance(selection, tuple)
                    and len(selection) == 2
                    and hasattr(selection[1], "query_latent")
                ):
                    action = selection[0]
                    retrieval = selection[1]
                    z_t = retrieval.query_latent
                    index = retrieval.indices
                else:
                    action, z_t, _, _, index = selection

                if not isinstance(action, torch.Tensor):
                    action = torch.tensor(action, dtype=torch.float32, device=base_vel.device)
                else:
                    action = action.to(base_vel.device, dtype=torch.float32)

                policy_action = action
                final_action = base_vel + action

            self.ego_view.set_linear_velocities(final_action.detach().cpu().numpy().reshape(1, -1))
            self.world.step(render=False)

            ego_pos_after, _ = self.ego.get_world_pose()
            ego_vel_after = self.ego.get_linear_velocity()
            event_next, radii_next = self.detection()

            dist_to_goal = np.linalg.norm(self.ego_goal - ego_pos_after)
            detected_intruders, min_center_distance, min_surface_distance = self._extract_proximity_metrics(
                event_next,
                radii_next,
            )
            collision_flag = int(min_surface_distance is not None and min_surface_distance < 0.0)
            warning_flag = int(
                min_surface_distance is not None and 0.0 <= min_surface_distance < SAFETY_THRESHOLD
            )
            success_flag = int(dist_to_goal < 2.0)
            termination_reason = "goal_reached" if success_flag else ""
            self._record_trajectory_point(
                step_index=i,
                decision_step_count=total_step,
                ego_pos=ego_pos_after,
                ego_vel=ego_vel_after,
                base_vel=base_vel.detach().cpu().numpy(),
                policy_action=policy_action.detach().cpu().numpy(),
                final_action=final_action.detach().cpu().numpy(),
                goal_distance=float(dist_to_goal),
                detected_intruders=detected_intruders,
                min_center_distance=min_center_distance,
                min_surface_distance=min_surface_distance,
                warning_flag=warning_flag,
                collision_flag=collision_flag,
                success_flag=success_flag,
                termination_reason=termination_reason,
            )

            if dist_to_goal < 2.0:
                success = 1
                reward = torch.tensor([10.0], device=device)
                reward_trace.append(float(reward.item()))
                break

            if event_next is None:
                reward = torch.tensor([0.5], dtype=torch.float32, device=device)
            else:
                rel_positions = event_next[:, 1:4]
                dists = torch.norm(rel_positions, dim=1)
                surface_dists = dists - radii_next - 0.25
                min_dist = torch.min(surface_dists)

                progress = prev_dist - dist_to_goal
                prev_dist = dist_to_goal

                if min_dist < 0:
                    reward_val = -10.0
                    if hasattr(self.agent.memory, "penalize_by_indices"):
                        self.agent.memory.penalize_by_indices(index, factor=0.01)
                    COLLISION += 1
                elif min_dist < SAFETY_THRESHOLD:
                    reward_val = -1.0 * (SAFETY_THRESHOLD - min_dist) ** 2
                    if hasattr(self.agent.memory, "penalize_by_indices"):
                        self.agent.memory.penalize_by_indices(index, factor=0.5)
                    WARNING += 1
                else:
                    reward_val = 0.1 * progress

                reward = torch.tensor([reward_val], device=device)

            reward_trace.append(float(reward.item()))
            self.scheduler.update_performance(float(reward.item()))

        intruder_loss = self.intruder_controller.update()
        if torch.is_tensor(intruder_loss):
            intruder_loss = intruder_loss.item()

        avg_r_phys = 0.0
        avg_j_perf = 0.0
        self.last_episode_summary = {
            "difficulty_name": self.last_difficulty_profile.name,
            "difficulty_scalar": float(self.last_difficulty_profile.scalar),
            "active_intruders": int(self.last_difficulty_profile.active_intruders),
            "mean_reward": float(np.mean(reward_trace)) if reward_trace else 0.0,
            "trajectory_points": int(len(self.last_episode_trajectory)),
            "env_steps": int(len(self.last_episode_trajectory)),
            "decision_steps": int(total_step),
        }

        return success, COLLISION, WARNING, dist_to_goal, total_step, avg_r_phys, avg_j_perf, intruder_loss
