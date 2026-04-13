import numpy as np
import torch
from macro import device, lambda_phys, lambda_perf, BATCH_SIZE
from macro import SAFETY_THRESHOLD, COLLISION_CRITICAL, EPOCHS
from intruders import DroneIntruder, BirdIntruder
from agents import EventCentricAgent
from intruders import apply_multiagent_intruder_behavior

class Trainer():
    def __init__(self):
        self.agent = EventCentricAgent(latent_dim=128)
        self.ego_start = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        self.optimizer = torch.optim.Adam(self.agent.parameters(), lr=1e-5)
        self.experience_buffer = []

    def detection(self):
        """Creates the Event List E_t from teh simulation state."""
        ego_pos, _ = self.ego.get_world_pose()
        ego_vel = self.ego.get_linear_velocity()
        event_list = []

        for intruder in self.active_scenario_intruders:
            pos, vel = intruder.get_state()
            dist = np.linalg.norm(pos - ego_pos)

            if dist <= self.d_threshold:
                # Use RELATIVE position and RELATIVE velocity
                rel_pos = pos - ego_pos
                rel_vel = vel - ego_vel

                if isinstance(intruder, DroneIntruder):
                    type_id = 0.0
                elif isinstance(intruder, BirdIntruder):
                    type_id = 1.0
                else:
                    type_id = 2.0

                # Concatenate intruder event with S_global (ego state)
                # Vector: [
                # id, rel_x, rel_y, rel_z, rel_vx, rel_vy, rel_vz,
                # ego_x, ego_y, ego_z, ego_vx, ego_vy, ego_vz
                # ]
                rel_goal = (self.ego_goal - ego_pos)
                s_global = ego_vel.tolist() + rel_goal.tolist()
                event = [type_id] + rel_pos.tolist() + \
                    rel_vel.tolist() + s_global
                event_list.append(event)

        return torch.tensor(
            event_list, dtype=torch.float32, device=device
        ) if event_list else None

    def train_agent(self, batch_size = BATCH_SIZE, epochs=EPOCHS):
        for _ in range(epochs):
            # [GPU OPTIMIZATION] Vectorized experience sampling
            indices = np.random.choice(
                len(self.experience_buffer), batch_size, replace=False
            )
            batch = [self.experience_buffer[i] for i in indices]

            self.optimizer.zero_grad()
            batch_loss = torch.tensor(0.0, device=device)
            valid_samples = 0

            for (E_t, a_t, r_t, E_next) in batch:
                if E_t is None or E_next is None: continue
                z_t = self.agent.encoder(E_t)

                # --- Retrieve ---
                weights, actions, latents = self.agent.memory.retrieve(z_t, k=5)
                if len(actions) == 0: continue

                # --- Physics consistency ---
                # Vectorized physics consistency penalty
                weights_t = weights if torch.is_tensor(weights) \
                    else torch.stack(weights)
                latents_t = torch.stack(latents).squeeze(1)
                R_phys = torch.sum(weights_t * torch.norm(z_t - latents_t, dim=1))

                # --- Performance ---
                log_probs = torch.log(weights + 1e-8)
                J_perf = torch.sum(log_probs) * r_t

                # Total loss
                batch_loss = batch_loss + lambda_phys * R_phys - \
                    lambda_perf * J_perf
                valid_samples += 1

            if valid_samples > 0:
                # Perform the constrained manifold update
                total_loss = batch_loss / valid_samples
                print(f"total_loss requires_grad: {total_loss.requires_grad}")
                print(f"R_phys requires_grad: {R_phys.requires_grad}")
                total_loss.backward()
                self.optimizer.step()

                # 5. Enforce Contractive Dynamics
                # Ensures ||z_{t+1}||^2 - ||z_t||^2 < 0 as per Eq. 11
                self.agent.enforce_contractive_dynamics()
                
                # print(f"[TRAINING] Loss: {total_loss.item():.4f}")

        # print(f"[TRAINING EPOCH COMPLETE]")

    """Total, collision, warning are only here!!!"""
    def run(self, steps, episode_seed, TOTAL=0, COLLISION=0, WARNING=0):   # NOTE: 200 for 0.5 dt is 10 seconds
        self.load_scenario(episode_seed)
        self.world.reset()
        self.experience_buffer = [] # Stores S_t

        goal_reached = 0
        dist_to_goal = 0.0
        ego_pos, _ = self.ego.get_world_pose()
        prev_dist = np.linalg.norm(self.ego_goal - ego_pos)
        success = 0
        num = 0

        for i in range(steps):
            self.manage_intruders(current_step=i)

            if len(self.active_scenario_intruders) > 0:
                intruder_loss = apply_multiagent_intruder_behavior(
                    self.intruder_controller,
                    self.ego,
                    self.active_scenario_intruders
                )

            # 1. Perception: Get the state at time t
            event_list = self.detection()
            z_t = None

            ego_pos, _ = self.ego.get_world_pose()
            dir_to_goal = self.ego_goal - ego_pos
            dir_to_goal /= (np.linalg.norm(dir_to_goal) + 1e-6)

            # --- External engine force ---
            # base engine pulling at 5.0 m/s toward the goal
            base_vel = torch.tensor(
                    dir_to_goal * 3.0, dtype=torch.float32, device=device
            )

            if event_list is None:
                final_action = base_vel
            else:
                # 2. Decision Making: Calculate Action at time t
                action, z_t, _, _ = self.agent.select_action(event_list, k=5)

                final_action = base_vel + action

            self.ego_view.set_linear_velocities(final_action.view(1, 3).cpu())

            # 3. Physics: Step the environment to apply a_t
            self.world.step(render=False)
            # 4. Perception: Get the resulting state at time t+1
            event_next = self.detection()

            # --- GOAL REACH CHECK ---
            dist_to_goal = np.linalg.norm(self.ego_goal - ego_pos)
            if dist_to_goal < 2.0: # 2 meter threshold
                print(f"        [GOAL REACHED] at step {i}!")
                goal_reached = 1
                success = 1
                # Give a massive reward for finishing
                reward = torch.tensor([50.0], device=device) 
                break

            # 5. Reward: Calculate r_t (NOTE: A simple heuristic for now)
            if event_next is None:
                reward = torch.tensor(
                    [1.0], dtype=torch.float32, device=device
                )  # Successfully cleared the threat
            else:
                # event_next: (N, 13)
                rel_positions = event_next[:, 1:4]  # (N, 3)

                dists = torch.norm(rel_positions, dim=1)
                min_dist = torch.min(dists).item()

                progress = prev_dist - dist_to_goal
                prev_dist = dist_to_goal

                if min_dist < COLLISION_CRITICAL:
                    reward_val = -10.0
                    print("[COLLISION] Collision encountered!")
                    COLLISION += 1
                elif min_dist < SAFETY_THRESHOLD:
                    reward_val = -1.0 * (SAFETY_THRESHOLD - min_dist) / \
                        (min_dist + 1e-6)
                    print("[WARNING] Collision might happen!!!")
                    WARNING += 1
                else:
                    reward_val = 1 * progress   # Add progress

                TOTAL += 1     # For the debugging

                reward = torch.tensor(
                    [reward_val], dtype=torch.float32, device=device
                )

                self.scheduler.update_performance(reward.item())

            # 6. Logging: Save the Status Code
            self.experience_buffer.append(
                (event_list, final_action, reward, event_next)
            )
            num += 1
            
            # if num % (BATCH_SIZE * 2) == 0:
                # self.train_agent()

            z_next = self.agent.encoder(
                event_next
            ) if event_next is not None else None

            if z_t is not None:
                self.agent.memory.add_experiences(
                    z_t.detach(),
                    final_action.detach(),
                    reward,
                    z_next.detach() if z_next is not None else None
                )

            if num % BATCH_SIZE * 5 == 0:
                self.agent.memory.build_index()

            # print(f"[LOGGED S_t] Reward: {reward.item()}")

        self.train_agent()

        status = "SUCCESS" if goal_reached else "TIMEOUT/CRASH"
        print(f"--- Episode Summary: {status} | \
            Final Dist: {dist_to_goal:.2f}m ---")
        print(f"[COMPLETE] Collected {num} experiences.")

        return success, COLLISION, WARNING, dist_to_goal
