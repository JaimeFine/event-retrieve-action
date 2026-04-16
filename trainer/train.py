import numpy as np
import torch
from macro import device, lambda_phys, lambda_perf, BATCH_SIZE
from macro import SAFETY_THRESHOLD, EPOCHS
from intruders import DroneIntruder, BirdIntruder
from agents import EventCentricAgent
from intruders import apply_multiagent_intruder_behavior

class Trainer():
    def __init__(self):
        self.agent = EventCentricAgent(latent_dim=128)
        self.ego_start = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        for param in self.agent.encoder.parameters():
            param.requires_grad = False
        self.agent.Gamma.requires_grad = False

        trainable_params = [p for p in self.agent.parameters() if p.requires_grad]
        self.optimizer = torch.optim.Adam(trainable_params, lr=1e-5)
        self.experience_buffer = []

    def detection(self):
        """Creates the Event List E_t from the simulation state."""
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

                # Concatenate intruder event with S_global (ego state)
                # Vector: [
                # id, rel_x, rel_y, rel_z, rel_vx, rel_vy, rel_vz,
                # ego_x, ego_y, ego_z, ego_vx, ego_vy, ego_vz
                # ]
                rel_goal = self.ego_goal - ego_pos
                event = np.concatenate([
                    [type_id],
                    rel_pos,
                    rel_vel,
                    ego_vel,
                    rel_goal
                ])
                
                event_list.append(event)
                radii_list.append(intruder.radius)

        if len(event_list) == 0:
            return None, None

        return (
            torch.from_numpy(np.array(event_list)).float().to(device),
            torch.from_numpy(np.array(radii_list)).float().to(device)
        )

    def train_agent(self, batch_size = BATCH_SIZE, epochs=EPOCHS):
        if len(self.experience_buffer) < batch_size:
            return 0.0, 0.0
        
        epoch_r_phys = 0.0
        epoch_j_perf = 0.0
        valid_updates = 0

        for _ in range(epochs):
            # [GPU OPTIMIZATION] Vectorized experience sampling
            indices = torch.randint(
                0, len(self.experience_buffer), (batch_size,), device=device
            )
            batch = [self.experience_buffer[i] for i in indices]

            batch = [b for b in batch if b[0] is not None and b[3] is not None]
            if len(batch) == 0: continue

            E_t_batch = torch.stack([b[0].mean(dim=0) for b in batch]).to(device)
            a_batch = torch.stack([b[1] for b in batch]).to(device)
            r_batch = torch.stack([b[2] for b in batch]).to(device)
            E_next_batch = torch.stack([b[3].mean(dim=0) for b in batch]).to(device)

            # NOTE: Using torch.no_grad() prevents memory leaks from frozen networks
            with torch.no_grad():
                z_batch = self.agent.encoder(E_t_batch)
                z_next_actual = self.agent.encoder(E_next_batch)

            weights_list, valid_indices = [], []
            for i in range(len(z_batch)):
                w, _, _, _ = self.agent.memory.retrieve(z_batch[i], k=5)
                if w is not None:
                    weights_list.append(w)
                    valid_indices.append(i)

            if not valid_indices: continue

            weights = torch.stack(weights_list)
            z_v = z_batch[valid_indices]
            a_v = a_batch[valid_indices]
            r_v = r_batch[valid_indices]
            z_next_actual_v = z_next_actual[valid_indices]

            z_next_pred = z_v @ self.agent.Psi + a_v @ self.agent.Gamma.t()
            R_phys = torch.nn.functional.mse_loss(z_next_pred, z_next_actual_v)

            log_weights = torch.log(weights + 1e-6).clamp(min=-5.0)
            J_perf = -torch.mean(r_batch * log_weights)

            loss = lambda_phys * R_phys + lambda_perf * J_perf

            self.optimizer.zero_grad()
            loss.backward()
            # NOTE: Diagnostic: Gradient Clipping
            torch.nn.utils.clip_grad_norm_(self.agent.parameters(), max_norm=1.0)
            self.optimizer.step()

            self.agent.enforce_contractive_dynamics()

            epoch_r_phys += R_phys.item()
            epoch_j_perf += J_perf.item()
            valid_updates += 1

            print(
                f"[TRAINING] Transition Loss: {loss.item():.4f} | " +
                f"R_phys: {R_phys.item():.4f} | J_perf: {J_perf.item():.4f}"
            )    
        print(f"[TRAINING EPOCH COMPLETE]")

        if valid_updates == 0:
            return 0.0, 0.0
        return epoch_r_phys / valid_updates, epoch_j_perf / valid_updates

    """Total, collision, warning are only here!!!"""
    def run(self, steps, episode_seed):   # NOTE: 200 for 0.5 dt is 10 seconds
        WARNING, COLLISION = 0, 0

        self.load_scenario(episode_seed)
        self.world.reset()
        self.experience_buffer = []

        ego_pos, _ = self.ego.get_world_pose()
        prev_dist = np.linalg.norm(self.ego_goal - ego_pos)

        index = None
        z_t = None

        success = 0
        total_step = 0

        for i in range(steps):
            self.manage_intruders(current_step=i)

            if len(self.active_scenario_intruders) > 0:
                intruder_loss = apply_multiagent_intruder_behavior(
                    self.intruder_controller,
                    self.ego,
                    self.active_scenario_intruders
                )

            # 1. Perception: Get the state at time t
            event_list, radii = self.detection()
            ego_pos, _ = self.ego.get_world_pose()

            dir_to_goal = self.ego_goal - ego_pos
            dir_to_goal /= (np.linalg.norm(dir_to_goal) + 1e-6)
            base_vel = torch.from_numpy(dir_to_goal).float().to(device) * 3.0

            if event_list is None:
                final_action = base_vel
                z_t = None
            else:
                total_step += 1
                # 2. Decision Making: Calculate Action at time t
                action, z_t, _, _, index = self.agent.select_action(event_list, k=5)
                final_action = base_vel + action

            self.ego_view.set_linear_velocities(
                final_action.detach().cpu().numpy().reshape(1, -1)
            )
            # 3. Physics: Step the environment to apply a_t
            self.world.step(render=False)

            # 4. Perception: Get the resulting state at time t+1
            event_next, radii = self.detection()

            # --- GOAL REACH CHECK ---
            dist_to_goal = np.linalg.norm(self.ego_goal - ego_pos)
            if dist_to_goal < 2.0: # 2 meter threshold
                print(f"        [GOAL REACHED] at step {i}!")
                success = 1
                # Give a massive reward for finishing
                reward = torch.tensor([10.0], device=device) 
                break

            # 5. Reward: Calculate r_t (NOTE: A simple heuristic for now)
            if event_next is None:
                reward = torch.tensor(
                    [0.5], dtype=torch.float32, device=device
                )  # Successfully cleared the threat
            else:
                # event_next: (N, 13)
                rel_positions = event_next[:, 1:4]  # (N, 3)
                dists = torch.norm(rel_positions, dim=1)

                surface_dists = dists - radii - 0.25
                min_dist = torch.min(surface_dists)

                progress = prev_dist - dist_to_goal
                prev_dist = dist_to_goal

                if min_dist < 0:
                    reward_val = -10.0
                    print("[COLLISION] Collision encountered!")
                    self.agent.memory.penalize_by_indices(index, factor=0.01)
                    COLLISION += 1
                elif min_dist < SAFETY_THRESHOLD:
                    reward_val = -1.0 * (SAFETY_THRESHOLD - min_dist) ** 2
                    print("[WARNING] Collision might happen!!!")
                    self.agent.memory.penalize_by_indices(index, factor=0.5)
                    WARNING += 1
                else:
                    reward_val = 0.1 * progress   # Add progress

                reward = torch.tensor(
                    [reward_val], device=device
                )

            # 6. Logging: Save the Status Code
            self.experience_buffer.append(
                (event_list, final_action.detach(), reward, event_next)
            )

            if z_t is not None:
                z_next = self.agent.encoder(
                    event_next
                ) if event_next is not None else None
                self.agent.memory.add_experiences(
                    z_t.detach(),
                    final_action.detach(),
                    reward,
                    z_next.detach() if z_next is not None else None
                )

            # print(f"[LOGGED S_t] Reward: {reward.item()}")

        intruder_loss = self.intruder_controller.update()

        if torch.is_tensor(intruder_loss):
            intruder_loss = intruder_loss.item()

        print(f"[INTRUDER TRAINING] Loss: {intruder_loss:.4f}")

        avg_r_phys, avg_j_perf = self.train_agent()

        status = "SUCCESS" if success else "TIMEOUT/CRASH"
        print(f"--- Episode Summary: {status} | \
            Final Dist: {dist_to_goal:.2f}m ---")
        print(f"[COMPLETE] Collected {total_step} experiences.")

        return success, COLLISION, WARNING, dist_to_goal, total_step, avg_r_phys, \
            avg_j_perf, intruder_loss
