# Isaac Sim Headless Template: Intruder & Environment

# pyright: reportMissingImports=false
# pyright: reportMissingModuleSource=false
import numpy as np
import torch
from omni.isaac.kit import SimulationApp

# 1. Start the simulation headless
simulation_app = SimulationApp({"headless": True})

from omni.isaac.core import World   # NOTE: World is the library
from omni.isaac.core.objects import DynamicSphere


class BaseIntruder:
    def __init__(self, name, position, color, radius=0.3):
        self.name = name
        self.initial_position = np.array(position)
        self.radius = radius

        self.prim = DynamicSphere(
            prim_path=f"/World/{name}",
            name=name,
            position=self.initial_position,
            radius=self.radius,
            color=np.array(color)
        )

    def apply_behavior(self):
        """To be overridden by child classes"""
        pass

    def set_state(self, position, velocity):
        """Teleports the intruder and applies a new velocity vector."""
        self.prim.set_world_pose(position=position)
        self.prim.set_linear_velocity(velocity)

    def get_state(self):
        pos, _ = self.prim.get_world_pose()
        vel = self.prim.get_linear_velocity()

        return pos, vel
    
    def reset(self):
        """
        For seeded scenarios: Zeroes out physics forces and returns the
        intruder to its exact spawn point.
        """
        self.prim.set_world_pose(position=self.initial_position)
        self.prim.set_linear_velocity(np.zeros(3))
        self.prim.set_angular_velocity(np.zeros(3))
    
class DroneIntruder(BaseIntruder):  # BLUE
    def __init__(self, name, position, velocity=None):
        super().__init__(name, position, color=np.array([1.0, 0.0, 0.0]))
        self.velocity = velocity if velocity is not None \
            else np.array([-3.0, 0.0, 0.0])

    def apply_behavior(self):
        self.prim.set_linear_velocity(self.velocity)

class BirdIntruder(BaseIntruder):   # GREEN
    def __init__(
        self, name, position, base_velocity=None,
        frequency=2.0, amplitude=2.0
    ):
        super().__init__(
            name, position, color=np.array([0.0, 1.0, 0.0]), radius=0.15
        )
        self.base_velocity = base_velocity if base_velocity is not None \
            else np.array([-2.0, 0.0, 0.0])
        self.frequency = frequency
        self.amplitude = amplitude

        self.step_count = 0
        self.dt = 0.05

    def apply_behavior(self):
        # Erratic, sinusoidal flight pattern
        self.step_count += 1
        
        t = self.step_count * self.dt
        y_vel = np.sin(t * self.frequency) * self.amplitude

        current_vel = np.copy(self.base_velocity)
        current_vel[1] = y_vel

        self.prim.set_linear_velocity(current_vel)

    def reset(self):
        super().reset()

        self.step_count = 0

class StaticObstacle(BaseIntruder):
    def __init__(self, name, position):
        super().__init__(name, position, color=np.array([0.5, 0.5, 0.5]))

    def apply_behavior(self):
        # Environment constraints (buildings, poles) do not move
        self.prim.set_linear_velocity(np.zeros(3))

class ScenarioGenerator:
    def __init__(self, num_intruders=5, spawn_radius=10.0):
        self.num_intruders = num_intruders
        self.spawn_radius = spawn_radius

    def get_scenario(
            self, seed, ego_start=np.array([0.0, 0.0, 1.5]),
            ego_goal=np.array([50.0, 20.0, 5.0])
    ):
        """
        Takes an integer seed and returns a dictionary of
        configurations. Using RandomState ensure we don't mess up
        the global numpy seed.
        """
        rng = np.random.RandomState(seed)
        configs = []

        path_vector = ego_goal - ego_start

        for i in range(self.num_intruders):
            # 1. Randomize Intruder Type
            intruder_type = rng.choice(["Drone", "Bird", "Static"])

            # 2. Randomize Spawn Position (Spawn in front of the ego drone)
            # Pick a point between 10% and 85% of the way to the goal
            t = rng.uniform(0.1, 0.85)
            base_point = ego_start + t * path_vector

            # Adding small scatter:
            x_pos = base_point[0] + rng.uniform(-2.0, 2.0)
            y_pos = base_point[1] + rng.uniform(-3.0, 3.0)
            # Don't spawn underground
            z_pos = np.clip(base_point[2] + rng.uniform(-1.5, 1.5), 0.5, 10.0)
            position = np.array([x_pos, y_pos, z_pos])

            # 3. Randomize Behavior Parameters
            if intruder_type == "Drone":
                vx = rng.uniform(-4.0, -1.0)
                vy = rng.uniform(-2.0, 0.0) if path_vector[1] > 0 \
                    else rng.uniform(0.0, 2.0)
                params = {"velocity": np.array([vx, vy, 0.0])}

            elif intruder_type == "Bird":
                vx = rng.uniform(-2.5, -0.5)
                freq = rng.uniform(1.0, 4.0)
                amp = rng.uniform(0.5, 2.5)
                params = {
                    "base_velocity": np.array([vx, 0.0, 0.0]),
                    "frequency": freq,
                    "amplitude": amp
                }

            else:
                params = {}

            configs.append({
                "name": f"Intruder_{i}",
                "type": intruder_type,
                "position": position,
                "params": params
            })

        return configs

def spawn(ego_pos, dir_to_goal, spawn_dist):
    spawn_pos = ego_pos + dir_to_goal * spawn_dist
    # Add lateral noise so it is not the perfect line
    # Pick a vector not parallel to dir_to_goal
    if abs(dir_to_goal[0]) < 0.9:
        arbitrary = np.array([1.0, 0.0, 0.0])
    else:
        arbitrary = np.array([0.0, 1.0, 0.0])

    # Build perpendicular basis
    v1 = np.cross(dir_to_goal, arbitrary)
    v1 /= np.linalg.norm(v1)

    v2 = np.cross(dir_to_goal, v1)
    v2 /= np.linalg.norm(v2)

    # Sample noise in a disk
    r = np.random.uniform(0, 4.0)  # radius control
    theta = np.random.uniform(0, 2*np.pi)

    lateral_noise = r * (np.cos(theta) * v1 + np.sin(theta) * v2)

    # Apply noise
    spawn_pos += lateral_noise

    return spawn_pos


class Environment:
    def __init__(self):
        self.dt = 0.05
        self.world = World(stage_units_in_meters=1.0, physics_dt=self.dt)

        # Formalized Navigation Task
        self.ego_start = np.array([0.0, 0.0, 1.5])
        self.ego_goal = np.array([50.0, 20.0, 5.0])

        # Simulation State
        self.d_threshold = 5.0
        self.active_scenario_intruders = [] # Initialized for the sensors

    def setup_environment(self):
        self.world.scene.add_default_ground_plane()

        # Spawn Ego Drone
        self.ego = DynamicSphere(
            prim_path="/World/ego_drone",
            name="ego_drone",
            position=self.ego_start,
            radius=0.25,
            color=np.array([0.0, 0.0, 1.0]) # Blue
        )
        self.world.scene.add(self.ego)

        # PRE-SPAWN a pool of intruders to avoid physics crashes later
        self.intruders = []
        # We spawn a mix of types in a "hidden" location far underground
        hidden_pos = [0.0, 0.0, -100.0]

        for i in range(10):
            self.intruders.append(DroneIntruder(f"pool_drone_{i}", hidden_pos))
            self.intruders.append(BirdIntruder(f"pool_bird_{i}", hidden_pos))
            self.intruders.append(StaticObstacle(f"pool_static_{i}", hidden_pos))

        for intruder in self.intruders:
            self.world.scene.add(intruder.prim)

    def load_scenario(self, seed):
        self.rng = np.random.RandomState(seed)
        self.active_scenario_intruders = []

        # Teleport everyone underground
        for intruder in self.intruders:
            intruder.set_state(np.array([0.0, 0.0, -100.0]), np.zeros(3))

        self.ego.set_world_pose(position=self.ego_start)
        self.ego.set_linear_velocity(np.zeros(3))

    def run_data_collection(self, steps=600):
        """Runs the VPF Teacher to generate the expert dataset"""
        self.world.reset()
        dataset = []

        for i in range(steps):
            ego_pos, _ = self.ego.get_world_pose()

            dist_to_goal = np.linalg.norm(self.ego_goal - ego_pos)
            if i % 100 == 0: # Prints every 100 steps to prevent console lag
                print(f"    [Step {i}] Dist to Goal: {dist_to_goal:.2f}m | Active Intr.: {len(self.active_scenario_intruders)}")

            # 1. Dynamic Spawning
            # Maintaining exactly 5 intruders in front of the ego
            while len(self.active_scenario_intruders) < 5:
                # Spawn 7-12 meters ahead of the current ego position
                spawn_dist = np.random.uniform(7.0, 12.0)

                # Direction
                dir_to_goal = (self.ego_goal - ego_pos)
                dir_to_goal /= np.linalg.norm(dir_to_goal)

                # Calculate the spawning point
                spawn_pos = spawn(ego_pos, dir_to_goal, spawn_dist)

                # Pick a random intruder from the pool that isn't active
                available = [
                    inst for inst in self.intruders \
                        if inst not in self.active_scenario_intruders 
                ]
                if not available: break

                new_intruder = available[0]

                # Set its velocity to intercept the ego's current location
                # We assume it takes ~1-3 seconds to meet
                t_meet = np.random.uniform(1, 3)

                vec_to_ego = ego_pos - spawn_pos
                dist = np.linalg.norm(vec_to_ego)

                v_dir = vec_to_ego / dist
                v_mag = dist / t_meet

                v_mag = np.clip(v_mag, 2.0, 5.0)

                new_intruder.set_state(spawn_pos, v_dir * v_mag)
                self.active_scenario_intruders.append(new_intruder)

            # --- Physics ---
            for intruder in self.active_scenario_intruders:
                intruder.apply_behavior()

            event_list = self.detection()
            act_np = self.get_expert_action()

            if event_list is not None:
                dataset.append((
                    event_list, torch.tensor(act_np, dtype=torch.float32)
                ))

            self.ego.set_linear_velocity(act_np)
            self.world.step(render=False)

            # --- Cleanup ---
            remaining = []
            for inst in self.active_scenario_intruders:
                pos, _ = inst.get_state()
                # Relative X check
                if (pos[0] - ego_pos[0]) < -3.0 or np.linalg.norm(pos - ego_pos) > 25.0:
                    inst.set_state(np.array([0, 0, -100.0]), np.zeros(3))
                else:
                    remaining.append(inst)
            self.active_scenario_intruders = remaining

            # Collision Check
            for intruder in self.active_scenario_intruders:
                int_pos, _ = intruder.get_state()
                min_d = intruder.radius + 0.25
                if np.linalg.norm(ego_pos - int_pos) < min_d:
                    print(f"    [!] Collision at step {i}. Discarding episode.")
                    return []   # Return empty so we don't learn from crashes
                
            if event_list is not None:
                print(f"    [Y] Yes events at step {i}.")

            if np.linalg.norm(self.ego_goal - ego_pos) < 1.0:
                print(f"Goal reached at step {i}")
                break

        return dataset

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
                # ego_x, ego_y, ego_z, ego_vx, ego_vy, ego_vz,
                # goal_x, goal_y, goal_z
                # ]
                rel_goal = (self.ego_goal - ego_pos)
                s_global = ego_vel.tolist() + rel_goal.tolist()
                event = [type_id] + rel_pos.tolist() + \
                    rel_vel.tolist() + s_global
                event_list.append(event)

        return torch.tensor(
            event_list, dtype=torch.float32
        ) if event_list else None

    def get_expert_action(self):
        """
        We apply a virtual potential field (VPF) teacher.
        calculate the mathematically safe vector to the goal while avoiding
        obstacles.
        """
        ego_pos, _ = self.ego.get_world_pose()

        # 1. Attractive Force to the Goal
        k_attr = 3.0    # Speed multiplier
        max_speed = 5.0

        dir_to_goal = self.ego_goal - ego_pos
        dist_to_goal = np.linalg.norm(dir_to_goal)

        if dist_to_goal > 0.5:
            f_attr = (dir_to_goal / dist_to_goal) * k_attr
        else:
            f_attr = np.zeros(3)    # Stop if reached the goal

        # 2. Repulsive Force away from the Intruders
        k_rep = 15.0
        f_rep = np.zeros(3)
        
        for intruder in self.active_scenario_intruders:
            pos, _ = intruder.get_state()
            vec_from_intruder = ego_pos - pos
            dist = np.linalg.norm(vec_from_intruder)

            if dist < self.d_threshold:
                # Closer = exponentially stronger push
                repulsion_mag = k_rep / (dist**2 + 1e-6)
                f_rep += (vec_from_intruder / dist) * repulsion_mag

        # 3. Final Action Computation
        action = f_attr + f_rep

        # 4. Tangential Velocity (The "Squeeze" Logic)
        # If the drone is stuck (f_attr and f_rep fighting), add a small perpendicular force
        # to encourage it to go AROUND the obstacle instead of backing up.
        if np.linalg.norm(action) < 1.0 and dist_to_goal > 2.0:
            perpendicular_dodge = np.array([-dir_to_goal[1], dir_to_goal[0], 0])
            action += (perpendicular_dodge / np.linalg.norm(perpendicular_dodge)) * 2.0

        # Kinematic Constraint: Clamp to the drone's max physical speed
        speed = np.linalg.norm(action)
        if speed > max_speed:
            action = (action / speed) * max_speed

        return action

if __name__ == "__main__":
    sim = Environment()
    sim.setup_environment()

    master_dataset = []

    # --- METRIC COUNTERS ---
    total_episodes = 50
    successful_episodes = 0
    reach_threshold = 2.0  # meters

    print("--- STARTING EXPERT DATA COLLECTION ---")

    for seed in range(total_episodes):
        sim.load_scenario(seed=seed)
        
        # Increase steps to 1000 to give the expert more time to navigate
        episode_data = sim.run_data_collection(steps=1000)
        
        # Check final state of the ego drone
        final_pos, _ = sim.ego.get_world_pose()
        dist_to_goal = np.linalg.norm(sim.ego_goal - final_pos)
        
        if dist_to_goal < reach_threshold:
            # Only save data from successful arrivals to ensure high-quality pretraining
            master_dataset.extend(episode_data)
            successful_episodes += 1
            print(f"  [SUCCESS] Seed {seed}: Reached goal ({dist_to_goal:.2f}m). Data saved.")
        else:
            print(f"  [TIMEOUT] Seed {seed}: Failed to reach goal ({dist_to_goal:.2f}m). Data discarded.")

    # --- CALCULATE AND PRINT RATE ---
    reach_rate = (successful_episodes / total_episodes) * 100
    print("\n" + "="*30)
    print(f"COLLECTION COMPLETE")
    print(f"Goal Reach Rate: {reach_rate:.2f}% ({successful_episodes}/{total_episodes})")
    print(f"Total Expert Pairs: {len(master_dataset)}")
    print("="*30)

    if len(master_dataset) > 0:
        torch.save(master_dataset, "expert_dataset.pt")
        print("[SAVED] Dataset saved to expert_dataset.pt")
    else:
        print("[WARNING] No data collected. Adjust VPF gains or increase steps.")

# OUTPUT:
# # ==============================
# COLLECTION COMPLETE
# Goal Reach Rate: 100.00% (50/50)
# Total Expert Pairs: 27075
# ==============================
# [SAVED] Dataset saved to expert_dataset.pt
