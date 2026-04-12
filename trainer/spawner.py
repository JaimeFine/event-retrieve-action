# pyright: reportMissingImports=false
import numpy as np
from challenger import CurriculumScheduler, AdversarialSpawner
from isaacsim.core.api.objects import DynamicSphere
from intruders import DroneIntruder, BirdIntruder, StaticObstacle
from omni.isaac.core import World
from isaacsim.core.prims import RigidPrim
from .train import Trainer
from macro import detection_threshold, ego

class Environment(Trainer):
    def __init__(self, seed):
        super().__init__()
        self.dt = 0.05
        self.world = World(stage_units_in_meters=1.0, physics_dt=self.dt)

        # Simulation State
        self.d_threshold = detection_threshold
        self.active_scenario_intruders = [] # Initialized for the sensors
        
        # For the training
        self.rng = np.random.RandomState(seed)
        self.scheduler = CurriculumScheduler(total_steps=500)
        self.spawner = AdversarialSpawner(self.rng)

        self.num_intruders = 25

    def setup_environment(self):
        self.world.scene.add_default_ground_plane()

        # Spawn Ego Drone
        self.ego = DynamicSphere(
            prim_path="/World/ego_drone",
            name="ego_drone",
            position=np.array([0.0, 0.0, 1.5]),
            radius=0.25,
            color=ego
        )
        self.world.scene.add(self.ego)

        # PRE-SPAWN a pool of intruders to avoid physics crashes later
        self.intruders = []
        # We spawn a mix of types in a "hidden" location far underground
        hidden_pos = [0.0, 0.0, -100.0]

        for i in range(self.num_intruders):
            self.intruders.append(DroneIntruder(f"pool_drone_{i}", hidden_pos))
            self.intruders.append(BirdIntruder(f"pool_bird_{i}", hidden_pos))
            self.intruders.append(StaticObstacle(f"pool_static_{i}", hidden_pos))

        for intruder in self.intruders:
            self.world.scene.add(intruder.prim)

        # Create GPU-backed view for the ego drone
        self.ego_view = RigidPrim(
            prim_paths_expr="/World/ego_drone",
            name="ego_view",
            track_contact_forces=False
        )
        self.world.scene.add(self.ego_view)

    def load_scenario(self, seed):
        self.rng = np.random.RandomState(seed)
        self.active_scenario_intruders = []
        self.ego.set_world_pose(position=self.ego_start)

        # Teleport everyone underground
        for intruder in self.intruders:
            intruder.set_state(np.array([0.0, 0.0, -100.0]), np.zeros(3))

        self.ego.set_world_pose(position=self.ego_start)
        self.ego.set_linear_velocity(np.zeros(3))

    def manage_intruders(self, current_step):
        """
        Dynamically maintains active adversarial intruders based on curriculum.
        """
        difficulty = self.scheduler.get_difficulty(current_step)
        ego_pos, _ = self.ego.get_world_pose()
        ego_vel = self.ego.get_linear_velocity()

        # Keep spawning until we have 5 active threats
        while len(self.active_scenario_intruders) < self.num_intruders:
            # Generate rigorous adversarial cases (collision, crossing, etc.)
            cases = self.spawner.spawn_event(ego_pos, ego_vel, difficulty)

            for spawn_pos, velocity in cases:
                available = [
                    inst for inst in self.intruders
                    if inst not in self.active_scenario_intruders
                ]
                if not available:
                    break # Pool is empty, wait for next step

                new_intruder = available[0]
                new_intruder.set_state(spawn_pos, velocity)
                self.active_scenario_intruders.append(new_intruder)
