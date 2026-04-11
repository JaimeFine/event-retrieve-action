# pyright: reportMissingImports=false
from omni.isaac.kit import SimulationApp

simulation_app = SimulationApp({"headless": True})

from macro import device
from trainer import Environment
import torch
import numpy as np

if __name__ == "__main__":
    import os

    sim = Environment(seed=42)
    sim.setup_environment()

    successes = []
    collisions = []

    # 1. Load Pretrained Knowledge
    if os.path.exists("agent_finetuned.pt"):
        print("Loading pretrained weights...")
        checkpoint = torch.load("agent_pretrained.pt", map_location=device)
        sim.agent.encoder.load_state_dict(checkpoint["encoder"])
        sim.agent.Psi.data.copy_(checkpoint["Psi"])
        sim.agent.Gamma.data.copy_(checkpoint["Gamma"])
        print("Pretrained model loaded successfully.")
    else:
        print(
            "WARNING: agent_pretrained.pt not found. \
                Starting with random initialization."
        )

    if os.path.exists("expert_dataset.pt"):
        print("Loading expert experiences into Knowledge Bank...")
        expert_data = torch.load("expert_dataset.pt", map_location=device)
        for event_list, action in expert_data:
            # Pass the event through the encoder to get the latent z_t
            z_t = sim.agent.encoder(event_list.to(device))
            # Add to memory so select_action has something to retrieve
            sim.agent.memory.add_experiences(z_t, action.to(device))
        sim.agent.memory.build_index()
        print(
            f"Knowledge Bank initialized with \
                {len(sim.agent.memory.maneuvers)} maneuvers."
        )

    # 2. Run Adversarial Training
    print("Starting Curriculum Training...")
    
    goal_rng = np.random.RandomState(42)

    num_episodes = 5
    for episode in range(num_episodes):
        TOTAL = 0
        WARNING = 0
        COLLISION = 0
        # 1. Sample random direction (uniform on sphere)
        direction = goal_rng.normal(size=3)
        direction[2] = abs(direction[2]) + 0.1  # Z-direction must be positive
        direction /= np.linalg.norm(direction) + 1e-8  # normalize

        # 2. Fix total distance to 100m
        distance = 100.0
        goal_offset = direction * distance

        # 3. Define goal relative to start
        sim.ego_goal = sim.ego_start + goal_offset

        print(f"\n=====================================================")
        print(f"--- EPISODE {episode + 1}/{num_episodes} ---")
        print(f"Start Pos: {sim.ego_start}")
        print(f"Goal Pos:  {sim.ego_goal}")
        print(f"Distance:  {np.linalg.norm(sim.ego_goal - sim.ego_start):.2f} m")
        print(f"=====================================================")

        s, c, w, d = sim.run(steps=1500, episode_seed=(100 + episode))

        TOTAL = 1500 # Or however many steps the episode actually took
        COLLISION = c
        WARNING = w

        successes.append(s)
        collisions.append(c > 0)
        print(f"Success Rate: {s:.3f}")
        print(f"Collision Rate: {c:.3f}")

        if TOTAL == 0:
            TOTAL = 1
            print(f"TOTAL: {TOTAL}")

        print(f"\n=====================================================")
        print(f"COLLISION: {COLLISION / TOTAL}")
        print(f"WARNING: {WARNING / TOTAL}")
        print(f"SUCCESS: {(TOTAL - WARNING - COLLISION) / TOTAL}")
        print(f"=====================================================")

    print(f"Success Rate: {np.mean(successes):.3f}")
    print(f"Collision Rate: {np.mean(collisions):.3f}")

    # 3. Save the Fine-Tuned Model
    print("Saving fine-tuned model...")
    torch.save({
        "encoder": sim.agent.encoder.state_dict(),
        "Psi": sim.agent.Psi.data,
        "Gamma": sim.agent.Gamma.data,
    }, "agent_finetuned.pt")

    print("Adversarial Training Complete.")

    simulation_app.close()