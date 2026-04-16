# pyright: reportMissingImports=false
from omni.isaac.kit import SimulationApp
simulation_app = SimulationApp({"headless": True})

from macro import device, seeds, total_epochs
from trainer import Environment
import torch
import numpy as np
import os
import json

def set_deterministic_seeds(seed_value):
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)

if __name__ == "__main__":
    set_deterministic_seeds(seeds)
    sim = Environment(seed=seeds)
    sim.setup_environment()
    
    training_logs = {
        "episodes": [],
        "success_rates": [],
        "collision_rates": [],
        "warning_rates": [],
        "phys_losses": [],     # R_phys tracking
        "perf_losses": [],     # J_perf tracking
        "intruder_losses": [],
        "knowledge_bank_size": []
    }
    
    # 1. Load Pretrained Knowledge
    if os.path.exists("agent_pretrained.pt"):
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
        print(
            f"Knowledge Bank initialized with \
                {len(sim.agent.memory.actions)} maneuvers."
        )

    # 2. Run Adversarial Training
    print(f"Starting Curriculum Training for {total_epochs} episodes...")
    goal_rng = np.random.RandomState(seeds)

    num_episodes = total_epochs
    for episode in range(num_episodes):
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

        s, c, w, d, t, r_phys, j_perf, i_loss = sim.run(
            steps=1500, episode_seed=(seeds + episode)
        )
        safe_t = max(1, t)
        success_rate = (t - w - c) / safe_t
        col_rate = c / safe_t
        warn_rate = w / safe_t

        print(f"\n=====================================================")
        print(f"COLLISION: {col_rate:.2f} | WARNING: {warn_rate:.2f} | SUCCESS: {success_rate:.2f}")
        print(f"=====================================================")

        training_logs["episodes"].append(episode)
        training_logs["success_rates"].append(float(success_rate))
        training_logs["collision_rates"].append(float(col_rate))
        training_logs["warning_rates"].append(float(warn_rate))
        training_logs["phys_losses"].append(float(r_phys))
        training_logs["perf_losses"].append(float(j_perf))
        training_logs["intruder_losses"].append(float(i_loss))
        training_logs["knowledge_bank_size"].append(int(len(sim.agent.memory.actions)))

    # 3. Save the Fine-Tuned Model
    print("Saving fine-tuned model...")
    torch.save({
        "encoder": sim.agent.encoder.state_dict(),
        "Psi": sim.agent.Psi.data,
        "Gamma": sim.agent.Gamma.data,
    }, "agent_finetuned.pt")

    # 4. Save the Knowledge Bank Data
    torch.save({
        "latents": sim.agent.memory.latents.cpu(),
        "actions": sim.agent.memory.actions.cpu(),
        "reliability": sim.agent.memory.reliability.cpu()
    }, "knowledge_bank_snapshot.pt")

    # 5. Save the Training Logs
    with open("training_results.json", "w") as f:
        json.dump(training_logs, f, indent=4)

    print("Adversarial Training Complete.")
    simulation_app.close()