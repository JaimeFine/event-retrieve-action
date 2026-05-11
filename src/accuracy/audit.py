# pyright: reportMissingImports=false
from omni.isaac.kit import SimulationApp
simulation_app = SimulationApp({"headless": True})

import torch
import numpy as np
import random

from macro import device, seeds
from trainer import Environment

def set_seed(seed=999):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def get_expert_action(sim_env, ego_pos, event_list):
    k_attr = 3.0
    max_speed = 5.0

    dir_to_goal = sim_env.ego_goal - ego_pos
    dist_to_goal = np.linalg.norm(dir_to_goal)

    if dist_to_goal > 0.5:
        f_attr = (dir_to_goal / dist_to_goal) * k_attr
    else:
        f_attr = np.zeros(3)

    k_rep = 15.0
    f_rep = np.zeros(3)

    # event_list[:,1:4] = relative positions
    rel_positions = event_list[:, 1:4].cpu().numpy()

    for rel_pos in rel_positions:
        vec_from_intruder = -rel_pos
        dist = np.linalg.norm(vec_from_intruder)

        if dist < sim_env.d_threshold and dist > 1e-6:
            repulsion_mag = k_rep / (dist**2 + 1e-6)
            f_rep += (vec_from_intruder / dist) * repulsion_mag

    action = f_attr + f_rep

    if np.linalg.norm(action) < 1.0 and dist_to_goal > 2.0:
        dodge = np.array([-dir_to_goal[1], dir_to_goal[0], 0.0])

        if np.linalg.norm(dodge) > 1e-6:
            dodge /= np.linalg.norm(dodge)
            action += dodge * 2.0

    speed = np.linalg.norm(action)

    if speed > max_speed:
        action = action / speed * max_speed

    return torch.tensor(
        action,
        dtype=torch.float32,
        device=device
    )

if __name__ == "__main__":
    set_seed(999)
    print("[AUDIT] Initializing Simulation...")
    sim = Environment(seed=999) # Novel seed for testing
    sim.setup_environment()

    # 1. Loading the fine-tuned model
    checkpoint = torch.load("agent_finetuned.pt", map_location=device)
    sim.agent.encoder.load_state_dict(checkpoint["encoder"])
    sim.agent.Psi.data.copy_(checkpoint["Psi"])
    sim.agent.Gamma.data.copy_(checkpoint["Gamma"])

    sim.agent.eval()
    for p in sim.agent.parameters():
        p.requires_grad = False

    # Load the knowledge bank
    kb_data = torch.load("knowledge_bank_snapshot.pt", map_location=device)
    sim.agent.memory.latents = kb_data["latents"].to(device)
    sim.agent.memory.actions = kb_data["actions"].to(device)
    sim.agent.memory.reliability = kb_data["reliability"].to(device)
    print(
        f"[AUDIT] Model and KB ({len(sim.agent.memory.actions)} entries) loaded."
    )

    # 2. Setup the Audit Episode
    sim.world.reset()
    goal_offset = np.array([100.0, 0.0, 10.0])
    sim.ego_goal = sim.ego_start + goal_offset

    audit_data = {
        "z_t": [],
        "a_t": [],
        "a_exp": [],
        "z_next": [],
        "weights": [],
        "min_dist": [],
        "goal_dist": [],
        "energy": []
    }

    print("[AUDIT] Running Trajectory...")
    for step in range(1500):

        sim.manage_intruders(current_step=step)

        event_list, radii = sim.detection()
        ego_pos, _ = sim.ego.get_world_pose()

        # -------------------------------------------------
        # Base velocity toward goal (same as training)
        # -------------------------------------------------
        dir_to_goal = sim.ego_goal - ego_pos
        goal_dist = np.linalg.norm(dir_to_goal)

        dir_to_goal /= (goal_dist + 1e-6)

        base_vel = torch.tensor(
            dir_to_goal,
            dtype=torch.float32,
            device=device
        ) * 3.0

        # -------------------------------------------------
        # Decision
        # -------------------------------------------------
        if event_list is None:

            final_action = base_vel
            z_t = None
            action = torch.zeros(3, device=device)
            a_exp = torch.zeros(3, device=device)
            weights_np = np.zeros(5)

        else:
            with torch.no_grad():
                action, z_t, _, valid_weights, index = sim.agent.select_action(
                    event_list, k=5
                )

            final_action = base_vel + action

            a_exp = get_expert_action(sim, ego_pos, event_list)

            if valid_weights is not None:
                weights_np = valid_weights.cpu().numpy()
            else:
                weights_np = np.zeros(5)

        # -------------------------------------------------
        # Safety Margin
        # -------------------------------------------------
        if event_list is not None:

            rel_positions = event_list[:,1:4]
            dists = torch.norm(rel_positions, dim=1)

            if radii is not None:
                surface = dists - radii - 0.25
                min_dist = torch.min(surface).item()
            else:
                min_dist = torch.min(dists).item()

        else:
            min_dist = sim.d_threshold

        # -------------------------------------------------
        # Execute Motion
        # -------------------------------------------------
        sim.ego_view.set_linear_velocities(
            final_action.detach().cpu().numpy().reshape(1,-1)
        )

        sim.world.step(render=False)

        # -------------------------------------------------
        # Next state
        # -------------------------------------------------
        event_next, _ = sim.detection()

        if event_next is not None and z_t is not None:
            with torch.no_grad():
                z_next = sim.agent.encoder(event_next)

            audit_data["z_t"].append(z_t.cpu().numpy())
            audit_data["a_t"].append(action.cpu().numpy())
            audit_data["a_exp"].append(a_exp.cpu().numpy())
            audit_data["z_next"].append(z_next.cpu().numpy())

        audit_data["goal_dist"].append(goal_dist)
        audit_data["min_dist"].append(min_dist)
        audit_data["weights"].append(weights_np)

        if z_t is not None:
            audit_data["energy"].append(torch.norm(z_t).pow(2).item())
        else:
            audit_data["energy"].append(0.0)

        # Goal reached
        if goal_dist < 2.0:
            print(f"[AUDIT] Goal reached at step {step}")
            break
        
    torch.save(audit_data, "audit_tensors.pt")
    print(
        f"[AUDIT] Complete! Logged {len(audit_data['z_t'])} interaction frames "
        f"to audit_tensors.pt"
    )
    simulation_app.close()