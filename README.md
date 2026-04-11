# Event-Centric Retrieval-Based Action Framework (ERA)
### Real-Time Drone Navigation in Isaac Sim with Latent Manifold Stability

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Isaac Sim](https://img.shields.io/badge/Simulation-NVIDIA%20Isaac%20Sim-76b900.svg)](https://developer.nvidia.com/isaac-sim)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

This project is still updating, the newest preprint is on arXiv: [arXiv-preprint-era](https://arxiv.org/abs/2604.07392)

If you find this work useful, please cite:
```bibtex
@misc{zhaowen2026eventcentricworldmodelingmemoryaugmented,
      title={Event-Centric World Modeling with Memory-Augmented Retrieval for Embodied Decision-Making}, 
      author={Fan Zhaowen},
      year={2026},
      eprint={2604.07392},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2604.07392}, 
}
```

## Overview
This repository implements an **Event-Centric Architecture** for autonomous drone navigation, as described in the accompanying research paper. Unlike traditional end-to-end RL, this framework maps environmental "events" (intruder proximity, goal vectors) into a stable **Latent Manifold** to retrieve mathematically verified maneuvers from a curated **Knowledge Bank**.

### Key Features:
* **Permutation-Invariant Encoder:** Processes a variable number of intruders using a batched `EventEncoder`.
* **Stability-Guaranteed Dynamics:** Implements **Contractive Dynamics** via SVD-clamping of the transition matrix ($\Psi$).
* **Bayesian Knowledge Retrieval:** Replaces "averaging" with a probabilistic selection to avoid the *Average-to-Collision* failure in "Fork" scenarios.
* **Hardware-Ready:** Optimized for deployment on **NVIDIA Jetson Orin Nano** via TensorRT and ONNX.

### Structure:

```txt
event-retrieve-action/
├── main.py
├── macro.py
├── agents/
│   ├── __init__.py
│   ├── agent.py
│   ├── bank.py
│   ├── encoder.py
│   └── stabilizer.py
├── challenger/
│   ├── __init__.py
│   ├── core.py
│   ├── situation.py
│   └── velocity.py
├── intruders/
│   ├── __init__.py
│   ├── base.py
│   ├── bird.py
│   ├── drone.py
│   └── static.py
└── trainer/
    ├── __init__.py
    ├── spawner.py
    └── train.py
```

> **Notes:**
> 
> - `main.py` and `pretrain.py` are the primary entry points for experiments and model pretraining.
> 
> - `event-retrieve-action/` contains a focused retrieval/manifold project (data, models, and tooling).
> 
> - Use this mindmap as a quick overview; open any folder for detailed file lists and docs.

---

## Getting Started

### Prerequisites
* **Ubuntu 22.04**
* **NVIDIA Isaac Sim (2023.x or later)**
* **RTX 40-series GPU** (recommended for training) or **Jetson Orin** (for edge inference)

### Installation
1. Clone the repository:
   ```bash
   git clone [https://github.com/JaimeFine/event-retrieve-action.git](https://github.com/JaimeFine/event-retrieve-action.git)
   cd event-retrieve-action
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## Technical Architecture
The framework operates in three distinct phases:
1. Event Encoding: Raw sensor data is compressed into a 128-dimensional latent vector ($z$).
2. Manifold Mapping: The $d_{phys}$ regularizer ensures that physical distance in the simulator correlates to Euclidean distance in the latent space.
3. Action Retrieval: The system queries a FAISS-indexed Knowledge Bank of ~27k expert maneuvers.

---

## Results & Benchmarks
The following results were obtained on an NVIDIA RTX 4090D:
- Pretraining Convergence: Loss reduced from $1.97 \to 0.52$ over 10 epochs.
- Inference Latency: $< 2ms$ per step (achieving real-time parity for 400Hz control loops).
- Cross-Platform Parity: MSE between 4090D and Jetson Orin Nano remains $< 10^{-6}$.

---

## License
This project is licensed under the **GNU General Public License v3.0 (GPL-3.0)**. See the [LICENSE](LICENSE) file for details.



