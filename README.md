<p align="center">
  <img src="res/ManeuverGPT-logo.png" alt="ManeuverGPT Logo" width="144">
</p>

<h1 align="center">ManeuverGPT</h1>

<p align="center">
  <a href="https://www.youtube.com/playlist?list=PLMcjQ-k9Bg8RPpmOUefSjn8F1C1TC2hTX" target="_blank">
    <img src="https://img.shields.io/badge/YouTube-FF0000?style=flat&logo=youtube&logoColor=white&labelColor=black&borderRadius=20" alt="YouTube Badge">
  </a>
  
  <a href="https://arxiv.org/abs/2503.09035" target="_blank">
    <img src="https://img.shields.io/badge/arXiv-2503.09035-b31b1b.svg" alt="YouTube Badge">
  </a>
</p>

<h3 align="center">Agentic Control for Safe Autonomous Stunt Maneuvers</h3>

<p align="center">
  <img src="res/fig_agentic.png" alt="ManeuverGPT agentic control diagram" width="480" style="margin:8px;">
  <img src="res/fig_maneuver.png" alt="ManeuverGPT maneuver overview" width="480" style="margin:8px;">

  <br>
  <em>Agentic Control Diagram &nbsp;&nbsp;|&nbsp;&nbsp; Maneuver Phases Overview</em>
</p>


📣 **Announcement**: 

###  Paper Accepted to IROS 2025! 🎉

We are excited to announce that our paper has been **accepted for publication at the IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS) 2025**, and it has been selected for an **oral presentation**! We appreciate your interest in our work!

---

## Overview

**ManeuverGPT** is an Agentic framework for generating and executing
high-dynamic stunt maneuvers in autonomous vehicles using Large Language
Model (LLM)-based agents as controllers.
This repository provides the implementation of
ManeuverGPT, including its multi-agent architecture, control pipeline, and
experimental evaluations in the **CARLA** simulator.

## Key Features

- **Agentic Architecture:** Comprises three specialized LLM-driven agents:
    - **Query Enricher Agent**: Contextualizes user commands for maneuver
      generation.
    - **Driver Agent**: Generates maneuver parameters based on enriched
      queries.
    - **Parameter Validator Agent**: Enforces physics-based and safety
      constraints.
- **High-Dynamic Maneuver Execution:** Enables vehicles to perform complex
  stunt maneuvers such as **J-turns** with textual prompt-based control.
- **Simulation-Based Evaluation:** Tested in **CARLA v0.9.14** to ensure
  maneuver feasibility across different vehicle models.
- **Adaptive Prompting Mechanism:** Allows maneuver refinement without
  requiring retraining of model weights.
- **Multi-Agent Collaboration:** Improves execution success and precision
  compared to single-agent approaches.

## Installation

### Prerequisites

- **Python 3.10+**
- **CARLA Simulator v0.9.14**
- **Chat Completion-compatible LLM API** (e.g., GPT-4o, etc.)

### Setup

Clone the repository and install dependencies:

```sh
git clone https://github.com/SHi-ON/ManeuverGPT.git
cd ManeuverGPT
uv sync
```

Ensure **CARLA** is installed and running before executing the scripts.

## Running Experiments

### J-Turn Execution

To execute a J-turn maneuver in the CARLA simulation environment:

```sh
python src/maneuvergpt/carla/drive.py --iterations 100
```

For additional parameters, refer to the help documentation:

```sh
python src/maneuvergpt/carla/drive.py --help
```

## Citation

If you use ManeuverGPT in your research, please cite:

```bibtex
@article{Azdam_ManeuverGPT_Agentic_Control_2025,
  author = {Azdam, Shawn and Doma, Pranav and Arab, Aliasghar Moj},
  journal = {arXiv preprint arXiv:2503.09035},
  title = {{ManeuverGPT Agentic Control for Safe Autonomous Stunt Maneuvers}},
  url = {https://arxiv.org/abs/2503.09035},
  year = {2025}
}
```

## License

This project is licensed under the **CC BY 4.0**.

