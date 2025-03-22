<p align="center">
  <img src="res/ManeuverGPT-logo.png" alt="ManeuverGPT Logo" width="200">
</p>

<h1 align="center">ManeuverGPT</h1>

<p align="center">
  <a href="https://www.youtube.com/playlist?list=PLMcjQ-k9Bg8RPpmOUefSjn8F1C1TC2hTX" target="_blank">
    <img src="https://img.shields.io/badge/YouTube-FF0000?style=flat&logo=youtube&logoColor=white&labelColor=black&borderRadius=20" alt="YouTube Badge">
  </a>
</p>

<h3 align="center">Agentic Control for Safe Autonomous Stunt Maneuvers</h3>


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
python experiments/j_turn_eval.py --vehicle sedan --iterations 100
```

Additional parameters:

```sh
--vehicle [sedan|sports_coupe]   # Selects the vehicle model
--iterations N                   # Number of trials to run
--output results.json            # Saves experiment results to a JSON file
```

### Parameter Adjustment via Prompting

The agents iteratively refine maneuver parameters based on textual prompts:

```sh
python src/main.py --prompt "Execute a J-turn avoiding obstacles."
```

## Citation

If you use ManeuverGPT in your research, please cite:

```bibtex
@article{azdam2025maneuvergpt,
  title={ManeuverGPT Agentic Control for Safe Autonomous Stunt Maneuvers},
  author={Azdam, Shawn and Doma, Pranav and Arab, Aliasghar Moj},
  journal={arXiv preprint arXiv:2503.09035},
  year={2025}
}
```

## License

This project is licensed under the **CC BY 4.0**.

