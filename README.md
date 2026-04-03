# AmortizedMPs

This repository contains the simulation and training code for the AmortizedMPs control framework introduced in:

**Hierarchical Amortized Motor Control with Intrinsic Dynamics for Biomimetic Legged Locomotion**

The project builds on Isaac Gym for large-scale GPU simulation, `legged_gym` for locomotion environments, and `rsl_rl` for PPO-based policy optimization.

> **Status**: Manuscript under review. Code will be released upon publication. It is currently included as supplementary material in the submission.

## Overview

The repository includes:

- motor primitive training environments for Unitree A1 and Go2
- hierarchical AmortizedMPs training environments for Unitree A1 and Go2
- modified `rsl_rl` components used by the hierarchical training pipeline
- reference gait data for motor primitive modeling
- bundled pretrained checkpoints for A1

## Repository Layout

- `legged_gym/legged_gym/envs/a1/`: A1 environments and intrinsic dynamics modules
- `legged_gym/legged_gym/envs/go2/`: Go2 environments and intrinsic dynamics modules
- `legged_gym/data/`: reference gait trajectories used for motor primitive modeling
- `legged_gym/logs/`: pretrained checkpoints included with this repository
- `rsl_rl/`: local PPO training code, including the `_hdrl` variants used for hierarchical optimization

Registered tasks are defined in `legged_gym/legged_gym/envs/__init__.py`:

- `MP_a1`
- `Amortized_a1`
- `MP_go2`
- `Amortized_go2`

## Requirements

Training and simulation rely on NVIDIA Isaac Gym Preview 4 and a CUDA-capable GPU.

Tested setup:

- Ubuntu 20.04
- Python 3.8
- CUDA 12.1
- PyTorch 2.4.1
- NVIDIA RTX 4090

Isaac Gym is officially distributed for Ubuntu 18.04 and 20.04. If you use a different OS or driver stack, expect to make environment-specific fixes yourself.

## Installation

1. Create and activate a Python 3.8 environment:

```bash
conda create -n amortized_mps python=3.8
conda activate amortized_mps
```

2. Install the repository requirements:

```bash
pip install -r requirements.txt
```

3. Install PyTorch with CUDA 12.1:

```bash
pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu121
```

4. Install Isaac Gym Preview 4:

- Download Isaac Gym from NVIDIA: <https://developer.nvidia.com/isaac-gym>
- Install the Python package from the Isaac Gym directory:

```bash
cd isaacgym/python
pip install -e .
cd ../..
```

- Optionally verify the installation:

```bash
cd isaacgym/python/examples
python 1080_balls_of_solitude.py
```

5. Install the local PPO package from this repository:

```bash
cd rsl_rl
pip install -e .
cd ..
```

6. Install the local `legged_gym` package from this repository:

```bash
cd legged_gym
pip install -e .
cd ..
```

## Training

Run training from the repository root.

Example: train motor primitives for A1

```bash
python legged_gym/legged_gym/scripts/train.py --task=MP_a1 --num_envs=4096
```

Useful flags:

- `--headless`: disable rendering
- `--task`: choose one of the registered task names above
- `--num_envs`: set the number of parallel simulation environments

If rendering is enabled, pressing `v` toggles visualization in the Isaac Gym viewer.

## Playing a Pretrained Policy

Run a bundled A1 checkpoint with:

```bash
python legged_gym/legged_gym/scripts/play.py --task=Amortized_a1
```

The current play path resumes from the A1 checkpoint hardcoded in `legged_gym/legged_gym/utils/task_registry.py`. Bundled checkpoints currently exist under `legged_gym/logs/A1/`.

Included A1 checkpoints:

- `legged_gym/logs/A1/amortized-control/model_500.pt`
- `legged_gym/logs/A1/pace-slow/model_500.pt`
- `legged_gym/logs/A1/trot-slow/model_500.pt`
- `legged_gym/logs/A1/canter-slow/model_500.pt`
- `legged_gym/logs/A1/pace-fast/model_1000.pt`
- `legged_gym/logs/A1/trot-fast/model_1000.pt`
- `legged_gym/logs/A1/canter-fast/model_1000.pt`

If you want to evaluate Go2 checkpoints, update the resume path logic accordingly or provide your own saved models.

## Notes

- The intrinsic dynamics modules for both robots are implemented in `dynamics/dynamics.py` under each robot-specific environment directory.
- The Go2 environments are included in this repository, but pretrained Go2 checkpoints are not bundled under `legged_gym/logs/`.
- Some README instructions from upstream `legged_gym` or `rsl_rl` do not apply directly here because both packages were modified locally.

## Acknowledgements

This codebase builds on several excellent open-source projects:

- [DeepTransition](https://github.com/MiladShafiee/DeepTransition)
- [Isaac Gym](https://developer.nvidia.com/isaac-gym)
- [rsl_rl](https://github.com/leggedrobotics/rsl_rl)
- [legged_gym](https://github.com/leggedrobotics/legged_gym)

Please also refer to the original licenses included in this repository. All redistributed code retains its original license terms.
