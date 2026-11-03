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

Registered tasks are defined in `legged_gym/legged_gym/envs/__init__.py`. The A1 motor primitive tasks are `MP_full_a1` and `MP_interoceptive_a1`. The Go2 tasks remain `MP_go2` and `Amortized_go2`.

`MP_full_a1` uses `adaptive_equilibrium_dynamics.py`: its 19-action, four-gait controller inherits directly from `Intrinsic_Dynamics` and preserves the trajectory and 46 controller observations of the archived `1028_00-13-26_A1MPfull-signed-dense-v1` run.

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

Train the signed-dense full A1 policy (the configuration saved in the October 28 run):

```bash
python legged_gym/legged_gym/scripts/train.py --task=MP_full_a1 --num_envs=4096 --headless
```

Continue the latest full checkpoint for ten more PPO updates:

```bash
python legged_gym/legged_gym/scripts/train.py --task=MP_full_a1 --resume \
  --load_run=1028_00-13-26_A1MPfull-signed-dense-v1 --checkpoint=200 \
  --max_iterations=10 --headless
```

For an interoceptive checkpoint, use `--task=MP_interoceptive_a1` and the desired run directory under `legged_gym/logs/A1/`. For example, `--load_run=1018_15-43-25_A1MPinteroceptive-trot-snn-PolicyF_5e+00-MF_1e+03-noResume --checkpoint=40` loads `model_40_200.pt` and its saved trot configuration. Omitting `--load_run` with `--resume` selects the newest compatible run for that task. `--checkpoint` refers to the first number in `model_<index>_<iteration>.pt`; omitting it selects the highest index. `--max_iterations` is the number of additional updates.

The bundled checkpoints contain model weights but no optimizer state. Resumed training restores the saved network and iteration number, then starts a fresh optimizer.

Use `--headless` to disable rendering and `--num_envs` to change the number of simulation environments.

If rendering is enabled, pressing `v` toggles visualization in the Isaac Gym viewer.

## Playing a Pretrained Policy

Play the latest full A1 checkpoint with:

```bash
python legged_gym/legged_gym/scripts/play.py --task=MP_full_a1 \
  --load_run=1028_00-13-26_A1MPfull-signed-dense-v1
```

For one of the six interoceptive gaits, use `--task=MP_interoceptive_a1` and that gait's run directory. The play command loads the saved `config.json` before constructing the environment and policy, then selects the checkpoint with `--checkpoint` in the same way as training.

## Notes

- The active A1 intrinsic dynamics are in `legged_gym/legged_gym/utils/dynamics/a1/`.
- The Go2 environments are included in this repository, but pretrained Go2 checkpoints are not bundled under `legged_gym/logs/`.
- Some README instructions from upstream `legged_gym` or `rsl_rl` do not apply directly here because both packages were modified locally.

## Acknowledgements

This codebase builds on several excellent open-source projects:

- [DeepTransition](https://github.com/MiladShafiee/DeepTransition)
- [Isaac Gym](https://developer.nvidia.com/isaac-gym)
- [rsl_rl](https://github.com/leggedrobotics/rsl_rl)
- [legged_gym](https://github.com/leggedrobotics/legged_gym)

Please also refer to the original licenses included in this repository. All redistributed code retains its original license terms.
