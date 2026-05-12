[![CI - Python Lint & Tests](https://github.com/matgat1/safe-bimanual-rl/actions/workflows/continuous-integration.yml/badge.svg)](https://github.com/matgat1/safe-bimanual-rl/actions/workflows/continuous-integration.yml)
# safe-bimanual-rl
Safe Reinforcement learning for tray pickup with Safety Filters

This project focuses on training reinforcement learning agents for **two-arm robotic manipulation tasks** in simulation, with an emphasis on **safety constraints, stable learning, and reproducibility**.

We use **MuJoCo-based environments** and the **MushroomRL** library to design tasks such as object reaching and tray-pickup manipulation, while integrating safety-aware design choices into the learning pipeline.

---
***Reach Cube experiment*** trained for 250 epochs


<img src="figs/reach_cube_agent_2026-04-19_20-21-46_52.gif" width="48%"/> <img src="figs/reach_cube_agent_2026-04-19_20-21-46_44.gif" width="48%"/> 



## Project Structure

```
├── figs/                         # Figures and GIFs used in the README
├── safe_bimanual_rl/
│   ├── configs/                  # Hydra config files (e.g. reach_cube_sac.yaml)
│   ├── environments/             # MuJoCo environments (bimanual, reach, tray pickup)
│   ├── rl_utils/                 # SAC networks and plotting helpers
│   ├── utils/                    # Evaluation, controller, and data collection utilities
│   ├── reach_point_experiment_sac.py      # Training: reach task
│   ├── tray_pickup_reach_train_sac.py     # Training: tray pickup — reach phase
│   └── tray_pickup_grasp_train_sac.py     # Training: tray pickup — grasp phase
├── tests/                        # Unit and integration tests
├── environment.yml               # Conda environment definition
├── Makefile                      # Shortcuts for common commands
└── README.md                     # Project description and documentation
```

## Setup

**Requirements:** Python 3.12, MuJoCo 3.6, PyTorch 2.9, CUDA 13.1, Hydra ≥1.3, WandB (see `environment.yml` for full details).

It is recommended to use the Conda environment.

Create and activate the environment:

```bash
conda env create -f environment.yml
conda activate safe_bimanual_rl
```

Or update it if already created:

```bash
conda env update -f environment.yml --prune
```

## How to use

### Visualize MuJoCo setup

```bash
python3 safe_bimanual_rl/environments/visualise.py
```

### Run environments

#### MushroomRL bimanual environment

```bash
python3 safe_bimanual_rl/environments/bimanual_table_env.py
```

####  Reach environment

```bash
python3 safe_bimanual_rl/environments/reach_env.py
```


### Simple controller demo

```bash
python3 safe_bimanual_rl/utils/sinusoidal_controller.py
```


## Training

To train the RL agent on the reach task:

```bash
python -m safe_bimanual_rl.reach_point_experiment_sac
```

The default configuration is in `configs/reach_cube_sac.yaml`. You can override any parameter directly from the command line:

```bash
python -m safe_bimanual_rl.reach_point_experiment_sac \
    n_epochs=100 \
    model_name="test" \
    contact_threshold=1.0
```

To run multiple experiments with different parameters:

```bash
python -m safe_bimanual_rl.reach_point_experiment_sac --multirun \
    contact_threshold=1.0,5.0,20.0
```

To use on the cluster:

```bash
python -m safe_bimanual_rl.reach_point_experiment_sac --multirun \
  hydra/launcher=cosmos \
  contact_threshold=1.0,2.0,3.0
```

### Tray pickup — reach phase

Train the agent to move the end-effectors to the tray handle positions with the correct orientation:

```bash
python -m safe_bimanual_rl.tray_pickup_reach_train_sac
```

Override key parameters:

```bash
python -m safe_bimanual_rl.tray_pickup_reach_train_sac \
    n_epochs=130 \
    model_name="reach_test" \
    reach_sharpness=0.4 \
    success_position_reward=500.0 \
    save_model=true \
    use_wandb=false
```

Sweep on the cluster:

```bash
python -m safe_bimanual_rl.tray_pickup_reach_train_sac --multirun \
  hydra/launcher=cosmos \
  reach_sharpness=0.3,0.4,0.5 \
  success_position_reward=100.0,500.0
```

### Tray pickup — grasp phase

Train the agent to apply the correct contact forces on the tray handles:

```bash
python -m safe_bimanual_rl.tray_pickup_grasp_train_sac
```

Override key parameters:

```bash
python -m safe_bimanual_rl.tray_pickup_grasp_train_sac \
    n_epochs=150 \
    model_name="grasp_test" \
    success_grasp_reward=30.0 \
    grasp_force_threshold=0.3 \
    contact_threshold=5.0 \
    save_model=true \
    use_wandb=false
```

Sweep on the cluster:

```bash
python -m safe_bimanual_rl.tray_pickup_grasp_train_sac --multirun \
  hydra/launcher=cosmos \
  success_grasp_reward=15.0,30.0 \
  grasp_force_threshold=0.2,0.3,0.5
```

## Utilities

### Collect absorbing positions

After training a reach model, collect the joint configurations at successful episode ends to use as initial states for the grasp phase:

```bash
python -m safe_bimanual_rl.utils.collect_absorbing_positions \
    --model_path "models/reach_best.msh"
```

With all options:

```bash
python -m safe_bimanual_rl.utils.collect_absorbing_positions \
    --model_path "models/reach_best.msh" \
    --n_episodes 50 \
    --output_path "safe_bimanual_rl/environments/data/initial_states/grasp_init_states_1.npz" \
    --render
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `--model_path` | `str` | required | Path to the saved `.msh` reach model |
| `--n_episodes` | `int` | `20` | Number of episodes to run |
| `--output_path` | `str` | `absorbing_positions.npz` | Path to save the collected states |
| `--render` | flag | `False` | Render the environment during collection |

## Evaluate models

To evaluate and display a model:

```bash
python -m safe_bimanual_rl.utils.evaluate_model --model_path "models/test.msh"
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `--model_path` | `str` | required | Path to the saved `.msh` model file |
| `--n_episodes` | `int` | `3` | Number of evaluation episodes |
| `--record` | flag | `False` | Save a video recording of the evaluation |
| `--env` | `str` | auto-detected | Environment to use (`reach_cube` or `tray_pickup`). Auto-detected from model path if not provided. |

Example with all options:

```bash
python -m safe_bimanual_rl.utils.evaluate_model \
    --model_path "models/tray_pickup_agent.msh" \
    --n_episodes 10 \
    --record \
    --env tray_pickup
```
