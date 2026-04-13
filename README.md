[![CI - Python Lint & Tests](https://github.com/matgat1/safe-bimanual-rl/actions/workflows/continuous-integration.yml/badge.svg)](https://github.com/matgat1/safe-bimanual-rl/actions/workflows/continuous-integration.yml)
# safe-bimanual-rl
Safe Reinforcement learning for tray pickup with Safety Filters

This project focuses on training reinforcement learning agents for **two-arm robotic manipulation tasks** in simulation, with an emphasis on **safety constraints, stable learning, and reproducibility**.

We use **MuJoCo-based environments** and the **MushroomRL** library to design tasks such as object reaching and tray-pickup manipulation, while integrating safety-aware design choices into the learning pipeline.


![](figs/reach_cube_attempt0.gif)
***Reach Cube experiment*** trained for 12 epochs with 4000 steps per epoch (γ=0.99, horizon=200, n_substeps=4), using a replay buffer of 5,000–200,000 samples, batch size 256, 128 hidden features, 10,000 warm-up transitions, τ=0.001, and α learning rate of 3×10⁻⁴.


## Project Structure

```
├── figs                          # Folder containing figures for the ReadME
├── safe_bimanual_rl
│   ├── environments              # Folder containing environments
│   └── utils                     # Folder containing utils programs
├── tests                         # Folder containing test files
├── requirements.txt              # Python dependencies required to run the project
├── Makefile                      # Make commands to run/test...
└── README.md                     # Project description and documentation
```

## Setup 

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

---

### Run environments

#### MushroomRL bimanual environment

```bash
python3 safe_bimanual_rl/environments/bimanual_table_env.py
```

####  Reach environment

```bash
python3 safe_bimanual_rl/environments/reach_env.py
```

---

### Simple controller demo

```bash
python3 safe_bimanual_rl/utils/sinusoidal_controller.py
```

---

## Training

To train the RL agent on the reach task:

```bash
python -m safe_bimanual_rl.reach_point_experiment
```

