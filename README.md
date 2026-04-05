[![CI - Python Lint & Tests](https://github.com/matgat1/safe-bimanual-rl/actions/workflows/continuous-integration.yml/badge.svg)](https://github.com/matgat1/safe-bimanual-rl/actions/workflows/continuous-integration.yml)
# safe-bimanual-rl
Safe Reinforcement learning for tray pickup with Safety Filters


![Sinusoidal-Controller](figs/sinusoidal_controller.gif)

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

## How to use

### Visualize the Mujoco setup :

```bash
python3 safe_bimanual_rl/environments/visualise.py
```

You can also use a simple sinusoidal controller :

```bash
python3 safe_bimanual_rl/utils/sinusoidal_controller.py
```

### Visualize the MushroomRL mujoco environment :

```bash
python3 safe_bimanual_rl/environments/bimanual_table_env.py
```