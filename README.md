```
physinet-operator/
│
├── physigym/                 # Synthetic PDE simulator (JAX)
│   ├── __init__.py
│   ├── configs.py            # Config dataclasses
│   ├── pde_wave.py           # 2D wave equation
│   ├── pde_heat.py           # 2D heat/diffusion equation
│   ├── pde_reacdiff.py       # Reaction–diffusion (Gray-Scott)
│   ├── env.py                # Unified gym-like interface + dataset generator
│   └── utils.py              # Grids, plotting hooks, helpers
│
├── physinet/                 # Neural operator models
│   ├── __init__.py
│   ├── encoder.py            # Multiscale encoder
│   ├── fno.py                # Fourier Neural Operator blocks
│   ├── head.py               # Deterministic + probabilistic heads
│   ├── model.py              # End-to-end PhysiNetOperator model
│   ├── data.py               # Data loader utilities
│   └── train.py              # Training loop
│
├── notebooks/                # Optional: exp notebooks
│   ├── 01_generate_data.ipynb
│   ├── 02_train_operator.ipynb
│   └── 03_visualize_rollouts.ipynb
│
├── data/
│   ├── synthetic/            # Generated trajectories (.npz)
│   └── external/             # Optional external PDE datasets
│
├── README.md
├── requirements.txt
└── pyproject.toml / setup.cfg (optional, kalau mau dipackage)
```
