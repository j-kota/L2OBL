# Learning to Solve Constrained Bilevel Control Co-Design Problems

Code for the paper:

**Learning to Solve Constrained Bilevel Control Co-Design Problems**  
arXiv: https://arxiv.org/abs/2507.09050v2

This repository contains experimental code for *learning-to-optimize* / *differentiable optimization* approaches to **constrained bilevel problems** arising in **control co-design**. The repo is organized into three self-contained experiments:

- **BQP/**: Bilevel Quadratic Program (QP)
- **HVAC/**: Building HVAC bilevel control/co-design (multi-zone capable)
- **TT/**: Two-tank system co-design with differentiable MPC

> Note: most experiment scripts save figures to `./plt/` and metrics/artifacts to `./pickle/` (and `./models/` for TT). Create these directories before running if they do not already exist.

---

## Repository layout

```text
.
├── master_env.yaml          # (optional) a shared conda env with most dependencies
├── BQP/
│   ├── main.py              # main BQP experiment
│   ├── QP_layer.py          # differentiable QP layer (cvxpylayers)
│   ├── QP_correction.py     # feasibility / coupling correction routine
│   ├── param_sol_data_*.mat # example datasets
│   └── BLOenv.yaml          # minimal env for BQP
├── HVAC/
│   ├── main.py              # main HVAC experiment
│   ├── building_MPC_layer.py
│   ├── building_correction.py
│   ├── data/                # example dataset(s)
│   └── HVACenv.yaml
└── TT/
    ├── main.py              # main two-tank experiment (diff MPC)
    ├── twotank_diffmpc.py   # differentiable MPC closed-loop solver
    ├── xf_dev_standard.p    # dev set target states
    └── TTenv.yaml
```

---

## Installation

### Option A: one shared conda environment (recommended)

```bash
conda env create -f master_env.yaml
conda activate master_env
```

This environment targets **Python 3.10** and includes `pytorch`, `cvxpy`, `diffcp`, `neuromancer`, `pyswarms`, and other common deps used across experiments.

### Option B: per-experiment environments

Each experiment folder contains its own environment file:

```bash
conda env create -f BQP/BLOenv.yaml
conda env create -f HVAC/HVACenv.yaml
conda env create -f TT/TTenv.yaml
```

Activate the one you need, e.g.:

```bash
conda activate BLOenv
```

---

## Running experiments

### Common setup

Many scripts write outputs under the experiment directory:

```bash
mkdir -p BQP/plt BQP/pickle
mkdir -p HVAC/plt HVAC/pickle
mkdir -p TT/plt TT/pickle TT/models
```

### BQP (Bilevel QP)

Run from the `BQP/` folder:

```bash
cd BQP
python main.py \
  --input_file param_sol_data_3_2_10000.mat \
  --epochs 30 \
  --n_corr_steps 10
```

Key flags:

- `--input_file`: `.mat` dataset containing problem parameters/solutions
- `--n_corr_steps`, `--alpha`: correction hyperparameters
- `--penalty`, `--lr`, `--epochs`: training hyperparameters

Outputs:

- Plots saved to `BQP/plt/`
- Metrics saved to `BQP/pickle/QP_outdict*.p`

### HVAC (Building bilevel control/co-design)

Run from the `HVAC/` folder:

```bash
cd HVAC
python main.py \
  --ntrain 10000 \
  --ntest 1000 \
  --nsteps 30 \
  --nzones 1 \
  --epochs 10
```

The default dataset path is:

```text
HVAC/data/zone_data_{nsteps}steps_{ntrain+ntest}samples.p
```

Outputs:

- Plots saved to `HVAC/plt/`
- Metrics saved to `HVAC/pickle/outdict*.p`

### TT (Two-tank differentiable MPC co-design)

Run from the `TT/` folder:

```bash
cd TT
python main.py \
  --n_train 3000 \
  --n_dev 1000 \
  --epochs 30 \
  --n_corr_steps 3
```

Outputs:

- Plots saved to `TT/plt/`
- Model checkpoints saved to `TT/models/`
- Metrics/dev artifacts saved to `TT/pickle/tt_diffmpc_outdict*.p`

---

## Notes on reproducibility

- Seeds are set inside some scripts (e.g., `HVAC/main.py`, `TT/main.py`), but results may still vary across hardware/software.
- If you run into solver/numerical issues, try reducing learning rate, reducing correction step size (`--alpha`), or increasing the number of correction steps (`--n_corr_steps`).

---

## Citation

If you use this code, please cite:

```bibtex
@article{l2obl2025,
  title   = {Learning to Solve Constrained Bilevel Control Co-Design Problems},
  journal = {arXiv preprint arXiv:2507.09050},
  year    = {2025},
  url     = {https://arxiv.org/abs/2507.09050v2}
}
```

---

## License

Add a license file (e.g., MIT/BSD/Apache-2.0) if you plan to make this repository publicly reusable.
