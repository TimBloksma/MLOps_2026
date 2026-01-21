````markdown
# MLOps UvA Bachelor AI Course: Medical Image Classification Skeleton Code

This repository contains an MLP model for patch-level classification on the PCAM dataset.

---

## 🚀 Quick Start

### 1. Installation
Clone the repository and set up your isolated environment.

```bash
# 1. Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 2. Install the package in "Editable" mode
pip install -e .

# 3. Install extra dependencies
pip install -r requirements.txt
````
### EXPERIMENT USE

```bash
python experiments/train.py --config experiments/configs/train_config.yaml
```

### SINGLE IMAGE

Use the `singleprediction.py` script to run a prediction on a sample image using a trained checkpoint:

```bash
python experiments/singleprediction.py \
    hydra.run.dir=. \
    +checkpoint_file=checkpoints/champion_lr0.01_bs64_hu64-32.pt
```

---


```text
.
├── src/ml_core/          # Source Code (Library)
│   ├── data/             # Data loaders and transformations
│   ├── models/           # PyTorch model architectures
│   ├── solver/           # Trainer class and loops
│   └── utils/            # Loggers and experiment trackers
├── experiments/          # The Laboratory
│   ├── configs/          # YAML files for hyperparameters
│   ├── results/          # Checkpoints and logs (Auto-generated)
│   ├── train.py          # Entry point for training
│   └── singleprediction.py # Run inference on a single image
├── scripts/              # Helper scripts (plotting, etc)
├── tests/                # Unit tests for QA
├── pyproject.toml        # Config for Tools (Ruff, Pytest)
└── setup.py              # Package installation script
```

---

Place the PCAM H5 files in the following folder structure:

```text
src/ml_core/data/pcam/
├── camelyonpatch_level_2_split_train_x.h5
├── camelyonpatch_level_2_split_train_y.h5
├── camelyonpatch_level_2_split_valid_x.h5
├── camelyonpatch_level_2_split_valid_y.h5
├── camelyonpatch_level_2_split_test_x.h5

```

---

To reproduce the best model:

```bash
python experiments/train.py \
    --config experiments/configs/train_config.yaml
```

Expected best model checkpoint:

```
checkpoints/champion_lr0.01_bs64_hu64-32.pt
```


Run single-image prediction with the checkpoint:

```bash
python experiments/singleprediction.py \
    hydra.run.dir=. \
    +checkpoint_file=checkpoints/champion_lr0.01_bs64_hu64-32.pt
```



