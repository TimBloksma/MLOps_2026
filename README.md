# MLOps UvA Bachelor AI Course: Medical Image Classification Skeleton Code

![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)
![Build Status](https://github.com/yourusername/mlops_course/actions/workflows/ci.yml/badge.svg)
![Code Style: Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)

A repo exemplifying **MLOps best practices**: modularity, reproducibility, automation, and experiment tracking.

This project implements a standardized workflow for training neural networks on medical data (PCAM/TCGA). 

The idea is that you fill in the repository with the necessary functions so you can execute the ```train.py``` function. Please also fill in this ```README.md``` clearly to setup, install and run your code. 

Don't forget to setup CI and linting!
---
# PCAM Histopathology Classification - Group 17

**Group Members:** [Lars van der Groep, Berend Veltkamp, Tim Bloksma, Julius Rademakers en Mingus Gaston]
**Student IDs:** [ , , , ,15113019]
**Course:** MLOps & ML Programming 2026, University of Amsterdam
**Repository:** [https://github.com/TimBloksma/MLOps_2026]

---
## Overview
This repository contains a complete MLOps pipeline for binary classification of histopathology images 
from the PCAM (PatchCamelyon) dataset. The goal is to detect metastatic tissue in 96x96 image patches using deep 
learning with proper MLOps practices.

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

# 3. Install pre-commit hooks
pre-commit install
```

### 2. Verify Setup
```bash
pytest tests/
```

### 3. Run an Experiment
```bash
python experiments/train.py --config experiments/configs/train_config.yaml
```
---
## Data setup

### 1. Download the PCAM dataset from SURFDrive and place the H5 files in the data/ directory:

data/
├── camelyonpatch_level_2_split_train_x.h5
├── camelyonpatch_level_2_split_train_y.h5
├── camelyonpatch_level_2_split_valid_x.h5
├── camelyonpatch_level_2_split_valid_y.h5
├── camelyonpatch_level_2_split_test_x.h5
└── camelyonpatch_level_2_split_test_y.h5

Note: Due to size constraints, the actual H5 files are not included in this repository.

```bash
### 2. Verify the setup:
pytest tests/
```

```bash
### 3. Run an experiment:
python experiments/train.py --config experiments/configs/train_config.yaml
```

---
## Training

```bash
### To reproduce our best model:
**python experiments/train.py --config experiments/configs/champion_model.yaml
```

### Expected Performance:
 
**Validation Accuracy: ~...%**

**Test Accuracy: ~...%**

**AUC-ROC: ~...**

**F1-Score: ~...**

**Precision: ~...**

**Recall: ~...**

### Training Configuration
```bash
Hier komt die configuration van champion model :))

```


---
## Inference


Weet ik niet, we hebben ook nog geen checkpoints



---
## 📂 Project Structure

Dit staat er nu maar moet morgen worden veranderd naar ons final product.

```text
.
├── src/ml_core/          # The Source Code (Library)
│   ├── data/             # Data loaders and transformations
│   ├── models/           # PyTorch model architectures
│   ├── solver/           # Trainer class and loops
│   └── utils/            # Loggers and experiment trackers
├── experiments/          # The Laboratory
│   ├── configs/          # YAML files for hyperparameters
│   ├── results/          # Checkpoints and logs (Auto-generated)
│   └── train.py          # Entry point for training
├── scripts/              # Helper scripts (plotting, etc)
├── tests/                # Unit tests for QA
├── pyproject.toml        # Config for Tools (Ruff, Pytest)
└── setup.py              # Package installation script
```
