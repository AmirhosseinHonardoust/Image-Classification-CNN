<div align="center">
       
# Image Classification with CNNs

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-CNN-orange)
![Dataset](https://img.shields.io/badge/Dataset-MNIST-green)
![Accuracy](https://img.shields.io/badge/Accuracy-99.35%25-brightgreen)
![Status](https://img.shields.io/badge/Status-Educational%20ML%20Project-purple)
[![CI](https://github.com/AmirhosseinHonardoust/Image-Classification-CNN/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/AmirhosseinHonardoust/Image-Classification-CNN/actions/workflows/ci.yml)

</div>

An image classification project built with **PyTorch**, using a **Convolutional Neural Network (CNN)** to recognize handwritten digits from the **MNIST** dataset, with **early stopping**, **best-model checkpointing**, **config-driven training**, evaluation metrics, and visual diagnostics.

> **Important:** This project is an **educational computer-vision demo**, not a production image-recognition system.
>
> The model, configuration, and reports are designed to demonstrate a clean, well-tested CNN training workflow. MNIST is a small, clean, well-studied benchmark; strong accuracy here does not imply the same architecture would perform well on noisy, real-world, or out-of-distribution images.

---

## Table of Contents

- [Project Overview](#project-overview)
- [What This Project Does](#what-this-project-does)
- [What This Project Does Not Do](#what-this-project-does-not-do)
- [Key Features](#key-features)
- [System Workflow](#system-workflow)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Training and Evaluation](#training-and-evaluation)
- [Configuration](#configuration)
- [Model Architecture](#model-architecture)
- [Model Output](#model-output)
- [Evaluation Metrics](#evaluation-metrics)
- [Visual Reports](#visual-reports)
- [Testing and CI](#testing-and-ci)
- [Code Quality](#code-quality)
- [Limitations](#limitations)
- [Responsible Use](#responsible-use)
- [Future Improvements](#future-improvements)
- [Tech Stack](#tech-stack)
- [Author](#author)
- [License](#license)

---

## Project Overview

Image classification is often the first serious computer-vision project people build, but it's easy to stop at "train a model, report accuracy" without building the surrounding workflow that makes a project reproducible and trustworthy:

- a typed, mergeable configuration instead of scattered hard-coded constants
- early stopping and checkpointing so training doesn't overfit or waste time
- consistent metrics written to disk instead of only printed to a terminal
- visual diagnostics that reveal *how* the model fails, not just whether it does

This project demonstrates an end-to-end, well-structured CNN training workflow on the MNIST handwritten-digit dataset. It includes a configurable training script, a separate evaluation script, typed configuration merging across defaults, JSON files, and the CLI, and a full test suite backed by CI.

The goal is to show how a standard CNN benchmark project can be turned into a **clean, reproducible engineering workflow**, not just a single accuracy number.

---

## What This Project Does

This project can:

- Train a CNN on the MNIST dataset (28×28 grayscale digit images)
- Apply early stopping and save the best-performing checkpoint
- Merge configuration from CLI flags, a JSON file, and sensible defaults
- Evaluate a trained checkpoint on the test set
- Compute accuracy, macro precision, macro recall, and macro F1
- Generate training curves, a confusion matrix, and sample predictions
- Save all metrics to a machine-readable `metrics.json`
- Run automated tests and a GitHub Actions CI workflow

---

## What This Project Does Not Do

This project does **not**:

- Classify arbitrary real-world photographs
- Handle color images, varying resolutions, or complex scenes
- Guarantee similar accuracy on noisy, rotated, or out-of-distribution digits
- Perform data augmentation beyond what is explicitly configured
- Serve the model behind an API or interactive app
- Replace a production-grade computer-vision pipeline

A production image-classification system would need a much larger and more diverse dataset, data augmentation, robustness testing, model serving infrastructure, and ongoing monitoring.

---

## Key Features

- **CNN architecture** with two convolution + ReLU + pooling blocks followed by fully connected layers
- **Adam optimizer** with **CrossEntropyLoss**
- **Early stopping** with a configurable patience, paired with **best-checkpoint saving**
- **Typed configuration system** (`TrainConfig` / `EvalConfig`) merging CLI flags, an optional JSON config file, and defaults
- **Separate training and evaluation scripts** so a checkpoint can be evaluated independently of training
- **Automatic MNIST download** via torchvision on first run
- **Visual diagnostics**: training curves, confusion matrix, sample predictions
- **Machine-readable metrics** written to `metrics.json`
- **Unit tests and GitHub Actions CI**
- **Pre-commit hooks** for Ruff, Black, and mypy

---

## System Workflow

```text
MNIST dataset (auto-downloaded via torchvision)
        ↓
Config merge (defaults + JSON file + CLI flags)
        ↓
CNN training (Conv → ReLU → Pool ×2 → Fully Connected)
        ↓
Adam optimizer + CrossEntropyLoss
        ↓
Early stopping + best checkpoint (best_cnn.pth)
        ↓
Evaluation on the test set
        ↓
Accuracy / precision / recall / F1 (metrics.json)
        ↓
Training curves, confusion matrix, sample predictions
```

---

## Project Structure

```text
image-classification-cnn/
│
├── .github/
│   └── workflows/
│       └── ci.yml
│
├── data/                       # auto-downloaded MNIST dataset (git-ignored)
│
├── outputs/
│   ├── metrics.json
│   ├── confusion_matrix.png
│   ├── training_curves.png
│   ├── sample_predictions.png
│   └── best_cnn.pth
│
├── src/
│   └── image_classification_cnn/
│       ├── __init__.py
│       ├── class_names.py       # shared dataset label constants (e.g. CIFAR10)
│       ├── config.py            # typed TrainConfig/EvalConfig + CLI/file/default merge
│       ├── model.py             # SimpleCNN architecture
│       ├── train_cnn.py         # training script (also installed as `train-cnn`)
│       ├── evaluate.py          # evaluation script (also installed as `evaluate-cnn`)
│       └── utils.py             # helper functions
│
├── tests/                       # pytest suite for the modules above
│
├── .dockerignore
├── .gitignore
├── .pre-commit-config.yaml
├── CONTRIBUTING.md
├── Dockerfile
├── LICENSE
├── Makefile
├── pyproject.toml
├── requirements.txt
├── requirements-dev.txt
├── train_config.example.json
└── README.md
```

---

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/AmirhosseinHonardoust/Image-Classification-CNN.git
cd Image-Classification-CNN
```

### 2. Create a Virtual Environment

On Windows CMD:

```cmd
python -m venv .venv
.venv\Scripts\activate
```

On macOS/Linux:

```bash
python -m venv .venv
source .venv/bin/activate
```

### 3. Install Requirements

```bash
pip install -r requirements.txt
pip install -e .
```

The `pip install -e .` step installs the `train-cnn` and `evaluate-cnn`
console scripts used below (an editable install, so local code edits take
effect immediately).

For development tools (pytest, Ruff, Black, mypy):

```bash
pip install -r requirements-dev.txt
```

---

## Quick Start

Train the model:

```bash
train-cnn --dataset mnist --epochs 10 --batch-size 64 --outdir outputs
```

Evaluate the trained checkpoint:

```bash
evaluate-cnn --model outputs/best_cnn.pth --dataset mnist --outdir outputs
```

`data/` is downloaded automatically by torchvision on first run, and `outputs/` is written by the training and evaluation scripts. Both are git-ignored.

---

## Training and Evaluation

The training script downloads MNIST if needed, builds the CNN, trains with early stopping, and writes the best checkpoint:

```bash
train-cnn \
  --dataset mnist \
  --epochs 10 \
  --batch-size 64 \
  --outdir outputs
```

Flags can also be supplied from a JSON file:

```bash
train-cnn --config train_config.example.json
```

Any flag passed explicitly on the CLI still overrides the same key from the config file, which itself overrides the built-in defaults.

The evaluation script loads a saved checkpoint and reports metrics on the test split:

```bash
evaluate-cnn --model outputs/best_cnn.pth --dataset mnist --outdir outputs
```

Generated outputs include:

```text
outputs/metrics.json
outputs/training_curves.png
outputs/confusion_matrix.png
outputs/sample_predictions.png
outputs/best_cnn.pth
```

### Docker

A `Dockerfile` is included for reproducible training without a local Python
environment:

```bash
docker build -t image-classification-cnn .
docker run --rm -v "$(pwd)/outputs:/app/outputs" image-classification-cnn
```

This runs `train_cnn.py --config train_config.example.json` by default; pass
different args after the image name to override (e.g. `--dataset cifar10`).

---

## Configuration

Configuration is layered, with each layer overriding the one before it:

```text
built-in defaults  →  JSON config file (--config)  →  explicit CLI flags
```

Example `train_config.example.json`:

```json
{
  "dataset": "mnist",
  "epochs": 10,
  "batch_size": 64,
  "lr": 0.001,
  "outdir": "outputs",
  "patience": 3
}
```

<div align="center">

| Key | Meaning |
|---|---|
| `dataset` | Which dataset to train on |
| `epochs` | Maximum number of training epochs |
| `batch_size` | Mini-batch size for training and evaluation |
| `lr` | Learning rate for the Adam optimizer |
| `outdir` | Directory where checkpoints and reports are written |
| `patience` | Number of epochs without improvement before early stopping triggers |

</div>

This merge behavior is covered by the test suite in `tests/`, so a change to the merge order or precedence would be caught by CI.

---

## Model Architecture

`src/model.py` defines a compact CNN sized for 28×28 grayscale input:

<div align="center">

| Stage | Operation |
|---|---|
| Block 1 | Conv2d → ReLU → MaxPool2d |
| Block 2 | Conv2d → ReLU → MaxPool2d |
| Head | Flatten → Fully Connected → output logits |

</div>

Training uses the **Adam** optimizer with **CrossEntropyLoss**. Early stopping monitors validation performance and halts training once it fails to improve for `patience` consecutive epochs, and the best-performing checkpoint (not necessarily the last epoch) is the one saved to `outputs/best_cnn.pth`.

---

## Model Output

Running evaluation prints and saves a metrics summary. With the bundled MNIST run:

<div align="center">

| Metric | Value |
|---|---|
| Accuracy | 99.35% |
| Precision (macro) | 0.9935 |
| Recall (macro) | 0.9934 |
| F1 Score (macro) | 0.9934 |

</div>

These are also written to a machine-readable `outputs/metrics.json` alongside the checkpoint used to produce them.

---

## Evaluation Metrics

Evaluation runs on the standard MNIST test split, held out from training.

<div align="center">

| Metric | Why it matters |
|---|---|
| Accuracy | Overall correctness across all ten digit classes |
| Macro Precision | Average precision treating every digit class equally |
| Macro Recall | Average recall treating every digit class equally |
| Macro F1 | Balances precision and recall across all classes |

</div>

> MNIST is a clean, balanced, well-studied dataset. High accuracy here reflects strong performance on this benchmark, not a guarantee of similar performance on noisier or more diverse real-world images.

---

## Visual Reports

### Training and evaluation charts

<div align="center">

| Confusion Matrix | Training Curves |
|---|---|
| <img width="760" height="760" alt="confusion_matrix" src="https://github.com/user-attachments/assets/516b6152-7cda-4f44-b2dc-89a3f8565677" /> | <img width="1120" height="800" alt="training_curves" src="https://github.com/user-attachments/assets/2cf61fd1-e025-4b05-8cdb-75d6db5ce0a4" /> |
| **Analysis:** The confusion matrix shows which digits the model confuses most often on the test set. Even at 99%+ accuracy, the per-class view highlights the handful of visually similar digit pairs that remain hardest. | **Analysis:** The training curves show loss and accuracy across epochs for both the training and validation sets, making it easy to see where early stopping triggered. |

</div>

### Sample predictions

<div align="center">
        
|Sample predictions|
|---|
| <img width="480" height="480" alt="sample_predictions" src="https://github.com/user-attachments/assets/77eb3e98-044c-4973-9125-2fffec5b2405" /> |   
|The sample-predictions grid shows a batch of test digits alongside the model's predicted labels, giving a quick qualitative sanity check beyond the aggregate metrics.|

</div>

---

## Testing and CI

Run unit tests locally:

```bash
pytest
```

Run the same checks CI runs, in one step, via the Makefile:

```bash
make check
```

Or individually:

```bash
ruff check src tests
black --check src tests
mypy src
pytest
```

The GitHub Actions workflow (`.github/workflows/ci.yml`) runs these same checks on every push and pull request:

- dependency installation
- linting with Ruff
- formatting check with Black
- static type checking with mypy
- unit tests with pytest

---

## Code Quality

The project separates responsibilities across modules:

<div align="center">

| Module | Purpose |
|---|---|
| `src/image_classification_cnn/config.py` | Typed `TrainConfig` / `EvalConfig` and the defaults → file → CLI merge logic |
| `src/image_classification_cnn/model.py` | The `SimpleCNN` architecture |
| `src/image_classification_cnn/train_cnn.py` | Training loop, early stopping, checkpoint saving (installed as `train-cnn`) |
| `src/image_classification_cnn/evaluate.py` | Loads a checkpoint and computes test-set metrics and charts (installed as `evaluate-cnn`) |
| `src/image_classification_cnn/utils.py` | Shared helper functions |

</div>

Tooling is configured through `pyproject.toml` (Ruff, Black, mypy, pytest) and `requirements-dev.txt`. Optional pre-commit hooks (`.pre-commit-config.yaml`) run the same linting, formatting, and type checks automatically before each commit:

```bash
pip install pre-commit
pre-commit install
```

---

## Limitations

This project has important limitations:

- MNIST is a small, clean, single-domain benchmark, not a real-world image corpus
- The model only handles fixed-size, grayscale, centered digit images
- High accuracy on MNIST does not transfer to noisy, rotated, or occluded inputs
- No data augmentation is applied beyond what is explicitly configured
- The model is not evaluated for robustness against adversarial or out-of-distribution inputs
- There is no inference API or deployment tooling included

The project is strongest as a portfolio demonstration of a clean, well-tested, config-driven CNN training workflow.

---

## Responsible Use

This repository is intended for:

- machine learning and computer-vision education
- learning CNN architecture design and training loops
- practicing config-driven, testable ML project structure
- portfolio demonstration

It should not be used as-is for:

- production image classification or recognition systems
- safety-critical or high-stakes decision-making
- classifying image domains outside handwritten digits without retraining
- any deployment without further validation and testing

Any real deployment would require a larger and more diverse dataset, robustness and fairness evaluation, and a proper serving and monitoring setup.

---

## Future Improvements

Potential next improvements:

- Add support for additional datasets (e.g. Fashion-MNIST, CIFAR-10)
- Add data augmentation (rotation, shift, noise) and measure its effect
- Add a confusion-matrix-driven error analysis report
- Add TensorBoard or Weights & Biases logging
- Add ONNX export and a minimal inference script
- Add Docker support for reproducible training
- Explore deeper architectures and compare against the baseline CNN

---

## Tech Stack

- Python
- PyTorch
- torchvision
- NumPy
- matplotlib
- pytest
- Ruff
- Black
- mypy
- GitHub Actions

---

## Author

**Amir Honardoust**

GitHub: [@AmirhosseinHonardoust](https://github.com/AmirhosseinHonardoust)

---

## License

This project is licensed under the MIT License. See `LICENSE` for details.
