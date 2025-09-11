# Image Classification with CNNs
Image classification with PyTorch using Convolutional Neural Networks (CNNs). Trains on MNIST with convolution, pooling, and fully connected layers. Achieves over 99% accuracy with early stopping and checkpoints. Includes training/evaluation scripts, metrics, confusion matrix, training curves, and sample prediction visualizations.

This project implements a Convolutional Neural Network (CNN) using **PyTorch** to classify handwritten digits from the MNIST dataset. The model achieves over **99% accuracy** and includes training scripts, evaluation, and visualizations.

---

## Features
- Dataset: MNIST (28×28 grayscale digits)
- CNN architecture: Conv → ReLU → Pool × 2 + Fully Connected
- Optimizer: Adam; Loss: CrossEntropyLoss
- Early stopping & best model checkpoint
- Visualizations: training curves, confusion matrix, sample predictions
- Outputs metrics to `metrics.json`

## Results
- **Accuracy:** 99.35%
- **Precision (macro):** 0.9935
- **Recall (macro):** 0.9934
- **F1 Score (macro):** 0.9934

---
### Confusion Matrix
<img width="960" height="960" alt="confusion_matrix" src="https://github.com/user-attachments/assets/effd44e9-02a9-4cd3-879e-a19fa46393d3" />

---
### Training Curves
<img width="1120" height="800" alt="training_curves" src="https://github.com/user-attachments/assets/73e60a0b-bfb5-4c6d-94d9-c4e1afab1ddb" />

---
### Sample Predictions
<img width="1280" height="1280" alt="sample_predictions" src="https://github.com/user-attachments/assets/7978af96-6c5c-4ec9-a486-7fd5387e9e24" />

---
## Project Structure
```
image-classification-cnn/
├─ README.md
├─ LICENSE
├─ requirements.txt
├─ data/                # auto-downloaded MNIST dataset
├─ src/
│  ├─ train_cnn.py      # training script
│  ├─ evaluate.py       # evaluation script
│  └─ utils.py          # helper functions
└─ outputs/
   ├─ metrics.json
   ├─ confusion_matrix.png
   ├─ training_curves.png
   ├─ sample_predictions.png
   └─ best_cnn.pth
```

## Setup
```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate
pip install -r requirements.txt
```

## Train
```bash
python src/train_cnn.py --dataset mnist --epochs 10 --batch-size 64 --outdir outputs
```

## Evaluate
```bash
python src/evaluate.py --model outputs/best_cnn.pth --dataset mnist --outdir outputs
```

**Outputs**
- `outputs/metrics.json` – accuracy, precision, recall, F1
- `outputs/training_curves.png`
- `outputs/confusion_matrix.png`
- `outputs/sample_predictions.png`
- `outputs/best_cnn.pth`
