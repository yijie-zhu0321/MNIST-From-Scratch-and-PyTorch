# MNIST Classification: NumPy from Scratch and PyTorch

**Author:** Yijie Zhu  
**Focus:** Machine-learning fundamentals, backpropagation, model evaluation, and deep-learning workflows

## Overview

This learning portfolio contains two implementations of handwritten-digit classification on the MNIST dataset:

1. **NumPy:** A fully connected neural network with manual forward propagation, softmax cross-entropy, backpropagation, and stochastic gradient descent (SGD).
2. **PyTorch:** A deeper multilayer perceptron (MLP) using PyTorch automatic differentiation, Adam optimization, and error analysis.

These are **educational implementations**, not novel research models. The two models have different architectures and optimizers; their test accuracies should not be interpreted as a controlled comparison.

## Implementations

| | NumPy | PyTorch |
|---|---|---|
| Architecture | 784 → 128 → 10 | 784 → 128 → 64 → 10 |
| Hidden-layer activation | ReLU | ReLU |
| Loss | Softmax cross-entropy | Cross-entropy |
| Gradients | Manually implemented | Autograd |
| Optimizer | SGD, lr=0.01 | Adam, lr=0.001 |
| Epochs | 5 | 5 |
| Training | Per-example updates | Mini-batches of 64 |

## Repository layout

```text
MNIST-From-Scratch-and-PyTorch/
├── numpy/                     # Manual neural network and training code
│   ├── train.py
│   ├── test.py
│   ├── download_data.py
│   └── legacy/                # Preserved early experiment (not runnable)
├── pytorch/
│   ├── mnist.py              # Training + evaluation
│   ├── mnist_visualization.py # Standalone error analysis
│   └── legacy/                # Preserved original visualization exercises
├── figures/                   # Saved analysis plots after running scripts
├── tests/test_smoke.py        # Synthetic-input sanity and gradient checks
├── requirements.txt
└── .gitignore
```

## Setup (macOS / Linux)

Run from the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
```

## Run the NumPy model

```bash
python numpy/download_data.py
python numpy/train.py
python numpy/test.py
```

`numpy/train.py` trains with **one image at a time** using plain NumPy; expect it to be much slower than PyTorch. It writes `numpy/mnist_model.npz`. If you already have the original trained checkpoint, you can copy it to `numpy/mnist_model.npz` and run `numpy/test.py` directly after downloading the dataset.

## Run the PyTorch model

```bash
python pytorch/mnist.py
python pytorch/mnist_visualization.py
```

The first script downloads MNIST if needed, trains the MLP, prints test loss and accuracy, saves `pytorch/mnist_mlp.pth`, and writes training/error figures to `figures/`. The second script loads the saved model **without retraining** and creates additional error-analysis plots and a confusion matrix.

## Verified Results

Both models were evaluated using their saved checkpoints on the standard MNIST test set of 10,000 handwritten-digit images.

| Implementation | Architecture | Test Accuracy |
|---|---|---|
| NumPy MLP | 784 → 128 → 10 | **97.15%** (9,715/10,000) |
| PyTorch MLP | 784 → 128 → 64 → 10 | **97.46%** (9,746/10,000) |

The NumPy model implements forward propagation, backpropagation, and stochastic gradient descent manually. The PyTorch model uses automatic differentiation and the Adam optimizer.

Because the architectures and training procedures differ, these results should not be interpreted as a controlled comparison.

The PyTorch evaluation also generates a confusion matrix, highest-loss examples, and confidently misclassified examples.


## What this project demonstrates

- Implementing a small neural network and backpropagation with NumPy
- Understanding cross-entropy, ReLU, gradients, and SGD updates
- Training and evaluating a multilayer perceptron using PyTorch
- Inspecting misclassifications, high-loss predictions, and a confusion matrix
- Organizing code into reusable, documented, reproducible scripts

## Notes on reproducibility

- The NumPy and PyTorch programs both use a fixed seed of 42 for model initialization, but exact results can vary with hardware and library versions.
- MNIST files are **not committed**; the scripts download them when needed.
- Original early-stage exercises are kept in `legacy/` and are not intended to run standalone.
- Trained checkpoints (`.npz` and `.pth`) are not included in the starter package; you can copy your existing checkpoints into the corresponding project folders.
- The original saved checkpoints were successfully evaluated on the MNIST test set. Full retraining of both models using the reorganized scripts has not yet been independently verified.

## Related project

[SHP2 Inhibitor SAR Analysis and Machine Learning Validation](https://github.com/yijie-zhu0321/SHP2-Inhibitor-SAR-ml) — a separate medicinal-chemistry application of machine learning.
