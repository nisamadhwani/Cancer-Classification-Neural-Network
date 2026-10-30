# Cancer Classification Using a Neural Network

## Overview
An educational binary classification project using the Breast Cancer Wisconsin Diagnostic dataset available through scikit-learn. A multilayer perceptron (MLP) neural network classifies records as malignant or benign.

## Dataset
- Source: `sklearn.datasets.load_breast_cancer()` (Wisconsin Diagnostic dataset)
- Records: 569
- Input features: 30 numeric measurements
- Target labels: 0 = malignant; 1 = benign
- The dataset CSV is included in `data/` for inspection and reproducibility.

## Method
1. Load the dataset and inspect feature/label structure.
2. Split into 80% training and 20% test/validation data with stratification.
3. Standardize features using training-only statistics.
4. Train an MLP with hidden layers (32, 16), ReLU activation and Adam optimizer.
5. Evaluate accuracy, precision, recall, F1 score and confusion matrix.

## Validation results
- Accuracy: 0.9211
- Precision: 1.0000
- Recall: 0.8750
- F1 score: 0.9333
- ROC AUC: 0.9957

## Run
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python src/train_evaluate.py
```

## Repository structure
- `data/`: dataset CSV
- `src/`: training/evaluation script
- `notebooks/`: explanatory notebook
- `results/`: validation predictions and metric plots
- `docs/`: report and presentation

## Important medical limitation
This is a student learning project using a public benchmark dataset. It is not clinically validated, does not replace a qualified clinician, and must not be used to diagnose or make treatment decisions.
