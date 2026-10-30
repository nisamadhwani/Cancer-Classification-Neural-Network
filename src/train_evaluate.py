"""Breast cancer classification using a feed-forward neural network (MLP)."""
from pathlib import Path
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

ROOT = Path(__file__).resolve().parents[1]
# The scikit-learn Wisconsin Diagnostic dataset is used as specified in the project guide.
dataset = load_breast_cancer()
X = pd.DataFrame(dataset.data, columns=dataset.feature_names)
y = pd.Series(dataset.target, name="label")  # 0 = malignant, 1 = benign
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)
model = Pipeline([
    ("scaler", StandardScaler()),
    ("neural_network", MLPClassifier(
        hidden_layer_sizes=(32, 16), activation="relu", solver="adam",
        alpha=0.001, max_iter=500, early_stopping=True,
        n_iter_no_change=20, random_state=42
    ))
])
model.fit(X_train, y_train)
pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, pred))
print("Precision:", precision_score(y_test, pred))
print("Recall:", recall_score(y_test, pred))
print("F1:", f1_score(y_test, pred))
print("Confusion matrix (rows=true, columns=predicted; 0=malignant, 1=benign):")
print(confusion_matrix(y_test, pred))
print(classification_report(y_test, pred, target_names=["Malignant", "Benign"]))

# Example prediction using the first held-out record.
sample = X_test.iloc[[0]]
sample_pred = int(model.predict(sample)[0])
print("Example prediction:", "Benign" if sample_pred == 1 else "Malignant")
print("Educational output only; not a medical diagnosis.")
