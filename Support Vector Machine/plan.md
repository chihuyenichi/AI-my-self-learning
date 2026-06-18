# Support Vector Machine (SVM)

## Summary
A Support Vector Machine (SVM) is a powerful and versatile supervised Machine Learning model capable of performing linear and nonlinear classification, regression, and outlier detection. It is particularly well-suited for complex but small to medium-sized datasets. The core idea is to find a decision boundary (hyperplane) that separates classes with the largest possible margin.

## Definition
SVM is a discriminative classifier defined by a separating hyperplane that maximizes the margin between classes. The model is trained to find the optimal hyperplane $$\mathbf{w}^T \mathbf{x} + b = 0$$ such that the distance to the nearest training points (support vectors) of any class is maximized.

- **Hard margin SVM**: Requires all instances to be correctly classified and outside the margin. Only works for linearly separable data and is sensitive to outliers.
- **Soft margin SVM**: Allows margin violations (controlled by hyperparameter `C`). A smaller `C` gives a wider margin but more violations; a larger `C` penalizes violations more heavily.

## Example

### Linear SVM Classification (Iris Dataset)
```python
import numpy as np
from sklearn import datasets
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

iris = datasets.load_iris()
X = iris["data"][:, (2, 3)]  # petal length, petal width
y = (iris["target"] == 2).astype(np.float64)  # Iris-Virginica

svm_clf = Pipeline([
    ("scaler", StandardScaler()),
    ("linear_svc", LinearSVC(C=1, loss="hinge")),
])
svm_clf.fit(X, y)
svm_clf.predict([[5.5, 1.7]])  # array([1.])
```

### Polynomial Kernel SVM (Moons Dataset)
```python
from sklearn.datasets import make_moons
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures

X, y = make_moons()
polynomial_svm_clf = Pipeline([
    ("poly_features", PolynomialFeatures(degree=3)),
    ("scaler", StandardScaler()),
    ("svm_clf", LinearSVC(C=10, loss="hinge"))
])
polynomial_svm_clf.fit(X, y)
```

### Gaussian RBF Kernel SVM
```python
from sklearn.svm import SVC

rbf_kernel_svm_clf = Pipeline([
    ("scaler", StandardScaler()),
    ("svm_clf", SVC(kernel="rbf", gamma=5, C=0.001))
])
rbf_kernel_svm_clf.fit(X, y)
```

## Explanation
The fundamental idea is **large margin classification** — fitting the widest possible "street" between classes. The decision boundary is determined only by the **support vectors** (the instances on the edge of the street). SVMs are sensitive to feature scales, so feature scaling (e.g., `StandardScaler`) is critical.

When data is not linearly separable, SVMs use the **kernel trick** to implicitly map data into higher-dimensional space without the computational cost of actually adding features. Common kernels:
- **Linear kernel**: Fast, works well for high-dimensional or large datasets.
- **Polynomial kernel**: Adds polynomial features; controlled by `degree` and `coef0`.
- **Gaussian RBF kernel**: Uses similarity features based on distance to landmarks; controlled by `gamma` (`γ`). A small `γ` produces a smoother boundary, a large `γ` produces a wiggly boundary.

The trade-off between margin width and margin violations is controlled by the hyperparameter `C`. Reducing `C` increases regularization (wider margin, more violations).

## Features & Hyperparameters

| Hyperparameter | Meaning | Effect |
|---|---|---|
| `C` | Regularization strength (inverse). Smaller `C` → wider margin, more violations. Larger `C` → narrower margin, fewer violations. | Controls overfitting vs underfitting. Reduce `C` if overfitting. |
| **kernel** | Type of kernel: `"linear"`, `"poly"`, `"rbf"`, `"sigmoid"`, or custom. | Determines how data is transformed. RBF works well in most cases. |
| `gamma` (γ) | Kernel coefficient for RBF, poly, and sigmoid kernels. | Small γ → smoother boundary (high bias). Large γ → more complex boundary (high variance). |
| **degree** | Degree of the polynomial kernel function. | Higher degree → more complex model. Reduce if overfitting. |
| **coef0** | Independent term in poly/sigmoid kernel. | Controls influence of high-degree vs low-degree polynomials. |
| `epsilon` (ε) | Margin width in SVM Regression (SVR). | Larger ε → wider street, more instances inside margin. |
| **tol** | Tolerance for stopping criterion. | Higher tolerance → faster training but less precise. |
| **loss** | Loss function. For SVM: `"hinge"` (standard). | Hinge loss is the standard SVM loss function. |
| **dual** | Solve dual or primal problem. Set `False` when `n_samples > n_features`. | Primal is faster for large datasets; dual enables kernel trick. |

## Applications
- **Text classification** (e.g., spam detection, document categorization) — SVMs work well with high-dimensional sparse data.
- **Image classification** (e.g., handwritten digit recognition, face detection).
- **Bioinformatics** (e.g., protein classification, cancer classification from gene expression data).
- **Handwriting recognition** — SVMs are used in OCR systems.
- **Outlier detection** — One-class SVM for anomaly detection.
- **Financial forecasting** — Stock market prediction, credit risk assessment.
- **Drug discovery** — Classifying molecules and compounds.
- **Speech recognition** — SVMs classify phonemes and spoken words.

