# Chapter 3: Classification — Summary

## 1. MNIST

**Definition:** MNIST is a dataset of 70,000 handwritten digit images (28×28 pixels, 784 features per image), often called the "Hello World" of Machine Learning. It serves as a benchmark for classification algorithms.

**Example:** Each image is labeled 0–9. Loading it with Scikit-Learn: `mnist = fetch_openml('mnist_784', version=1)`. The first image is a "5". The dataset is pre-split: 60,000 training, 10,000 test.

**Explanation:** MNIST is small enough to experiment quickly but complex enough to be meaningful. Each pixel's intensity (0–255) is a feature, making it a 784-dimensional problem. It is already shuffled to avoid ordering bias.

---

## 2. Training a Binary Classifier

**Definition:** A binary classifier distinguishes between exactly two classes. Here we build a "5-detector" that predicts whether an image is a 5 or not-5.

**Example:** `y_train_5 = (y_train == 5)` creates boolean labels. `SGDClassifier` is trained on these labels. Predicting on the first image returns `True` (correct).

**Explanation:** SGD (Stochastic Gradient Descent) handles large datasets efficiently by processing one instance at a time. It is well-suited for online learning but relies on randomness, so `random_state` should be set for reproducibility.

---

## 3. Performance Measures

### 3.1 Measuring Accuracy Using Cross-Validation

**Definition:** K-fold cross-validation splits the training set into K folds, trains on K−1 folds and evaluates on the remaining fold, repeating K times. Stratified folds preserve class proportions.

**Example:** `cross_val_score(sgd_clf, X_train, y_train_5, cv=3, scoring="accuracy")` yields ~93% accuracy. However, a "never-5" classifier that always predicts False still gets ~90% due to class imbalance (only 10% are 5s).

**Explanation:** Accuracy alone is misleading for skewed datasets. When one class dominates, a dumb classifier can appear accurate. Always check class distribution before trusting accuracy.

### 3.2 Confusion Matrix

**Definition:** A table where rows represent actual classes and columns represent predicted classes. It shows true negatives (TN), false positives (FP), false negatives (FN), and true positives (TP).

**Example:**
```
[[53057  1522]
 [ 1325  4096]]
```
- TN = 53,057 (correct non-5s)
- FP = 1,522 (non-5s wrongly labeled as 5)
- FN = 1,325 (5s missed)
- TP = 4,096 (correct 5s)

**Explanation:** The confusion matrix gives a full picture of where the classifier succeeds and fails. A perfect classifier has non-zero valu## Core Structure of a Confusion Matrix

A Confusion Matrix is a $2 \times 2$ grid (for binary classification) that cross-references actual ground-truth values against the model's predicted values.

| | Predicted: **Negative (0)** | Predicted: **Positive (1)** |
| :--- | :--- | :--- |
| **Actual: Negative (0)** | **True Negative (TN)**<br>• Model correctly guessed the negative outcome.<br>• *Example: Healthy person tested healthy.* | **False Positive (FP)**<br>• Model flagged an error (Type I Error).<br>• *Example: Healthy person tested positive.* |
| **Actual: Positive (1)** | **False Negative (FN)**<br>• Model missed the target (Type II Error).<br>• *Example: Sick person tested negative.* | **True Positive (TP)**<br>• Model correctly guessed the positive outcome.<br>• *Example: Sick person tested positive.* |

---

### Key Feature Metrics Derived from the Matrix

From these four core blocks, you can calculate the primary evaluation metrics used to judge a machine learning model:

#### 1. Accuracy
The overall percentage of correct predictions out of all total cases.
* **Formula:** $$\text{Accuracy} = \frac{\text{TP} + \text{TN}}{\text{TP} + \text{TN} + \text{FP} + \text{FN}}$$

#### 2. Precision (Positive Predictive Value)
Measures exactness. Out of everything the model *claimed* was positive, how much was actually right?
* **Formula:** $$\text{Precision} = \frac{\text{TP}}{\text{TP} + \text{FP}}$$

#### 3. Recall (Sensitivity / True Positive Rate)
Measures completeness. Out of all the *real* positive instances in the dataset, how many did the model manage to capture?
* **Formula:** $$\text{Recall} = \frac{\text{TP}}{\text{TP} + \text{FN}}$$

#### 4. F1-Score
The harmonic mean of Precision and Recall. It gives you a balanced metric if your data has an uneven class distribution (imbalanced dataset).
* **Formula:** $$\text{F1-Score} = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$$es only on the main diagonal.



### 3.3 Precision and Recall

**Definition:**
- **Precision** = TP / (TP + FP) — how many positive predictions are actually correct.
- **Recall** = TP / (TP + FN) — how many actual positives were correctly found.
- **F1 score** = harmonic mean of precision and recall, balancing both.

**Example:** `precision_score` ≈ 72.9%, `recall_score` ≈ 75.6%, `f1_score` ≈ 74.2%.

**Explanation:** High precision means few false positives; high recall means few false negatives. The F1 score is useful when comparing classifiers but favors balanced precision-recall. The choice depends on the problem (e.g., safe-kid videos → high precision; shoplifter detection → high recall).

### 3.4 Precision/Recall Tradeoff

**Definition:** Increasing the decision threshold increases precision but decreases recall, and vice versa. The threshold determines above which score an instance is classified as positive.

![alt](/image-materials/decision-threshold.png)
**Example:** With threshold = 0, the SGDClassifier returns `True` for a 5. Raising threshold to 8,000 misses that same 5. Choosing a threshold of ~7,816 achieves 90% precision but recall drops to ~43.7%.

**Explanation:** There is no free lunch — you must choose the tradeoff that suits your application. The `precision_recall_curve()` function helps visualize this tradeoff across all thresholds.

### 3.5 The ROC Curve

**3.5.1. Purpose/Definition** 
- The ROC curve a graph that shows how well a classification model can tell the difference of two things (like "Spam" vs "Not Spam"). It calculates the a probability score.
- To make a decision, you have to set a threshold line (like 50%). Anything below 50% is "Not Spam" and anything above 50% is "Spam"
-> The ROC curve plots the model's performance across every possible threshold, then pick the most balance for our target  

**3.5.2. Features and Meanings** 
- The ROC graph with X-axis and Y-axis: 
    - The Y-axis: True Positive Rate 
        - Recall / Sensitivity 
        - It's out of all actual positive cases. We want the percentage to be close to 100% 
    - The X-axis: False Positive Rate 
        - Value: `1 - specificity`; $\displaystyle{specificity = \frac{\text{True Negative}}{\text{True Negative} + \text{False Positive}}}$
        - Out of all the actual negative case, what percentage does the model recognize as positive ? We need this score to be close to 0%
    - Diagonal Base Line : It's straight line from (0, 0) -> (1, 1)
    - AUC (Area Under the Curve) :  
        - It's a single score between 0.0 and 1.0 that measures the entire space underneath the model's curved line 
        - It summarizes how good the model is overall 

- How the Graph Helps You: 
    - The ROC curve maps out these exact points on a chart. As a human manager, you look at the curve and pick the exact spot on the line that matches your goal. In an airport, you would accept a high False Positive Rate (more bag checks) to ensure your True Positive Rate stays near 100% (no weapons slip past).

**Example:** SGDClassifier ROC AUC ≈ 0.961. RandomForestClassifier ROC AUC ≈ 0.998, significantly better. The PR curve is preferred when the positive class is rare; ROC is preferred otherwise.

**Explanation:** ROC curves are useful for comparing classifier performance regardless of class distribution. However, when positives are rare, the ROC curve can look overly optimistic — use the PR curve instead.

---

## 4. Multiclass Classification

**Definition:** Classifying instances into more than two classes. Strategies include:
- **One-versus-All (OvA):** Train N binary classifiers (one per class), pick the highest score.
- **One-versus-One (OvO):** Train N×(N−1)/2 binary classifiers (every pair), pick the class that wins most duels.

**Example:** `SGDClassifier` on MNIST with 10 classes automatically uses OvA. `OneVsOneClassifier(SGDClassifier())` explicitly uses OvO (45 classifiers). `RandomForestClassifier` handles multiclass natively without OvA/OvO.

**Explanation:** Scikit-Learn auto-selects OvA for most binary classifiers and OvO for SVMs. OvO is faster for algorithms that scale poorly with dataset size. Feature scaling (StandardScaler) raised accuracy from ~84% to ~89%.

---

## 5. Error Analysis

**Definition:** Systematically examining the types and patterns of errors a classifier makes, typically using the confusion matrix and inspecting misclassified instances.

**Example:** The confusion matrix shows many images are misclassified as "8". Normalizing by row (actual class counts) reveals that 3s and 5s are frequently confused. Plotting misclassified 3s and 5s shows that some are genuinely ambiguous, but many are obvious errors caused by the linear model's sensitivity to pixel shifts.

**Explanation:** Error analysis guides improvement strategies — gather more data for confusing classes, engineer new features (e.g., count closed loops), or preprocess images (centering, rotation correction). Understanding that SGD is a linear model explains why it confuses similar digits.

---

## 6. Multilabel Classification

**Definition:** A classification system that outputs multiple binary labels per instance (e.g., tagging multiple people in a photo).

**Example:** For digit "5", predict two labels: "is it large (≥7)? False" and "is it odd? True" → output `[False, True]`. `KNeighborsClassifier` supports this natively.

**Explanation:** Unlike binary or multiclass (single label per instance), multilabel systems can assign any number of labels. Evaluation often uses averaged F1 scores across labels, with options like `"macro"` (unweighted) or `"weighted"` (by label support).

---

## 7. Multioutput Classification

**Definition:** A generalization of multilabel classification where each label can have multiple possible values (multiclass). The output is a vector, and each position can take more than two values.

**Example:** Removing noise from digit images — the input is a noisy image, and the output is a clean image. Each of the 784 pixels is a label with 256 possible values (0–255). `KNeighborsClassifier` is trained on `X_train_mod` (noisy) vs `y_train_mod` (clean original).

**Explanation:** Multioutput systems blur the line between classification and regression (predicting pixel intensity is regression-like). They are the most general classification framework, capable of handling complex output structures.

---

## Relationships Between Topics

The topics form a **progressive hierarchy** from simplest to most general:

1. **Binary Classification** → **Multiclass Classification**: Binary handles 2 classes. Multiclass extends to N classes via OvA or OvO strategies. The same binary classifier can be wrapped to handle N classes.

2. **Multiclass Classification** → **Multilabel Classification**: Multiclass assigns exactly 1 of N labels. Multilabel allows any subset of N binary labels, useful when multiple attributes are simultaneously relevant.

3. **Multilabel Classification** → **Multioutput Classification**: Multilabel restricts each output to binary (0/1). Multioutput generalizes each label to have multiple possible values (e.g., 0–255), making it the most general form.

4. **Performance Measures** (Confusion Matrix, Precision, Recall, ROC) apply to binary tasks directly and extend to multiclass/multilabel via averaging strategies (macro, weighted).

5. **Precision/Recall Tradeoff** and **ROC Curves** are tools for selecting the optimal decision threshold, which connects directly to every classification task.

6. **Error Analysis** feeds back into all topics — once a model is built (binary, multiclass, multilabel, or multioutput), confusion matrices and error inspection guide iterative improvement.

In short: the chapter starts with the simplest case (binary MNIST 5-detector), builds evaluation tools, then generalizes step by step to the most flexible framework (multioutput classification), with error analysis as a universal improvement loop.
