# Traning models 
- Inside this post, we will discuss about some trainings models: 
    - Linear Regression 
    - Decision Tree Regressor 
    - Random Forest Regressor 


## Linear Regression 
- It's a foundational machine learning algorithm used to model the relations between a dependenct variable (the target) and one or more independent variables (features) by generating a linear equation to the observed data 

### Key Features : 
- Line of best fit: This model calculates a straight line that minimizes the distance from data points to line 
- The Equation : 
    $$y = \beta_0 + \beta_1x_1 + \beta_2x_2 + ... + \beta_nx_n + \varepsilon$$
    - Where $y$ is the prediction, $\beta_0$ is the intercept, $\beta_{i}$ is the coefficient, and the $\varepsilon$ is the error term


### Advantages & Disadvantages
- Advantages : 
    - Incredibly simple to implement and understand 
    - Highly interpretable (each coeffiecient tell the change of target's feature)
    - Computationally efficient and fast to train 
- Disadvantags : 
    - Struggles with complex, non-linear relationships 
    - Highly affected by the outliers 
    

### Best Use For
- Prediciting continuous values where the trend is directional (eg: predicting housing price based on square footage, ...) 
- Scenarios where you need to explain why a prediction was made 


## Decision Tree Regressor 
- It's a non-parametic model that breaks down the dataset into smaller subsets, while simultaneously developing the associated decision tree. The final result is a tree with decision nodes and leaf nodes 

### Key Features: 
- If-Then Spilitting: The splitting based on feature values  
- Non-linearity: It captures non-linearity data without needing complex data tranformations
- Leaf node output: The predicted value for a test point is mean average of the training targets in the leaf node it fall into 

### Advantages & Disadvantages
- Advantages: 
    - Easy to visualize and interpret 
    - Requires minimal data preprocessing 
    - Handles both numerical and categorical data  
- Disadvantages: 
    - High variance: 
    - Small change can make big change in tree structure

### Best Used For
- Datasets with complex, non-linear interactions where strict mathematical assumptions don't hold.
- Rule-based decision-making environments (e.g., segmenting customers into specific risk brackets for insurance premiums).


## Random Forest Regressor 
- It's a learning method that solves the overliftting problem of decision tree. It works by constructing many decision trees at training time and outputting the average prediction of individual trees 

### Key Features : 
- Bagging : It train each tree with a random sample of the data 
- Feature of Randomness : When splitting a node, it only considers a random subset of features, ensuring the trees are decorrelated and diverse
- Averaging : It calculates the outputs of all individual trees to product the final, stable prediction 

### Advantages & Disadvantages
- Advantages: 
    - Highly accurate and robust against overfitting compared to a single decision tree
    - Handles large datasets with higher dimensionality exceptionally well
    - Provides built-in "Feature Importance" metrics to show which variables matter most
- Disadvantage: 
    - Black Box: It loses the easy visual interpretability of a single decision tree
    - Computationally heavy and slower to predict, as it has to run through hundreds of trees
    - Can be memory-intensive

### Best Used For
- High-stakes predictions where accuracy is a priority over raw speed


## Cross-Validation (CV) (Evalation Method)
- A resampling technique used to evaluate ML models by training on **k-1 folds** of the data and validating on the remaining **1 fold**, repeating this process **k times** so every sample is used for validation exactly once

### Key Features
- **k-fold CV**: Data split into *k* equal folds; performance metric is averaged across all *k* trials
- **Stratified k-fold**: Preserves the proportion of classes/categories in each fold (used in this project for income-based stratified splitting)
- **Leave-One-Out (LOO)**: k = number of samples — trains on all but one data point (very expensive)
- Returns both **mean** and **standard deviation** of the evaluation metric (e.g., RMSE), giving a more complete picture of model stability

### Advantages & Disadvantages
- Advantages:
    - **More reliable** than a single train-test split — uses the entire dataset for both training and validation
    - **Reduces variance** of the performance estimate, making model comparisons more trustworthy
    - **Detects overfitting** — a large gap between training score and CV score signals that the model memorized rather than learned (e.g., Decision Tree: RMSE ≈ 0 on training vs ~71,000 on CV)
    - Works well with small datasets where a single held-out test set would waste too much data
- Disadvantages:
    - **Computationally expensive** — trains *k* models instead of one (e.g., 10x slower than a single split)
    - **Not suitable for time-series data** without specialized variants (use `TimeSeriesSplit` instead)
    - Does not replace a **final held-out test set** — if you tune hyperparameters using CV, you still need an untouched test set to get an unbiased final evaluation

### Best Used For
- **Model selection** — comparing algorithms like Linear Regression vs Decision Tree vs Random Forest
- **Hyperparameter tuning** — `GridSearchCV` and `RandomizedSearchCV` use CV internally to find optimal parameters
- **Small to medium datasets** (<100k samples) where the extra compute is affordable
- **Detecting overfitting** — in this project, CV revealed that the Decision Tree (training RMSE ≈ $0) severely overfits, while Random Forest's CV RMSE (~$44,000) confirmed it generalizes far better


## Note about **Overfitting**
- It occurs when a machine learning model learning a training data to well -- capturing not just basic patterns, but also the random noise, outliers
- An overliftted model has low bias, but high variance -> It generates poorly with unseen information  