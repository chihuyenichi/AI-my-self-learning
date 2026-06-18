# End-to-End Machine Learning Project - California Housing Price Prediction

## Project Summary

This project implements a complete end-to-end machine learning pipeline to predict median house values in California districts using the California Housing dataset. It covers the full ML workflow: data loading, exploration, preprocessing, model training, hyperparameter tuning, evaluation, and deployment. The dataset contains 20,640 records with 10 features including geographic location, housing age, room counts, population, income, and ocean proximity.

Source: Chapter 2 of *Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow* (3rd ed.) by Aurélien Géron.

---

## Project Plan & Step-by-Step Breakdown

### Step 1: Data Loading
- Downloads `housing.tgz` from GitHub if not already cached locally
- Extracts and loads `housing.csv` into a Pandas DataFrame
- Features: `longitude`, `latitude`, `housing_median_age`, `total_rooms`, `total_bedrooms`, `population`, `households`, `median_income`, `median_house_value` (target), `ocean_proximity` (categorical)

### Step 2: Exploratory Data Analysis (EDA)
- **`head()`**: Preview first 5 rows to inspect structure and sample values
- **`info()`**: Check data types, non-null counts, memory usage — reveals 207 missing values in `total_bedrooms`
- **`value_counts()`**: Shows distribution of `ocean_proximity` (5 categories)
- **`describe()`**: Summary statistics (mean, std, min/max, quartiles) for numerical features
- **Histograms**: Visualize distributions of all numerical columns (right-skewed income, capped house values at $500k)
- **Geographic scatter plots**: Plot districts on a map colored by house value, sized by population — reveals price clustering near the coast
- **Correlation analysis**: `corr()` + scatter matrix — `median_income` is the strongest predictor of house value
- **Feature engineering**: Creates `rooms_per_house`, `bedrooms_ratio`, `people_per_house` — `bedrooms_ratio` has a moderate negative correlation with price

### Step 3: Train-Test Splitting
Four approaches demonstrated:

**a) Simple Random Split** (`shuffle_and_split_data`)
- Shuffles indices via `np.random.permutation`, splits at 20% threshold
- Needs `np.random.seed()` for reproducibility

**b) Hash-Based Split** (`split_data_with_id_hash`)
- Uses `zlib.crc32()` to hash each row's identifier
- Test membership if `crc32(id) < test_ratio * 2^32`
- **Advantage**: stable split even when dataset grows — new rows don't reshuffle existing assignments

**c) Scikit-Learn's `train_test_split`**
- Simple one-liner with `random_state=42`

**d) Stratified Split** (`StratifiedShuffleSplit`)
- Creates `income_cat` by binning `median_income` into 5 categories (bins: 0-1.5, 1.5-3, 3-4.5, 4.5-6, 6-inf)
- Uses `stratify=housing["income_cat"]` to preserve income distribution in both splits
- Stratified error: <0.4% vs random error: up to ~6.5%

### Step 4: Data Preparation & Cleaning

**Handling Missing Values (total_bedrooms):**
- Option 1: `dropna()` — drops entire rows, losing other columns' data
- Option 2: `drop()` column — loses the feature entirely
- Option 3 (chosen): fill with median via `SimpleImputer(strategy="median")`

**Outlier Detection (bonus):**
- Uses `IsolationForest` to identify outliers but does not remove them

**Categorical Encoding (ocean_proximity):**
- `OrdinalEncoder`: maps categories to integers (not ideal for this feature)
- `OneHotEncoder` (chosen): creates 5 binary columns, handles unknown categories with `handle_unknown="ignore"`

**Feature Scaling:**
- `MinMaxScaler`: scales to a fixed range (e.g., -1 to 1)
- `StandardScaler` (chosen): standardizes to zero mean and unit variance

**Custom Transformers:**
- `FunctionTransformer`: applies log transforms, RBF kernel similarity (age similarity to 35, geographic similarity to San Francisco), column ratios
- `ClusterSimilarity`: uses `KMeans` to create cluster-based similarity features for geographic location (10 clusters, RBF gamma similarity)

**Transformation Pipelines:**
- `Pipeline`: chains imputation → scaling for numerical features
- `ColumnTransformer`: applies different pipelines to numerical vs categorical columns
- `make_column_transformer` / `make_column_selector`: concise API for column-based preprocessing

### Step 5: Model Training & Evaluation

**Purpose:** Train regression models to predict `median_house_value`, evaluate their performance, and select the best model.

**How it works:**

**5a. Train Baseline Models:**
- **Linear Regression**: Simple baseline, not powerful enough for this data
- **Decision Tree Regressor**: Fits training data perfectly (RMSE ≈ 0), but cross-validation reveals poor generalization (CV RMSE ≈ 71,000)
- **Random Forest Regressor**: Ensemble of 100 decision trees, significantly better

**5b. Cross-Validation:**
- Uses `cross_val_score` with `scoring="neg_root_mean_squared_error"` and `cv=10`
- Reports mean and std of RMSE across 10 folds
- Compares training RMSE vs validation RMSE to detect overfitting

### Step 6: Fine-Tuning

**Purpose:** Optimize hyperparameters to improve model performance.

**How it works:**

**6a. Grid Search (`GridSearchCV`):**
- Exhaustively tries all combinations in `param_grid`
- For Random Forest: `n_estimators` (100, 200, 300), `max_features` (4, 6, 8), `min_samples_split` (10, 30, 50)
- Finds best params: `n_estimators=300`, `max_features=6`, `min_samples_split=30`

**6b. Randomized Search (`RandomizedSearchCV`):**
- Samples hyperparameters from distributions (`randint`, `loguniform`)
- More efficient for high-dimensional spaces; finds near-optimal combos faster
- Tunes: `n_clusters` (3-50), `n_estimators`, `max_features`, `min_samples_split`

**6c. Halving Random Search (`HalvingRandomSearchCV`):**
- Uses successive halving: allocates more resources to promising candidates

**Feature Importance Analysis:**
- Extracts `feature_importances_` from the trained Random Forest
- Top features: `median_income`, `bedrooms_ratio`, `longitude`, `latitude`

### Step 7: Final Evaluation on Test Set

**Purpose:** Get an unbiased estimate of the model's real-world performance.

**How it works:**
- Applies the best model (from randomized search) to the held-out `strat_test_set`
- Computes RMSE on test predictions
- Calculates 95% confidence interval using bootstrap (`scipy.stats.bootstrap`)
- Confidence interval: e.g., (43,280 — 47,460) — the true RMSE is 95% likely to fall in this range

### Step 8: Model Persistence & Deployment

**Purpose:** Save the trained model for production use.

**How it works:**
- Saves model with `joblib.dump(final_model, "my_california_housing_model.pkl")`
- Loads with `joblib.load()` in production script
- Applies same preprocessing pipeline to new data, calls `predict()`

---

## Key Insights

1. **Stratified sampling is critical** — Naive random splitting introduces bias (up to ~6.5% error vs <0.4% for stratified) in preserving income distribution across train/test sets.

2. **Hash-based splitting ensures stability** — CRC32-based splits guarantee consistent test/train assignment even when new data is added, preventing data leakage.

3. **Median imputation with `SimpleImputer` is the preferred missing-data strategy** — Preserves all 20,640 records and fits naturally into a scikit-learn `Pipeline`.

4. **Feature engineering matters** — Derived features like `bedrooms_ratio` (total_bedrooms / total_rooms) ranked as the 2nd most important feature, ahead of raw `total_rooms`.

5. **Geographic clustering boosts performance** — The `ClusterSimilarity` transformer (10 KMeans clusters + RBF kernel) captures spatial price trends (coastal vs inland) and is more effective than raw lat/lon coordinates.

6. **Random Forest beats simpler models** — Linear Regression and Decision Trees underperform; Random Forest achieves the best RMSE (~44,000–47,000). The training RMSE (~16,000) vs CV RMSE (~44,000) indicates overfitting, mitigated by tuning.

7. **Target variable is capped at $500,001** — This creates a ceiling effect; the model underestimates high-end properties. Possible fixes: remove capped districts or treat as a censored regression problem.

8. **Pipeline composability** — The scikit-learn `Pipeline` + `ColumnTransformer` API makes preprocessing reproducible and deployable. The entire workflow (impute → scale → encode → cluster → predict) is encapsulated in a single object.

9. **Bootstrap confidence intervals** — Using `scipy.stats.bootstrap` gives a robust interval estimate for the test RMSE, more informative than a single point estimate.

10. **Auto-ML exploration** — Exercises show that SVR with randomized search can be competitive, and feature selection (`SelectFromModel`) does not always improve performance.
