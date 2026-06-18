# End-to-End Machine Learning Project

A complete end-to-end machine learning pipeline to predict median house values in California districts, based on **Chapter 2** of *Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow* (3rd ed.) by Aurélien Géron.

## Project Overview

This project walks through the full ML workflow:
- **Data Loading** — Downloading and extracting the California Housing dataset
- **Exploratory Data Analysis** — Visualizing geographic patterns, correlations, and feature distributions
- **Data Preparation** — Handling missing values, encoding categorical features, feature scaling, and custom transformers (geographic clustering)
- **Model Training** — Linear Regression, Decision Tree, and Random Forest
- **Hyperparameter Tuning** — Grid search, randomized search, and halving search
- **Evaluation** — Cross-validation, test set performance, bootstrap confidence intervals
- **Deployment** — Model persistence with joblib and a production prediction script

## Dataset

The California Housing dataset contains **20,640 records** with **10 features**:
- `longitude`, `latitude` — geographic coordinates
- `housing_median_age` — median age of houses in the district
- `total_rooms`, `total_bedrooms` — room counts
- `population`, `households` — population and household counts
- `median_income` — median income of residents
- `median_house_value` — **target variable** (median house price)
- `ocean_proximity` — categorical (e.g., NEAR BAY, INLAND, <1H OCEAN)

## Setup

```bash
# Clone the repo
git clone https://github.com/chihuyenichi/End-to-end-project-Based-on-hand-on-ml-book.git
cd End-to-end-project-Based-on-hand-on-ml-book

# Create and activate virtual environment
python3 -m venv env
source env/bin/activate

# Install dependencies
pip install -r requirements.txt   # if available
# Or manually:
pip install pandas numpy matplotlib scikit-learn scipy joblib
```

## Usage

Open the notebook:

```bash
jupyter notebook end-to-end-project.ipynb
```

Or run in Google Colab: [Open In Colab](https://colab.research.google.com/github/ageron/handson-ml3/blob/main/02_end_to_end_machine_learning_project.ipynb)

## Key Results

| Model | Training RMSE | CV RMSE (10-fold) |
|---|---|---|
| Linear Regression | ~68,000 | ~70,000 |
| Decision Tree | ~0 (overfit) | ~71,000 |
| Random Forest (tuned) | ~16,000 | ~44,000 |

**Best model:** Random Forest with `n_estimators=300`, `max_features=6`, `min_samples_split=30`

**Test set RMSE:** ~44,900 (95% CI: [43,280 — 47,460])

## Resources

- [Book website](https://homl.info)
- [Original notebook on GitHub](https://github.com/ageron/handson-ml3/blob/main/02_end_to_end_machine_learning_project.ipynb)
