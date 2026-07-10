"""
train.py
--------
Reproduces the ASD screening pipeline from the notebook end-to-end:

  1. Load data/train.csv
  2. Clean (drop ID/age_desc, fix country names, fill missing categories)
  3. Label-encode categorical columns (encoders saved for inference)
  4. Replace outliers in age/result with the median
  5. Train/test split + SMOTE oversampling on the training set
  6. Cross-validate Decision Tree / Random Forest / XGBoost
  7. Hyperparameter-tune all three with RandomizedSearchCV
  8. Keep whichever has the best CV accuracy, evaluate on the held-out test set
  9. Save artifacts/best_model.pkl, artifacts/encoders.pkl, artifacts/feature_columns.pkl

Usage:
    python train.py
Requires:
    data/train.csv  (the standard ASD screening dataset -
                      https://www.kaggle.com/datasets/shivamshinde123/autismprediction)
"""

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "train.csv"
ARTIFACTS_DIR = BASE_DIR / "artifacts"
ARTIFACTS_DIR.mkdir(exist_ok=True)

RANDOM_STATE = 42


def load_and_clean(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["age"] = df["age"].astype(int)

    # drop non-predictive columns
    df = df.drop(columns=["ID", "age_desc"])

    # normalize country names
    country_mapping = {
        "Viet Nam": "Vietnam",
        "AmericanSamoa": "United States",
        "Hong Kong": "China",
    }
    df["contry_of_res"] = df["contry_of_res"].replace(country_mapping)

    # fill missing categorical values
    df["ethnicity"] = df["ethnicity"].replace({"?": "Others", "others": "Others"})
    df["relation"] = df["relation"].replace(
        {"?": "Others", "Relative": "Others", "Parent": "Others", "Health care professional": "Others"}
    )
    return df


def replace_outliers_with_median(df: pd.DataFrame, column: str) -> pd.DataFrame:
    q1, q3 = df[column].quantile(0.25), df[column].quantile(0.75)
    iqr = q3 - q1
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    median = df[column].median()
    df[column] = df[column].apply(lambda x: median if x < lower or x > upper else x)
    return df


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Expected training data at {DATA_PATH}. "
            "Download the ASD screening dataset and place it there as 'train.csv'."
        )

    print(f"Loading data from {DATA_PATH} ...")
    df = load_and_clean(DATA_PATH)

    # label encode categorical columns, saving encoders for inference
    object_columns = df.select_dtypes(include=["object"]).columns
    encoders = {}
    for column in object_columns:
        encoder = LabelEncoder()
        df[column] = encoder.fit_transform(df[column])
        encoders[column] = encoder
    with open(ARTIFACTS_DIR / "encoders.pkl", "wb") as f:
        pickle.dump(encoders, f)
    print(f"Saved encoders for columns: {list(object_columns)}")

    # outlier handling
    df = replace_outliers_with_median(df, "age")
    df = replace_outliers_with_median(df, "result")

    # split
    X = df.drop(columns=["Class/ASD"])
    y = df["Class/ASD"]
    feature_columns = list(X.columns)
    with open(ARTIFACTS_DIR / "feature_columns.pkl", "wb") as f:
        pickle.dump(feature_columns, f)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    # SMOTE on training set only
    smote = SMOTE(random_state=RANDOM_STATE)
    X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
    print(f"After SMOTE: {y_train_smote.value_counts().to_dict()}")

    # hyperparameter grids
    param_grid_dt = {
        "criterion": ["gini", "entropy"],
        "max_depth": [None, 10, 20, 30, 50, 70],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
    }
    param_grid_rf = {
        "n_estimators": [50, 100, 200, 500],
        "max_depth": [None, 10, 20, 30],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "bootstrap": [True, False],
    }
    param_grid_xgb = {
        "n_estimators": [50, 100, 200, 500],
        "max_depth": [3, 5, 7, 10],
        "learning_rate": [0.01, 0.05, 0.1, 0.2],
        "subsample": [0.6, 0.8, 1.0],
    }

    searches = {
        "Decision Tree": RandomizedSearchCV(
            DecisionTreeClassifier(random_state=RANDOM_STATE), param_grid_dt,
            n_iter=20, cv=5, scoring="accuracy", random_state=RANDOM_STATE, n_jobs=-1,
        ),
        "Random Forest": RandomizedSearchCV(
            RandomForestClassifier(random_state=RANDOM_STATE), param_grid_rf,
            n_iter=20, cv=5, scoring="accuracy", random_state=RANDOM_STATE, n_jobs=-1,
        ),
        "XGBoost": RandomizedSearchCV(
            XGBClassifier(random_state=RANDOM_STATE, eval_metric="logloss"), param_grid_xgb,
            n_iter=20, cv=5, scoring="accuracy", random_state=RANDOM_STATE, n_jobs=-1,
        ),
    }

    best_model, best_model_name, best_score = None, None, 0.0
    for name, search in searches.items():
        print(f"Tuning {name} ...")
        search.fit(X_train_smote, y_train_smote)
        print(f"  {name} best CV accuracy: {search.best_score_:.4f}")
        if search.best_score_ > best_score:
            best_model, best_model_name, best_score = search.best_estimator_, name, search.best_score_

    print(f"\nBest model: {best_model_name} (CV accuracy {best_score:.4f})")

    # final evaluation on held-out test set
    y_pred = best_model.predict(X_test)
    print("\nTest Accuracy:", accuracy_score(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))

    with open(ARTIFACTS_DIR / "best_model.pkl", "wb") as f:
        pickle.dump(best_model, f)
    print(f"\nSaved model to {ARTIFACTS_DIR / 'best_model.pkl'}")
    print(f"Saved encoders to {ARTIFACTS_DIR / 'encoders.pkl'}")
    print(f"Saved feature column order to {ARTIFACTS_DIR / 'feature_columns.pkl'}")


if __name__ == "__main__":
    main()
