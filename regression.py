import sys
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.linear_model import Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils import load_data

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return mse, r2

def main():
    print("\n Loading dataset...")
    df = load_data()
    X = df.drop("MEDV", axis=1)
    y = df["MEDV"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("\n Starting hyperparameter tuning for 3 models...\n")

    models_and_params = {
        "Ridge": {
            "model": Ridge(),
            "params": {
                "alpha": [0.01, 0.1, 1.0, 10.0],
                "fit_intercept": [True, False],
                "solver": ['auto', 'svd', 'cholesky']
            }
        },
        "DecisionTree": {
            "model": DecisionTreeRegressor(random_state=42),
            "params": {
                "max_depth": [3, 5, 10, None],
                "min_samples_split": [2, 5, 10],
                "min_samples_leaf": [1, 2, 4]
            }
        },
        "RandomForest": {
            "model": RandomForestRegressor(random_state=42),
            "params": {
                "n_estimators": [50, 100],
                "max_depth": [None, 10, 20],
                "min_samples_split": [2, 5],
                "min_samples_leaf": [1, 2]
            }
        }
    }

    for name, obj in models_and_params.items():
        print(f"\n Tuning: {name}")
        grid = GridSearchCV(estimator=obj["model"], param_grid=obj["params"], cv=5, n_jobs=-1, scoring='r2')
        grid.fit(X_train, y_train)
        best_model = grid.best_estimator_
        mse, r2 = evaluate_model(best_model, X_test, y_test)

        print(f" Best Parameters: {grid.best_params_}")
        print(f" MSE: {mse:.2f}")
        print(f" R²: {r2:.2f}")

if __name__ == "__main__":
    main()
