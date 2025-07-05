import sys
import os
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))

from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from utils import load_data

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return mse, r2

def main():
    df = load_data()
    X = df.drop("MEDV", axis=1)
    y = df["MEDV"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    param_grids = {
        "LinearRegression": {},  # No hyperparams for basic LinearRegression
        "DecisionTree": {
            "max_depth": [3, 5, 10, None],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4]
        },
        "RandomForest": {
            "n_estimators": [50, 100],
            "max_depth": [None, 10, 20],
            "min_samples_split": [2, 5]
        }
    }

    models = {
        "LinearRegression": LinearRegression(),
        "DecisionTree": DecisionTreeRegressor(random_state=42),
        "RandomForest": RandomForestRegressor(random_state=42)
    }

    for name, model in models.items():
        print(f"\n--- {name} ---")
        if param_grids[name]:
            grid = GridSearchCV(model, param_grids[name], cv=5, n_jobs=-1, scoring='r2')
            grid.fit(X_train, y_train)
            best_model = grid.best_estimator_
            print(f"Best Parameters: {grid.best_params_}")
        else:
            model.fit(X_train, y_train)
            best_model = model

        mse, r2 = evaluate_model(best_model, X_test, y_test)
        print(f"MSE: {mse:.2f}, R²: {r2:.2f}")

if __name__ == "__main__":
    main()
