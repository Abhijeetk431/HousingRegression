# HousingRegression/regression.py
# Using Ridge, Decision Tree and Random Forest regression models

from utils import load_data, split_data
from sklearn.linear_model import Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV 
import pandas as pd

import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)

def train_and_evaluate(model, param_grid, X_train, y_train, X_test, y_test, model_name):
    """
    Performs hyperparameter tuning using GridSearchCV, trains the best model,
    and evaluates its performance on the test set.

    Args:
        model (estimator): The scikit-learn model object (e.g., LinearRegression()).
        param_grid (dict): Dictionary with parameters names (str) as keys and lists of
                           parameter settings to try as values.
        X_train (pd.DataFrame or np.array): Training features.
        y_train (pd.Series or np.array): Training target.
        X_test (pd.DataFrame or np.array): Test features.
        y_test (pd.Series or np.array): Test target.
        model_name (str): A descriptive name for the model.

    Returns:
        dict: A dictionary containing MSE and R2 scores for the model.
    """
    print(f"\n--- Training and Evaluating {model_name} with Hyperparameter tuning---")

    grid_search = GridSearchCV(estimator=model, param_grid=param_grid,
                               cv=5, scoring='neg_mean_squared_error', n_jobs=-1, verbose=1)
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    best_score = -grid_search.best_score_
    print(f"Best parameters found for {model_name}: {best_params}")
    print(f"Best cross-validation MSE (from training set): {best_score:.4f}")
    
    y_pred = best_model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"  MSE: {mse:.4f}")
    print(f"  R2: {r2:.4f}")

    return {"MSE": mse, "R2": r2, "Best Params": best_params}

def main():
    """
    Main function to run the regression workflow:
    Load data, split data, train multiple models, and compare their performance.
    """
    print("Starting ML Regression Workflow...")

    # Load Data
    print("Loading data...")
    df = load_data()
    print("Data loaded successfully.")
    print(df.head())
    print(f"Dataset shape: {df.shape}")

    # Split Data
    print("\nSplitting data into training and testing sets...")
    X_train, X_test, y_train, y_test = split_data(df, target_column='MEDV', test_size=0.2, random_state=42)
    print(f"Training data shape: {X_train.shape}")
    print(f"Testing data shape: {X_test.shape}")

    # Initialize Models
    print("\nInitializing regression models...")
    # Added random_state for reproducibility
    ridge_model = Ridge(random_state=42)
    ridge_param_grid = {
        'alpha': [0.1, 1.0, 10.0, 100.0],
        'fit_intercept': [True, False],
        'solver': ['auto', 'svd', 'cholesky', 'lsqr', 'sag']
    }
    # Decision Tree Regressor
    dt_model = DecisionTreeRegressor(random_state=42)
    dt_param_grid = {
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    # Random Forest Regressor
    rf_model = RandomForestRegressor(random_state=42)
    rf_param_grid = {
        'n_estimators': [100, 200, 300],
        'max_features': ['sqrt', 'log2', None],
        'min_samples_leaf': [1, 2, 4]
    }
    
    models_and_grids = {
        "Ridge Regression": {"model": ridge_model, "param_grid": ridge_param_grid},
        "Decision Tree Regressor": {"model": dt_model, "param_grid": dt_param_grid},
        "Random Forest Regressor": {"model": rf_model, "param_grid": rf_param_grid}
    }

    results = {}

    # Train and Evaluate Each Model
    print("\nTraining and evaluating models...")
    for model_name, config in models_and_grids.items():
        model_scores = train_and_evaluate(config["model"], config["param_grid"],
                                          X_train, y_train, X_test, y_test, model_name)
        results[model_name] = model_scores

    # Performance Comparison
    print("\n===== Performance Comparison =====")
    summary_results = {k: {key: val for key, val in v.items() if key != 'Best Params'} for k, v in results.items()}
    results_df = pd.DataFrame(summary_results).T
    print(results_df)

    print("\n===== Best Parameters Found for Each Model =====")
    for model_name, res in results.items():
        print(f"{model_name}: {res['Best Params']}")


if __name__ == "__main__":
    main()