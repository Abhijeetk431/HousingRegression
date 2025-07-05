# HousingRegression/regression.py
# Using Linear , Decision Tree and Random Forest regression models

from utils import load_data, split_data
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd

def train_and_evaluate(model, X_train, y_train, X_test, y_test, model_name):
    """
    Trains a given scikit-learn model and evaluates its performance on the test set.

    Args:
        model (estimator): The scikit-learn model object (e.g., LinearRegression()).
        X_train (pd.DataFrame or np.array): Training features.
        y_train (pd.Series or np.array): Training target.
        X_test (pd.DataFrame or np.array): Test features.
        y_test (pd.Series or np.array): Test target.
        model_name (str): A descriptive name for the model.

    Returns:
        dict: A dictionary containing MSE and R2 scores for the model.
    """
    print(f"\n--- Training and Evaluating {model_name} ---")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"  MSE: {mse:.4f}")
    print(f"  R2: {r2:.4f}")

    return {"MSE": mse, "R2": r2}

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
    models = {
        "Linear Regression": LinearRegression(),
        "Decision Tree Regressor": DecisionTreeRegressor(random_state=42),
        "Random Forest Regressor": RandomForestRegressor(random_state=42)
    }

    results = {}

    # Train and Evaluate Each Model
    print("\nTraining and evaluating models...")
    for model_name, model_instance in models.items():
        model_scores = train_and_evaluate(model_instance, X_train, y_train, X_test, y_test, model_name)
        results[model_name] = model_scores

    # Performance Comparison
    print("\n===== Performance Comparison =====")
    results_df = pd.DataFrame(results).T
    print(results_df)


if __name__ == "__main__":
    main()