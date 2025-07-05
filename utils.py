# HousingRegression/utils.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def load_data():
    """
    Loads the Boston Housing dataset manually from a URL as specified in the assignment.
    Returns a pandas DataFrame with features and the target variable 'MEDV'.
    """
    data_url = "http://lib.stat.cmu.edu/datasets/boston"
    raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)

    # now we split this into data and target
    data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
    target = raw_df.values[1::2, 2]

    # These are the Feature names based on the original dataset
    feature_names = [
        'CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE',
        'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT'
    ]

    # Create a DataFrame
    df = pd.DataFrame(data, columns=feature_names)
    df['MEDV'] = target  # here MEDV is our target variableb

    return df

def split_data(df, target_column='MEDV', test_size=0.2, random_state=42):
    """
    Splits the DataFrame into features (X) and target (y),
    then further splits them into training and testing sets.

    Args:
        df (pd.DataFrame): The input DataFrame.
        target_column (str): The name of the target column.
        test_size (float): The proportion of the dataset to include in the test split.
        random_state (int): Controls the shuffling applied to the data before applying the split.

    Returns:
        tuple: X_train, X_test, y_train, y_test
    """
    X = df.drop(columns=[target_column])
    y = df[target_column]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    return X_train, X_test, y_train, y_test