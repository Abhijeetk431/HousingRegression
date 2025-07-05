# HousingRegression/utils.py

import pandas as pd
import numpy as np

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