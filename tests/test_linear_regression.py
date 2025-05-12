# pytest .\tests\test_linear_regression.py
import numpy as np
from utils.data_loaders import load_tabular_data
from regression import LinearRegression
from sklearn.linear_model import LinearRegression as SklearnLinearRegression
from sklearn.preprocessing import StandardScaler

file_path = r"data/simple_linear_regression_data.csv"

def test_linear_regression():
    x, y = load_tabular_data(file_path, x = ["x1", "x2", "x3"], y = ["x4"])
    x = StandardScaler().fit_transform(x) # this is necessary to avoid gradient blow ups

    # Custom implementation
    model = LinearRegression(lr = 0.1, n_iters = 1000)
    model.fit(x, y)
    predictions = model.predict(x)

    # sklearn implementation
    sk = SklearnLinearRegression()
    sk.fit(x, y)
    predictions_sk = sk.predict(x)

    diff = np.mean((predictions_sk - predictions) ** 2)
    assert diff < 1e-2, "The implementation seems to be incorrect"
