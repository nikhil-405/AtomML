# pytest .\tests\test_logistic_regression.py
import numpy as np
from utils.data_loaders import load_tabular_data
from classification import LogisticRegression
from sklearn.linear_model import LogisticRegression as SklearnLogisticRegression
from sklearn.preprocessing import StandardScaler

file_path = r"data/simple_logistic_regression_data.csv"

def test_logistic_regression():
    x, y = load_tabular_data(file_path, x = ["x1", "x2", "x3", "x4"], y = ["y"])
    x = StandardScaler().fit_transform(x) # this is necessary to avoid gradient blow ups

    # Custom implementation
    model = LogisticRegression(lr = 0.1, n_iters = 1000)
    model.fit(x, y)
    predictions = model.predict_proba(x)

    # sklearn implementation
    sk = SklearnLogisticRegression()
    sk.fit(x, y)
    predictions_sk = sk.predict_proba(x)

    # print(predictions)
    # print(np.sum(predictions, axis = 1))
    # print(predictions_sk)

    # acc = np.mean(predictions == predictions_sk)
    # assert acc > 0.95, f"Accuracy too low: {acc:.3f}"

    diff = np.abs(predictions - predictions_sk).mean()
    assert diff < 1e-1, f"Prediction difference too high: {diff:.3f}"