import numpy as np

from metrics.classification import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import accuracy_score as sklearn_accuracy_score, precision_score as sklearn_precision_score, recall_score as sklearn_recall_score, f1_score as sklearn_f1_score

from core.tensor import Tensor
import pytest

# Classification metrics
def test_accuracy_score():
    y_true = Tensor(np.array([1, 0, 1, 1, 0]))
    y_pred = Tensor(np.array([1, 0, 0, 1, 0]))
    assert accuracy_score(y_true, y_pred) == pytest.approx(0.8)

def test_precision_score():
    y_true = Tensor(np.array([1, 0, 1, 1, 0]))
    y_pred = Tensor(np.array([1, 0, 0, 1, 0]))
    assert precision_score(y_true, y_pred) == pytest.approx(sklearn_precision_score(y_true.data, y_pred.data), rel=1e-4)

def test_recall_score():
    y_true = Tensor(np.array([1, 0, 1, 1, 0]))
    y_pred = Tensor(np.array([1, 0, 0, 1, 0]))
    assert recall_score(y_true, y_pred) == pytest.approx(sklearn_recall_score(y_true.data, y_pred.data), rel=1e-4)

def test_f1_score():
    y_true = Tensor(np.array([1, 0, 1, 1, 0]))
    y_pred = Tensor(np.array([1, 0, 0, 1, 0]))
    assert f1_score(y_true, y_pred) == pytest.approx(sklearn_f1_score(y_true.data, y_pred.data), rel=1e-4)


# Regression metrics
from metrics.regression import mean_squared_error, root_mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import mean_squared_error as sklearn_mse, mean_absolute_error as sklearn_mae, r2_score as sklearn_r2, root_mean_squared_error as sklearn_rmse

def test_mean_squared_error():
    y_true = Tensor(np.array([3.0, -0.5, 2.0, 7.0]))
    y_pred = Tensor(np.array([2.5, 0.0, 2.0, 8.0]))
    assert mean_squared_error(y_true, y_pred) == pytest.approx(sklearn_mse(y_true.data, y_pred.data), rel=1e-4)

def test_root_mean_squared_error():
    y_true = Tensor(np.array([3.0, -0.5, 2.0, 7.0]))
    y_pred = Tensor(np.array([2.5, 0.0, 2.0, 8.0]))
    assert root_mean_squared_error(y_true, y_pred) == pytest.approx(sklearn_rmse(y_true.data, y_pred.data), rel=1e-4)

def test_mean_absolute_error():
    y_true = Tensor(np.array([3.0, -0.5, 2.0, 7.0]))
    y_pred = Tensor(np.array([2.5, 0.0, 2.0, 8.0]))
    assert mean_absolute_error(y_true, y_pred) == pytest.approx(sklearn_mae(y_true.data, y_pred.data), rel=1e-4)

def test_r2_score():
    y_true = Tensor(np.array([3.0, -0.5, 2.0, 7.0]))
    y_pred = Tensor(np.array([2.5, 0.0, 2.0, 8.0]))
    assert r2_score(y_true, y_pred) == pytest.approx(sklearn_r2(y_true.data, y_pred.data), rel=1e-4)