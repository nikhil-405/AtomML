import numpy as np
import pytest
from core.tensor import Tensor
from losses.regression import MSE, RMSE, MAE, R2

def numeric_grad(f, x, eps = 1e-6):
    """
    Compute numerical gradient of f at x via centered finite differences.
    f: a function mapping Tensor → Tensor (scalar).
    x: a Tensor of any shape.
    """
    x0 = x.data
    grad = np.zeros_like(x0)
    it = np.nditer(x0, flags = ['multi_index'], op_flags = ['readwrite'])
    while not it.finished:
        idx = it.multi_index
        orig = x0[idx]
        
        x0[idx] = orig + eps
        f_plus = f(Tensor(x0.copy(), requires_grad=False)).data
        
        x0[idx] = orig - eps
        f_minus = f(Tensor(x0.copy(), requires_grad=False)).data
        
        grad[idx] = (f_plus - f_minus) / (2 * eps)
        x0[idx] = orig
        it.iternext()
    return grad

# MSE test
def test_mse():
    y_true = np.array([0.0, 2.0], dtype = float)
    y_pred = np.array([1.0, 1.0], dtype = float)
    
    yt = Tensor(y_true.reshape(-1,1), requires_grad = False)
    yp = Tensor(y_pred.reshape(-1,1), requires_grad = True)
    
    loss = MSE(yt, yp)
    # forward check
    expected_mse = np.mean((y_true - y_pred)**2)
    assert pytest.approx(expected_mse, rel=1e-6) == loss.data
    
    # backward check
    loss.backward() # analytic gradient: d/dyp = 2*(yp - yt)/N
    N = y_true.size
    expected_grad = 2*(y_pred - y_true)/N
    np.testing.assert_allclose(yp.grad.flatten(), expected_grad, rtol=1e-5)
    
    # numeric check
    def f(pred_tensor):
        return MSE(yt, pred_tensor)
    num_grad = numeric_grad(f, yp)
    np.testing.assert_allclose(yp.grad, num_grad, rtol=1e-4)

# RMSE test
def test_rmse():
    y_true = np.array([1.0, 3.0], dtype = float)
    y_pred = np.array([2.0, 2.0], dtype = float)
    
    yt = Tensor(y_true.reshape(-1,1), requires_grad = False)
    yp = Tensor(y_pred.reshape(-1,1), requires_grad = True)
    
    # forward check
    loss = RMSE(yt, yp)
    assert pytest.approx(1.0, rel = 1e-6) == loss.data
    
    # backward check
    loss.backward() # analytic gradient: d/dyp RMSE = (1/(2*sqrt(MSE))) * 2*(yp - yt)/N
    N = y_true.size
    expected_grad = (yp.data.flatten() - y_true) / (N * loss.data)
    np.testing.assert_allclose(yp.grad.flatten(), expected_grad, rtol = 1e-5)

# MAE tests
def test_mae():
    y_true = np.array([0.0, 3.0])
    y_pred = np.array([1.0, 1.0])
    
    yt = Tensor(y_true.reshape(-1,1), requires_grad = False)
    yp = Tensor(y_pred.reshape(-1,1), requires_grad = True)
    
    # forward check
    loss = MAE(yt, yp)
    assert pytest.approx(1.5, rel=1e-6) == loss.data
    
    # backward check
    loss.backward() # analytic gradient: sign(yp-yt)/N = [1, -1]/2
    expected_grad = np.array([1.0, -1.0]) / y_true.size
    np.testing.assert_allclose(yp.grad.flatten(), expected_grad, rtol=1e-5)

# R-squared test
def test_r2():
    y_true = np.array([1.0, 3.0, 5.0])
    y_pred = np.array([2.0, 3.0, 4.0])
    
    yt = Tensor(y_true.reshape(-1,1), requires_grad = False)
    yp = Tensor(y_pred.reshape(-1,1), requires_grad = True)
    
    # forward check
    score = R2(yt, yp)
    mean = np.mean(y_true)
    ss_tot = np.sum((y_true-mean)**2)
    expected = 1 - 2/ss_tot
    assert pytest.approx(expected, rel=1e-6) == score.data