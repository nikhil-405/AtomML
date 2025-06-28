import numpy as np
import pytest
from core.tensor import Tensor
from core.optimizers import Adam, SGD

def test_sgd_1():
    a = Tensor(1.0, requires_grad = True)
    b = Tensor(2.0, requires_grad = True)
    c = Tensor(3.0, requires_grad = True)
    loss = a * b + c
    # no momentum
    optimizer = SGD(params = [a, b, c], lr = 0.1)
    loss.backward()
    optimizer.step()
    assert pytest.approx(0.8, rel=1e-6) == a.data
    assert pytest.approx(1.9, rel=1e-6) == b.data
    assert pytest.approx(2.9, rel=1e-6) == c.data

def test_sgd_2():
    a = Tensor(1.0, requires_grad = True)
    b = Tensor(2.0, requires_grad = True)
    c = Tensor(3.0, requires_grad = True)
    loss = a * b + c
    # non-zero momentum
    optimizer = SGD(params = [a, b, c], lr = 0.1, momentum = 0.9)

    loss.backward()
    optimizer.step()
    assert pytest.approx(0.98, rel=1e-6) == a.data
    assert pytest.approx(1.99, rel=1e-6) == b.data
    assert pytest.approx(2.99, rel=1e-6) == c.data
    
    # second step --> since we do not call loss.backward() again, we will be using outdated gradients
    optimizer.step()
    assert pytest.approx(0.942, rel=1e-6) == a.data
    assert pytest.approx(1.971, rel=1e-6) == b.data
    assert pytest.approx(2.971, rel=1e-6) == c.data


def test_adam():
    a = Tensor(1.0, requires_grad = True)
    b = Tensor(2.0, requires_grad = True)
    c = Tensor(3.0, requires_grad = True)
    loss = a * b + c
    optimizer = Adam(params = [a, b, c], lr = 0.1)
    loss.backward()
    optimizer.step()
    assert pytest.approx(0.9, rel=1e-6) == a.data
    assert pytest.approx(1.9, rel=1e-6) == b.data
    assert pytest.approx(2.9, rel=1e-6) == c.data