import numpy as np
from nn import Linear, ReLU, MSELoss, SGD
from core.tensor import Tensor

def test_simple_network():
    # y = 2*x + 3
    X = np.linspace(-1, 1, 10).reshape(-1, 1)
    y = 2 * X + 3

    layer = Linear(1, 1)
    loss_fn = MSELoss()
    opt = SGD(layer.parameters(), lr = 0.1)

    for _ in range(200):
        x_t = Tensor(X, requires_grad = False)
        y_t = Tensor(y, requires_grad = False)
        preds = layer(x_t)
        loss = loss_fn(preds, y_t)

        opt.zero_grad()
        loss.backward()
        opt.step()

    final_w = layer.w.data.flatten()[0]
    final_b = layer.b.data.flatten()[0]
    assert abs(final_w - 2) < 1e-1, "Faulty implementation of Linear layer"
    assert abs(final_b - 3) < 1e-1, "Faulty implementation of Linear layer"