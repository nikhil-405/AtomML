import numpy as np

class Tensor:
    def __init__(self, data, _children = (), _op = "", requires_grad = False):
        self.data = np.array(data, dtype=float)
        self.requires_grad = requires_grad

        if self.requires_grad:
            self.grad = np.zeros_like(self.data)
        else:
            self.grad = None

        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op

    def zero_grad(self):
        if self.requires_grad:
            self.grad = np.zeros_like(self.data)
        else:
            self.grad = None

    @staticmethod
    def _unbroadcast(grad, shape):
        """ Sum gradients to match the target shape after broadcasting"""
        while grad.ndim > len(shape):
            grad = grad.sum(axis = 0)
        # Sum along broadcasted axes
        for i, dim in enumerate(shape):
            if dim == 1:
                grad = grad.sum(axis = i, keepdims = True)
        return grad

    def __add__(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(other)

        out = Tensor(self.data + other.data,
                     (self, other),
                     "+",
                     requires_grad=(self.requires_grad or other.requires_grad))

        def _backward():
            grad = out.grad
            if self.requires_grad:
                grad_self = Tensor._unbroadcast(grad, self.data.shape)
                self.grad += grad_self
            if other.requires_grad:
                grad_other = Tensor._unbroadcast(grad, other.data.shape)
                other.grad += grad_other

        out._backward = _backward
        return out

    def __sub__(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(other)

        out = Tensor(self.data - other.data,
                     (self, other),
                     "-",
                     requires_grad=(self.requires_grad or other.requires_grad))

        def _backward():
            grad = out.grad
            if self.requires_grad:
                grad_self = Tensor._unbroadcast(grad, self.data.shape)
                self.grad += grad_self
            if other.requires_grad:
                grad_other = Tensor._unbroadcast(grad * -1, other.data.shape)
                other.grad += grad_other

        out._backward = _backward
        return out

    def __mul__(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(other)

        out = Tensor(self.data * other.data,
                     (self, other),
                     "*",
                     requires_grad=(self.requires_grad or other.requires_grad))

        def _backward():
            grad = out.grad
            if self.requires_grad:
                grad_self = Tensor._unbroadcast(grad * other.data, self.data.shape)
                self.grad += grad_self
            if other.requires_grad:
                grad_other = Tensor._unbroadcast(grad * self.data, other.data.shape)
                other.grad += grad_other

        out._backward = _backward
        return out

    def __neg__(self):
        return self * -1

    def __radd__(self, other):
        return self + other

    def __rsub__(self, other):
        return Tensor(other) - self

    def __rmul__(self, other):
        return self * other

    def __truediv__(self, other):
        return self * (other ** -1)

    def __rtruediv__(self, other):
        return Tensor(other) * (self ** -1)

    def __pow__(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(other)

        out = Tensor(self.data ** other.data,
                     (self, other),
                     "**",
                     requires_grad=(self.requires_grad or other.requires_grad))

        def _backward():
            if self.requires_grad:
                self.grad += out.grad * (other.data * (self.data ** (other.data - 1)))
            if other.requires_grad:
                self_term = np.where(self.data > 0, self.data, 1.0)
                other.grad += out.grad * (out.data * np.log(self_term))

        out._backward = _backward
        return out

    def exp(self):
        out = Tensor(np.exp(self.data),
                     (self,),
                     "exp",
                     requires_grad=self.requires_grad)

        def _backward():
            if self.requires_grad:
                self.grad += out.grad * out.data

        out._backward = _backward
        return out

    def log(self):
        out = Tensor(np.log(self.data),
                     (self,),
                     "log",
                     requires_grad=self.requires_grad)

        def _backward():
            if self.requires_grad:
                self.grad += (1.0 / self.data) * out.grad

        out._backward = _backward
        return out

    def __matmul__(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(other)

        out = Tensor(self.data.dot(other.data),
                     (self, other),
                     "@",
                     requires_grad=(self.requires_grad or other.requires_grad))

        def _backward():
            if self.requires_grad:
                self.grad += np.matmul(out.grad, other.data.T)
            if other.requires_grad:
                other.grad += np.matmul(self.data.T, out.grad)

        out._backward = _backward
        return out

    def sum(self):
        out = Tensor(self.data.sum(),
                     (self,),
                     "sum",
                     requires_grad = self.requires_grad)

        def _backward():
            if self.requires_grad:
                self.grad += out.grad * np.ones_like(self.data)

        out._backward = _backward
        return out

    def mean(self):
        out = Tensor(self.data.mean(),
                     (self,),
                     "mean",
                     requires_grad = self.requires_grad)

        def _backward():
            if self.requires_grad:
                self.grad += out.grad * np.ones_like(self.data) / self.data.size

        out._backward = _backward
        return out

    # Activation functions
    def relu(self):
        out = Tensor(np.maximum(0, self.data),
                    (self,),
                    "ReLU",
                    requires_grad = self.requires_grad)

        def _backward():
            if self.requires_grad:
                self.grad += out.grad * (out.data > 0).astype(float)

        out._backward = _backward
        return out

    def sigmoid(self):
        sig = 1.0 / (1.0 + np.exp(-self.data))
        out = Tensor(sig,
                     (self,),
                     "sigmoid",
                     requires_grad = self.requires_grad)

        def _backward():
            if self.requires_grad:
                self.grad += out.grad * (sig * (1 - sig))

        out._backward = _backward
        return out
    
    def softmax(self, dim = -1):
        x = self.data
        max_vals = np.max(x, axis = dim, keepdims = True)
        exp_shift = np.exp(x - max_vals)

        sums = np.sum(exp_shift, axis = dim, keepdims = True)
        probs = exp_shift / sums

        out = Tensor(probs, 
                     _children = (self,),
                     _op = "softmax",
                     requires_grad = self.requires_grad)

        def _backward():
            if not self.requires_grad:
                return
            
            g = out.grad # upstream grad
            dot = np.sum(g * probs, axis = dim, keepdims = True)
            grad_input = probs * (g - dot) 
            self.grad += grad_input

        out._backward = _backward
        return out

    def leaky_relu(self, alpha = 0.01):
        out = Tensor(np.where(self.data > 0, self.data, alpha * self.data),
                     (self,),
                     "leaky_relu",
                     requires_grad = self.requires_grad)

        def _backward():
            if self.requires_grad:
                grad = out.grad
                grad_self = np.where(self.data > 0, grad, alpha * grad)
                self.grad += grad_self

        out._backward = _backward
        return out

    # https://arxiv.org/abs/1606.08415
    def gelu(self):
        out = Tensor(self.data * self.data.sigmoid(),
                     (self,),
                     "gelu",
                     requires_grad=self.requires_grad)

        def _backward():
            if self.requires_grad:
                grad = out.grad
                self.grad += grad * (self.data.sigmoid() + self.data * grad.sigmoid())

        out._backward = _backward
        return out

    # backprop
    def backward(self):
        if not self.requires_grad:
            raise RuntimeError("Cannot call backwards on Tensors that do not require gradients!!")

        topo = []
        visited = set()

        def build_topo(tensor):
            if tensor not in visited:
                visited.add(tensor)
                for child in tensor._prev:
                    build_topo(child)
                topo.append(tensor)

        build_topo(self)

        self.grad = np.ones_like(self.data)
        for tensor in reversed(topo):
            tensor._backward()