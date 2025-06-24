from core.tensor import Tensor
from nn.module import BaseModule

class ReLU(BaseModule):
    def forward(self, x):
        return x.relu()
    
class Sigmoid(BaseModule):
    def forward(self, x):
        return x.sigmoid()
    
class Softmax(BaseModule):
    def __init__(self, dim: int = -1):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        return x.softmax(dim = self.dim)

class LeakyReLU(BaseModule):
    def __init__(self, alpha = 0.01):
        super().__init__()
        self.alpha = alpha

    def forward(self, x):
        return x.leaky_relu(alpha = self.alpha)

class GeLU(BaseModule):
    def forward(self, x):
        return x.gelu()
