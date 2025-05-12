from abc import ABC, abstractmethod

class Optimizer(ABC):
    """
    Base class for all optimizers.
    """
    def __init__(self, params):
        self.params = list(params)

    @abstractmethod
    def step(self):
        """ update step (optimizer specific)"""
        pass

    def zero_grad(self):
        """ set all parameter gradients to zero """
        for p in self.params:
            p.zero_grad()