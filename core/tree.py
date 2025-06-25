import numpy as np

class Node:
    def __init__(self, value):
        self.value = value
        self.left = None
        self.right = None
        # self.split_feature = None
        # self.split_value = None

    def __sub__(self, other):
        return np.sum((self.value - other.value)**2) **0.5
    
class 