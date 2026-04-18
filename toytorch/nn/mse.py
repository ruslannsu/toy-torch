import numpy as np
from .parameter import Parameter

class MSE():
    def __init__(self) -> None:
        self._item = None


    def calc_loss(self, y: np.array, target: np.array) -> np.array:
        self._item = np.power((y - target), 2) * 0.5
        self.input = y
        self.target = target

    def backward(self):
        self.backward_grad = self.input - self.target

        
        
    def __call__(self, y, target):
        self.calc_loss(y=y, target=target)

    def item(self):
        return self._item
    