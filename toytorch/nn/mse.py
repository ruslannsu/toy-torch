import numpy as np
from .parameter import Parameter

class MSE():
    def __init__(self) -> None:
        self._item = None

    def calc_loss(self, y: np.ndarray, target: np.ndarray) -> None:
        self._item = np.power((y - target), 2) * 0.5
        self.input = y
        self.target = target

    def backward(self, Parameter):
        self.backward_grad = self.input - self.target
        for layer in Parameter.layers[::-1]:
            if Parameter.calling[layer] is None:
               continue
            if layer.grad is None:
                layer.grad = np.zeros(Parameter.calling[layer].shape)
            layer.grad += layer.x_input.T @ self.backward_grad
            self.backward_grad = self.backward_grad @ Parameter.calling[layer].T
                    
    def __call__(self, y, target):
        self.calc_loss(y=y, target=target)

    def item(self):
        return self._item
    