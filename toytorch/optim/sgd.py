from toytorch.nn import Parameter
import numpy as np

class SGD:
    def __init__(self, lr: float, parameters: Parameter) -> None:
        self.lr = lr
        self.parameters = parameters

    def zero_grad(self):
        for layer in Parameter.layers:
            if Parameter.calling[layer] is None:
                continue
            layer.grad.fill(0)

    def step(self):
        for layer in Parameter.layers:
            if Parameter.calling[layer] is None:
                continue
            Parameter.calling[layer] -= self.lr * layer.grad
