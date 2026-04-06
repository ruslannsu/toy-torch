import numpy as np

class Linear:
    def __init__(self, input_size: int, output_size: int) -> None:
        self.parameters = np.random.default_rng().normal(size=(output_size, input_size))
        self.input_size = input_size 
        self.output_size = output_size

    def forward(self, x: np.array) -> np.array:
        self.input = x
        return x @ self.parameters.T

    def backward(self, pred_backward: np.array) -> np.array:
        self.grad = np.outer(pred_backward, self.input)
        return pred_backward @ self.parameters
    
    def __call__(self, x: np.array):
        return self.forward(x)
