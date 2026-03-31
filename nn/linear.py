import numpy as np

class Linear:
    def __init__(self, input_size: int, output_size: int) -> None:
        self.matrix = np.random.default_rng().normal(size=(output_size, input_size))
        self.input_size = input_size
        self.output_size = output_size

    def forward(self, x: np.array):
        return np.matmul(self.matrix, x)

    def __call__(self, x: np.array):
        return self.forward(x)
    