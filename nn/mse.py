import numpy as np

class MSE():
    def __init__(self) -> None:
        pass

    def calc_loss(self, output: np.array, target: np.array) -> np.array:
        return np.power((output - target), 2) * 0.5
    
    def __call__(self, output, target):
        return self.calc_loss(output=output, target=target)
