import numpy as np
from .parameter import Parameter

class Linear:
    def __init__(self, input_size: int, output_size: int) -> None:
        self.input_size = input_size 
        self.output_size = output_size
        Parameter([self, np.random.default_rng().normal(size=(input_size, output_size))])
        self.grad = None
    
    def __call__(self, x: np.ndarray):
        self.x_input = x
        return x @ Parameter.calling[self]
        
        
