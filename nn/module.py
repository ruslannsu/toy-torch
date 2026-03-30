import numpy as np

class Module:
    def __init__(self) -> None:
        pass
    
    def forward(self, x):
        pass

    def __call__(self, *args):
        self.forward(*args)

