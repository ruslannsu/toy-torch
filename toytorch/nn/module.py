from .parameter import Parameter

class Module:
    def __init__(self) -> None:
        self.parameter = Parameter([self, None])
        
    def forward(self, x):
        pass

    def __call__(self, *args):
        return self.forward(*args)


