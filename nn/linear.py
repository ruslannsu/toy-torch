import numpy as np

class Linear:
    def __init__(self, input_size: int, output_size: int, activation=None) -> None:
        self.parameters = np.random.default_rng().normal(size=(output_size, input_size))
        self.input_size = input_size 
        self.output_size = output_size
        self.activation = activation
        self.d_activation = np.eye(self.input_size) if activation is None else activation.diff
         
    def forward(self, x: np.array):
        self.input = x
        return self.parameters @ x

    def backward(self, pred_backward: np.array):
        '''no graph implementation'''
        self.grad = np.outer(pred_backward, self.d_activation @ self.input)
        self.backward_grad =  pred_backward @ self.parameters #надо добавить дифференциал активатора

    def __call__(self, x: np.array):
        return self.forward(x)

