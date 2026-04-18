from toytorch import nn
import numpy as np
class Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layer = nn.Linear(3, 4)
        self.layer2 = nn.Linear(4, 5)

    def forward(self, x):
        x = self.layer(x)
        return self.layer2(x)
    
    


model = Model()

inp = np.array([[[1, 2, 3], [5, 4, 3]]])
out= model(inp)

y = np.array([[[1, 2, 3, 5, 6], [5, 4, 3, 2, 1]]])
print(y)
a = nn.MSE()
a.calc_loss(y, out)
a.backward(model.parameter)

