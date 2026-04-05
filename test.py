import numpy as np
import nn

loss = nn.MSE()
model = nn.Linear(5, 2)

X = np.array([1, 3, 4, 5 ,6, 6, 10])

target = np.array([3, 5])
eta = 0.0001

model_1 = nn.Linear(7, 5)

for epoch in range(1000):
    out = model_1(X)
    out  = model(out)
    loss(y=out, target=target)
    print(loss.item(), out)
    loss.backward()
    loss_backward = loss.backward_pass
    model.backward(loss_backward)
    model_1.backward(model.backward_grad)
    #model.parameters = model.parameters - eta * model.grad
    model_1.parameters = model_1.parameters - eta * model_1.grad


