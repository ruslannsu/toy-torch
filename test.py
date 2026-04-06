
import numpy as np

import nn


model = nn.Linear(5, 6)

z  = np.eye(5)

print(model(z))

k = np.eye(6)

model.backward(k)
