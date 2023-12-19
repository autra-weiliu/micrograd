from nn import MLP
from engine import ScalarValue

import random

mlp = MLP(in_dim=3, out_dim=1, dims_between=[4, 3, 4])
inputs = [ScalarValue(random.uniform(-1, 1)) for _ in range(3)]
outputs = mlp(inputs)
outputs.backward()

print(mlp.layers[0].neutrons[0].w[0].grad)
print(len(mlp.parameters()))
