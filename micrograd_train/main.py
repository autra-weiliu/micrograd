from nn import MLP, MSELoss
from engine import ScalarValue
from optimizer import SGDOptimizer

inputs = [ScalarValue(float(i+1)) for i in range(5)]
outputs = [ScalarValue(1.0), ScalarValue(2.0), ScalarValue(3.0)]

mlp = MLP(5, 3, dims_between=[10, 20, 30, 20, 10])
optimizer = SGDOptimizer(params=mlp.parameters())
loss = MSELoss()

epoch_num = 2000
for epoch in range(1, epoch_num+1, 1):
    forward_output = mlp(inputs=inputs)
    mse_loss = loss(pred_outputs=forward_output, gt_outputs=outputs)
    mse_loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    print(f'epoch: {epoch}, loss: {mse_loss.item()}, output: {forward_output}')
