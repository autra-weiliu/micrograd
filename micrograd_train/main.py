from engine import ScalarValue
import torch

val = ScalarValue(3.0)
another_val = val / 5.12
another_val.backward()
print(val.grad)

t = torch.tensor([3.], requires_grad=True)
another_t = t / 5.12
another_t.backward()
print(t.grad)
