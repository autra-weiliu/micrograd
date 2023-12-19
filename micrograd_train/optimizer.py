from engine import ScalarValue
from typing import List

class Optimizer:
    def __init__(self, params: List[ScalarValue]) -> None:
        self._params: List[ScalarValue] = params

    @property
    def params(self) -> List[ScalarValue]:
        return self._params

    def zero_grad(self):
        for param in self._params:
            param.zero_grad()

    def step(self):
        raise NotImplementedError('step is not implemented in optimizer')


class SGDOptimizer(Optimizer):
    def __init__(self, params: List[ScalarValue], lr: float=0.01, decay=0.005) -> None:
        super().__init__(params)
        self._lr = lr
        self._decay = decay

    def step(self):
        for param in self._params:
            param.data -= self._lr * param.grad
        self._lr = self._lr * (1 - self._decay)
