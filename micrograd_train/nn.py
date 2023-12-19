from engine import ScalarValue
from typing import List, Union, Tuple

import random


class Module:
    def parameters(self) -> List[ScalarValue]:
        raise NotImplementedError('parameters method is not implemented')

    def forward(self, inputs: List[ScalarValue]) -> Union[ScalarValue, List[ScalarValue]]:
        raise NotImplementedError('forward method is not implemented')

    def __call__(self, inputs: List[ScalarValue]) -> Union[ScalarValue, List[ScalarValue]]:
        return self.forward(inputs=inputs)

    def zero_grad(self):
        for param in self.parameters():
            param.zero_grad()


class Neutron(Module):
    def __init__(self, in_dim: int) -> None:
        self.w: List[ScalarValue] = [ScalarValue(random.uniform(-1, 1)) for _ in range(in_dim)]
        # use a constant initial value first
        self.b: ScalarValue = ScalarValue(0.0)

    def forward(self, inputs: List[ScalarValue]) -> Union[ScalarValue, List[ScalarValue]]:
        assert len(self.w) == len(inputs), 'dim is not aligned between w and inputs'
        return sum([w_val * x_val for (w_val, x_val) in zip(self.w, inputs)], self.b)

    def parameters(self) -> List[ScalarValue]:
        return self.w + [self.b]


class Layer(Module):
    def __init__(self, name: str, in_dim: int, out_dim: int) -> None:
        self._neutrons: List[Neutron] = [Neutron(in_dim=in_dim) for _ in range(out_dim)]
        self._name = name
        self._in_dim = in_dim
        self._out_dim = out_dim

    @property
    def neutrons(self) -> List[Neutron]:
        return self._neutrons

    def forward(self, inputs: List[ScalarValue]) -> Union[ScalarValue, List[ScalarValue]]:
        return [n(inputs) for n in self._neutrons]

    def parameters(self) -> List[ScalarValue]:
        return [param for neutron in self._neutrons for param in neutron.parameters()]

    def __repr__(self) -> str:
        return f'Layer: {self._name}, in_dim: {self._in_dim}, out_dim: {self._out_dim}'


class MLP(Module):
    def __init__(self, in_dim: int, out_dim: int, dims_between: Union[Tuple[int], List[int]]) -> None:
        self._dims: List[int] = [in_dim] + list(dims_between) + [out_dim]
        self._layers: List[Layer] = []
        # build dims
        for dim_array_idx in range(len(self._dims)-1):
            self._layers.append(Layer(name=f'layer_{dim_array_idx+1}', in_dim=self._dims[dim_array_idx], out_dim=self._dims[dim_array_idx+1]))

    @property
    def layers(self) -> List[Layer]:
        return self._layers

    def forward(self, inputs: List[ScalarValue]) -> Union[ScalarValue, List[ScalarValue]]:
        for layer in self._layers:
            inputs = layer(inputs=inputs)
        return inputs[0] if len(inputs) == 1 else inputs
    
    def parameters(self) -> List[ScalarValue]:
        return [param for layer in self._layers for param in layer.parameters()]

    def __repr__(self) -> str:
        mlp_data = f'MLP dims: {self._dims}'
        layer_data = list(map(lambda layer: str(layer), self.layers))
        return '\n'.join([mlp_data] + layer_data)
