from __future__ import annotations
from enum import Enum
from typing import Union, Tuple, List

import math


class OpType(Enum):
    LEAF = 0
    ADD = 1
    MUL = 2
    RELU = 3
    EXP = 4
    SIGMOID = 5
    POW = 6


class ScalarValue:
    def __init__(self,
        data: float = 0,
        grad: float = 0,
        children: Tuple[ScalarValue] = (),
        op: OpType = OpType.LEAF,) -> None:
        assert type(data) == float, 'scalar value should be float value'
        self.data = data
        self.children = set(children)
        self.op = op
        self.grad = grad
        self._backward = lambda: None

    def __add__(self, val: Union[ScalarValue, float]) -> ScalarValue:
        val = self._parse_val(val=val)
        add_value = ScalarValue(self.data + val.data, children=(self, val), op=OpType.ADD)
        def _backward():
            self.grad += add_value.grad
            val.grad += add_value.grad
        add_value._backward = _backward
        return add_value

    def __sub__(self, val: Union[ScalarValue, float]) -> ScalarValue:
        return self + (- val)

    def __mul__(self, val: Union[ScalarValue, float]) -> ScalarValue:
        val = self._parse_val(val=val)
        mul_value = ScalarValue(self.data * val.data, children=(self, val), op=OpType.MUL)
        def _backward():
            self.grad += val.data * mul_value.grad
            val.grad += self.data * mul_value.grad
        mul_value._backward = _backward
        return mul_value

    def __neg__(self) -> ScalarValue:
        return -1.0 * self

    def __abs__(self) -> ScalarValue:
        return self if self.data > 0 else (- self)

    def __truediv__(self, val: Union[ScalarValue, float]) -> ScalarValue:
        val = self._parse_val(val=val)
        return self * val.pow(-1.0)

    def __rmul__(self, val: Union[ScalarValue, float]) -> ScalarValue:
        val = self._parse_val(val=val)
        return self * val

    def __pow__(self, val) -> ScalarValue:
        return self.pow(val=val)

    def item(self) -> float:
        return self.data

    def pow(self, val: Union[ScalarValue, float]) -> ScalarValue:
        val = self._parse_val(val=val)
        pow_value = ScalarValue(data=math.pow(self.data, val.data), children=(self, ), op=OpType.POW)
        def _backward():
            self.grad += (val.data * math.pow(self.data, val.data - 1)) * pow_value.grad
        pow_value._backward = _backward
        return pow_value

    def exp(self) -> ScalarValue:
        exp_value = ScalarValue(data=math.exp(self.data), children=(self, ), op=OpType.EXP)
        def _backward():
            self.grad += exp_value.grad * exp_value.data
        exp_value._backward = _backward
        return exp_value

    def relu(self) -> ScalarValue:
        relu_value = ScalarValue(data=self.data if self.data > 0 else 0.0, children=(self, ), op=OpType.RELU)
        def _backward():
            self.grad += (1.0 if self.data > 0 else 0.0) * relu_value.grad
        relu_value._backward = _backward
        return relu_value

    def _parse_val(self, val: Union[ScalarValue, float]) -> ScalarValue:
        if type(val) == float:
            return ScalarValue(data=val)
        if type(val) == ScalarValue:
            return val
        raise RuntimeError(f'Invalid val type: {type(val)}')

    def zero_grad(self):
        self.grad = 0.0

    def backward(self, grad: float=1.0):
        self.grad = grad
        queue: List[ScalarValue] = [self]
        topo_list: List[ScalarValue] = []
        vis_set = set([self])
        while queue:
            node = queue.pop(0)
            # skip leaf nodes' backward, for now all nodes require gradient
            if len(node.children) > 0:
                topo_list.append(node)
            for prev_node in node.children:
                if prev_node not in vis_set:
                    vis_set.add(prev_node)
                    queue.append(prev_node)
        for node in topo_list:
            node._backward()

    def __repr__(self) -> str:
        return f'data: {self.data}'
