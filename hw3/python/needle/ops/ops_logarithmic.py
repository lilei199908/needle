from typing import Optional, Any, Union
from ..autograd import NDArray
from ..autograd import Op, Tensor, Value, TensorOp
from ..autograd import TensorTuple, TensorTupleOp

from .ops_mathematic import *

import numpy as array_api

class LogSoftmax(TensorOp):
    def compute(self, Z: NDArray) -> NDArray:
        ### BEGIN YOUR SOLUTION
        max_z = Z.max(axis=-1, keepdims=True)
        Z_shift = Z - max_z
        logsumexp_z = array_api.log(array_api.sum(array_api.exp(Z_shift), axis=-1, keepdims=True))
        return Z_shift - logsumexp_z
        ### END YOUR SOLUTION

    def gradient(self, out_grad: Tensor, node: Tensor):
        ### BEGIN YOUR SOLUTION
        z = node.inputs[0]
        y = logsoftmax(z)            # (B, C)
        softmax_z = exp(y)         # exp(logsoftmax) = softmax
        sum_grad = summation(out_grad, axes=(-1,))
        sum_grad = sum_grad.reshape(softmax_z.shape[:-1] + (1,))  # keepdims
        return out_grad - softmax_z * sum_grad
        ### END YOUR SOLUTION


def logsoftmax(a: Tensor) -> Tensor:
    return LogSoftmax()(a)


class LogSumExp(TensorOp):
    def __init__(self, axes: Optional[tuple] = None) -> None:
        self.axes = axes

    def compute(self, Z: NDArray) -> NDArray:
        ### BEGIN YOUR SOLUTION
        M = array_api.max(Z, axis=self.axes, keepdims=True)
        Z = Z - M
        return array_api.log(array_api.sum(array_api.exp(Z), axis=self.axes)) + M.squeeze()
        ### END YOUR SOLUTION

    def gradient(self, out_grad: Tensor, node: Tensor):
        ### BEGIN YOUR SOLUTION
        z = node.inputs[0]
        max_z = z.realize_cached_data().max(self.axes, keepdims=True)
        exp_z = exp(z - max_z)
        sum_exp_z = summation(exp_z, self.axes)
        grad_sum_exp_z = out_grad / sum_exp_z
        expand_shape = list(z.shape)
        axes = range(len(expand_shape)) if self.axes is None else self.axes
        for axis in axes:
            expand_shape[axis] = 1
        grad_exp_z = grad_sum_exp_z.reshape(expand_shape).broadcast_to(z.shape)
        return grad_exp_z * exp_z
        ### END YOUR SOLUTION


def logsumexp(a: Tensor, axes: Optional[tuple] = None) -> Tensor:
    return LogSumExp(axes=axes)(a)

class Softmax(TensorOp):
    def compute(self, Z):
        Z_max = array_api.max(Z, axis=-1, keepdims=True)
        exp_z = array_api.exp(Z - Z_max)
        return exp_z / array_api.sum(exp_z, axis=-1, keepdims=True)

    def gradient(self, out_grad, node):
        z = node.inputs[0]
        s = softmax(z)
        inner = summation(out_grad * s, axes=-1, keepdims=True)
        return out_grad - inner * s

def softmax(a):
    return Softmax()(a)