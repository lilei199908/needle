"""Operator implementations."""

from numbers import Number
from typing import Optional, List, Tuple, Union

from ..autograd import NDArray
from ..autograd import Op, Tensor, Value, TensorOp
from ..autograd import TensorTuple, TensorTupleOp
import numpy

# NOTE: we will import numpy as the array_api
# as the backend for our computations, this line will change in later homeworks

BACKEND = "np"
import numpy as array_api

class EWiseAdd(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a + b

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad, out_grad


def add(a, b):
    return EWiseAdd()(a, b)


class AddScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a + self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad


def add_scalar(a, scalar):
    return AddScalar(scalar)(a)


class EWiseMul(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a * b

    def gradient(self, out_grad: Tensor, node: Tensor):
        lhs, rhs = node.inputs
        return out_grad * rhs, out_grad * lhs


def multiply(a, b):
    return EWiseMul()(a, b)


class MulScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a * self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        return (out_grad * self.scalar,)


def mul_scalar(a, scalar):
    return MulScalar(scalar)(a)


class EWisePow(TensorOp):
    """Op to element-wise raise a tensor to a power."""

    def compute(self, a: NDArray, b: NDArray) -> NDArray:
        ### BEGIN YOUR SOLUTION
        return a ** b
        ### END YOUR SOLUTION
        
    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        a, b = node.inputs
        return out_grad * b * power(a, b - 1), out_grad * log(a) * power(a, b)
        ### END YOUR SOLUTION

def power(a, b):
    return EWisePow()(a, b)


class PowerScalar(TensorOp):
    """Op raise a tensor to an (integer) power."""

    def __init__(self, scalar: int):
        self.scalar = scalar

    def compute(self, a: NDArray) -> NDArray:
        ### BEGIN YOUR SOLUTION
        return a ** self.scalar
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        a = node.inputs[0]
        return out_grad * self.scalar * a ** (self.scalar - 1)
        ### END YOUR SOLUTION


def power_scalar(a, scalar):
    return PowerScalar(scalar)(a)


class EWiseDiv(TensorOp):
    """Op to element-wise divide two nodes."""

    def compute(self, a, b):
        ### BEGIN YOUR SOLUTION
        return a / b
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        a, b = node.inputs
        return out_grad / b, out_grad * (-a / (b * b))
        ### END YOUR SOLUTION


def divide(a, b):
    return EWiseDiv()(a, b)


class DivScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return a / self.scalar
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return out_grad / self.scalar
        ### END YOUR SOLUTION


def divide_scalar(a, scalar):
    return DivScalar(scalar)(a)


class Transpose(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        axes = list(range(a.ndim))
        if self.axes is None:
            # swap last two dims
            i, j = a.ndim - 2, a.ndim - 1
        else:
            i, j = self.axes
        axes[i], axes[j] = axes[j], axes[i]
        return array_api.transpose(a, axes)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return transpose(out_grad, self.axes)
        ### END YOUR SOLUTION


def transpose(a, axes=None):
    return Transpose(axes)(a)


class Reshape(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return a.reshape(self.shape)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return out_grad.reshape(node.inputs[0].shape)
        ### END YOUR SOLUTION


def reshape(a, shape):
    return Reshape(shape)(a)


class BroadcastTo(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.broadcast_to(a, self.shape)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        input_shape = node.inputs[0].shape
        out_shape = out_grad.shape
        
        # Step 1: 对前面新加的维度 sum
        extra_dims = len(out_shape) - len(input_shape)
        axes = list(range(extra_dims))
        
        # Step 2: 对 input 中为 1 的维度 sum
        for i, (in_size, out_size) in enumerate(zip(input_shape, out_shape[extra_dims:])):
            if in_size == 1:
                axes.append(i + extra_dims)
        
        # Step 3: 真正的 backward
        return summation(out_grad, tuple(axes)).reshape(input_shape)


def broadcast_to(a, shape):
    return BroadcastTo(shape)(a)


class Summation(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return a.sum(axis=self.axes)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        x = node.inputs[0]
        in_shape = x.shape
        axes = self.axes

        # sum over all axes -> out_grad 是标量
        if axes is None:
            # 先 reshape 成 (1, 1, ..., 1)，再 broadcast 回去
            new_shape = (1,) * len(in_shape)
            return broadcast_to(reshape(out_grad, new_shape), in_shape)

        # 统一成 tuple
        if not isinstance(axes, tuple):
            axes = (axes,)

        # 处理负轴
        axes = tuple(a if a >= 0 else a + len(in_shape) for a in axes)

        # 在被 sum 掉的轴位置插入 1
        out_shape = list(out_grad.shape)
        for ax in sorted(axes):
            out_shape.insert(ax, 1)

        out_grad_reshaped = reshape(out_grad, tuple(out_shape))
        return broadcast_to(out_grad_reshaped, in_shape)


def summation(a, axes=None):
    return Summation(axes)(a)


class MatMul(TensorOp):
    def compute(self, a, b):
        ### BEGIN YOUR SOLUTION
        return a @ b
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        a, b = node.inputs

        # 理论上的两个“伴随”矩阵
        adjoint1 = out_grad @ transpose(b)       # 对 a 的梯度（但可能带 batch broadcast）
        adjoint2 = transpose(a) @ out_grad       # 对 b 的梯度

        # 如果在前向里有 broadcast，NumPy 会在前面补 batch 维；
        # 反向就要把这些“多出来的前导维度” sum 掉：
        if len(adjoint1.shape) > len(a.shape):
            adjoint1 = summation(adjoint1,
                                 axes=tuple(range(len(adjoint1.shape) - len(a.shape))))
        if len(adjoint2.shape) > len(b.shape):
            adjoint2 = summation(adjoint2,
                                 axes=tuple(range(len(adjoint2.shape) - len(b.shape))))

        return adjoint1, adjoint2


def matmul(a, b):
    return MatMul()(a, b)


class Negate(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return -a
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return -out_grad
        ### END YOUR SOLUTION


def negate(a):
    return Negate()(a)


class Log(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.log(a)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return out_grad / node.inputs[0]
        ### END YOUR SOLUTION


def log(a):
    return Log()(a)


class Exp(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.exp(a)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return out_grad * exp(node.inputs[0])
        ### END YOUR SOLUTION


def exp(a):
    return Exp()(a)


class ReLU(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.maximum(a, 0)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        out = node.realize_cached_data().copy()
        out[out > 0] = 1
        return out_grad * Tensor(out)
        ### END YOUR SOLUTION


def relu(a):
    return ReLU()(a)

