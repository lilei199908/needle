"""Optimization module"""
import needle as ndl
import numpy as np
from collections import defaultdict

class Optimizer:
    def __init__(self, params):
        self.params = params

    def step(self):
        raise NotImplementedError()

    def reset_grad(self):
        for p in self.params:
            p.grad = None


class SGD(Optimizer):
    def __init__(self, params, lr=0.01, momentum=0.0, weight_decay=0.0):
        super().__init__(params)
        self.lr = lr
        self.momentum = momentum
        self.u = {}
        self.weight_decay = weight_decay

    def step(self):
        for w in self.params:
            if w.grad is None:
                continue
            grad = w.grad.data
            if self.weight_decay != 0:
                grad = grad + self.weight_decay * w.data
            # init momentum buffer with correct shape/dtype
            if w not in self.u:
                self.u[w] = np.zeros_like(w.data)
            # momentum update
            self.u[w] = self.momentum * self.u[w] + (1 - self.momentum) * grad

            # SGD update
            w.data = w.data - self.lr * self.u[w]


    def clip_grad_norm(self, max_norm=0.25):
        """
        Clips gradient norm of parameters.
        Note: This does not need to be implemented for HW2 and can be skipped.
        """
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION


class Adam(Optimizer):
    def __init__(
        self,
        params,
        lr=0.01,
        beta1=0.9,
        beta2=0.999,
        eps=1e-8,
        weight_decay=0.0,
    ):
        super().__init__(params)
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self.t = 0

        self.m = {}
        self.v = {}

    def step(self):
        ### BEGIN YOUR SOLUTION
        self.t += 1
        for w in self.params:
            if w.grad is None:
                continue
            grad = w.grad.data
            if self.weight_decay != 0:
                grad = grad + self.weight_decay * w.data
            # init momentum buffer with correct shape/dtype
            if w not in self.v:
                self.v[w] = np.zeros_like(w.data)
                self.m[w] = np.zeros_like(w.data)
            # momentum update
            self.v[w] = self.beta1 * self.v[w] + (1 - self.beta1) * grad
            self.m[w] = self.beta2 * self.m[w] + (1 - self.beta2) * (grad ** 2)
            v_corrected = self.v[w] / (1 - self.beta1 ** self.t)
            m_corrected = self.m[w] / (1 - self.beta2 ** self.t)
            # SGD update
            w.data = w.data - self.lr * v_corrected / (m_corrected ** 0.5 + self.eps)
        ### END YOUR SOLUTION
