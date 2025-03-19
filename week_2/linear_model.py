"""
线性回归模型
"""

from utils import Scalar


def mse(errors):
    n = len(errors)
    wrt = {}
    value = 0.0
    requires_grad = False
    for item in errors:
        value += item.value**2 / n
        wrt[item] = 2 * item.value / n
        requires_grad = requires_grad or item.requires_grad
    out = Scalar(value, errors, "mse", requires_grad=requires_grad)
    out.grad_wrt = wrt
    return out


class Linear:

    def __init__(self):
        self.a = Scalar(0.0, label="a")
        self.b = Scalar(0.0, label="b")

    def forward(self, x):
        return self.a * x + self.b

    def error(self, x, y):
        return y - self.forward(x)

    def string(self):
        return f"y={self.a.value:.2f}x+{self.b.value:.2f}"

    def parameters(self):
        return [self.a, self.b]

    def __call__(self, x):
        return self.forward(x)
