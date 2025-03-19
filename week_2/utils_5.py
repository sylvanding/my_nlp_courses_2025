import torch as th
import torch.nn.functional as F


class Dropout:
    def __init__(self, p=0.5):
        self.p = p
        self.training = True
        self.mask = None

    def forward(self, x):
        if not self.training or self.p == 0:
            return x

        # 创建随机掩码，保留概率为(1-p)
        self.mask = th.bernoulli(th.ones_like(x) * (1 - self.p))

        # 缩放输出，使得期望值保持不变
        # 在训练时，将保留的值除以(1-p)，这样在测试时就不需要进行任何缩放
        scale = 1.0 / (1.0 - self.p)

        return x * self.mask * scale

    def __call__(self, x):
        return self.forward(x)

    def train(self, mode=True):
        self.training = mode

    def eval(self):
        self.training = False
