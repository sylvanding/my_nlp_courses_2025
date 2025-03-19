import torch as th
import torch.nn.functional as F


class BatchNorm1d:
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        self.num_features = num_features
        self.eps = eps
        self.training = True
        # 可训练参数
        self.gamma = th.ones(num_features, requires_grad=True)
        self.beta = th.zeros(num_features, requires_grad=True)
        # 估计所有数据的均值和方差
        self.momentum = momentum
        self.running_mean = th.zeros(num_features)
        self.running_var = th.ones(num_features)

    def forward(self, x):
        if self.training:
            xmean = x.mean(dim=0, keepdim=True)
            # unbiased = True: 无偏估计，使用 N-1 作为除数，用于从总体中抽样并估计总体方差
            # 相反，则为有偏估计，N 作为除数，计算总体方差
            # 在 BatchNorm 中，并不是在做统计推断来估计总体方差
            # 当前 Batch 的样本就是希望标准化的总体，或者说是完整的数据集
            xvar = x.var(dim=0, keepdim=True, unbiased=False)
        else:
            # 非训练模式下，使用训练过程中估计的均值和方差
            xmean = self.running_mean
            xvar = self.running_var
        # 标准化
        xhat = (x - xmean) / th.sqrt(xvar + self.eps)
        # 缩放和平移
        self.out = self.gamma * xhat + self.beta
        if self.training:
            with th.no_grad():
                self.running_mean = (
                    1 - self.momentum
                ) * self.running_mean + self.momentum * xmean
                self.running_var = (
                    1 - self.momentum
                ) * self.running_var + self.momentum * xvar
        return self.out

    def __call__(self, x):
        return self.forward(x)

    def parameters(self):
        return [self.gamma, self.beta]

    def train(self, mode=True):
        self.training = mode

    def eval(self):
        self.training = False


class BatchNorm2d:
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        self.num_features = num_features  # C
        self.eps = eps
        self.training = True
        # 可训练参数
        self.gamma = th.ones(num_features, requires_grad=True)
        self.beta = th.zeros(num_features, requires_grad=True)
        # 估计所有数据的均值和方差
        self.momentum = momentum
        self.running_mean = th.zeros(num_features)
        self.running_var = th.ones(num_features)

    def forward(self, x):
        # x的形状为 [N, C, H, W]
        N, C, H, W = x.shape
        # 将x重塑为[N, C, H*W]以便在 H*W 维度上计算统计量
        x_reshaped = x.reshape(N, C, -1)

        if self.training:
            # 在N和H*W维度上计算均值和方差
            xmean = x_reshaped.mean(dim=[0, 2], keepdim=True)  # [1, C, 1]
            xvar = x_reshaped.var(dim=[0, 2], keepdim=True, unbiased=False)  # [1, C, 1]

            # 调整形状以便广播
            xmean = xmean.reshape(1, C, 1, 1)
            xvar = xvar.reshape(1, C, 1, 1)
        else:
            # 非训练模式下，使用训练过程中估计的均值和方差
            xmean = self.running_mean.reshape(1, C, 1, 1)
            xvar = self.running_var.reshape(1, C, 1, 1)

        # 标准化
        xhat = (x - xmean) / th.sqrt(xvar + self.eps)

        # 缩放和平移 (gamma和beta需要调整形状以便广播)
        gamma_expanded = self.gamma.reshape(1, C, 1, 1)
        beta_expanded = self.beta.reshape(1, C, 1, 1)
        self.out = gamma_expanded * xhat + beta_expanded

        if self.training:
            with th.no_grad():
                # 更新running_mean和running_var (需要去除多余的维度)
                self.running_mean = (
                    1 - self.momentum
                ) * self.running_mean + self.momentum * xmean.reshape(C)
                self.running_var = (
                    1 - self.momentum
                ) * self.running_var + self.momentum * xvar.reshape(C)

        return self.out

    def __call__(self, x):
        return self.forward(x)

    def parameters(self):
        return [self.gamma, self.beta]

    def train(self, mode=True):
        self.training = mode

    def eval(self):
        self.training = False


class LayerNorm:
    def __init__(self, normalized_shape, eps=1e-5):
        """
        对于形状为[N, C, H, W]的图片输入：
        如果设置normalized_shape=(C, H, W)，LayerNorm会对每个样本的所有特征通道、高度和宽度进行归一化
        如果设置normalized_shape=(H, W)，则会对每个样本的每个通道分别进行归一化
        dims = tuple(range(-len(self.normalized_shape), 0)) 动态计算要归一化的维度
        """

        if isinstance(normalized_shape, int):
            self.normalized_shape = (normalized_shape,)
        else:
            self.normalized_shape = tuple(normalized_shape)

        self.eps = eps

        self.weight = th.ones(self.normalized_shape, requires_grad=True)
        self.bias = th.zeros(self.normalized_shape, requires_grad=True)

        self.training = True

    def forward(self, x):
        dims = tuple(range(-len(self.normalized_shape), 0))

        mean = x.mean(dims, keepdim=True)
        var = x.var(dims, keepdim=True, unbiased=False)
        x_norm = (x - mean) / th.sqrt(var + self.eps)

        # 确保 weight 和 bias 的维度与 x_norm 的最后几个维度匹配
        x_norm = self.weight * x_norm + self.bias

        return x_norm

    def __call__(self, x):
        return self.forward(x)

    def parameters(self):
        return [self.weight, self.bias]

    def train(self, mode=True):
        self.training = mode

    def eval(self):
        self.training = False
