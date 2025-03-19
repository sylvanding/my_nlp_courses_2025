"""
实现 MLP 各组件，如线性层、激活函数、Sequential 等。
"""

import torch
from sklearn.datasets import make_circles
from torch.utils.data import TensorDataset, DataLoader


class Linear:
    def __init__(self, in_features, out_features, bias=True):
        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias
        self.weight = torch.randn(in_features, out_features)
        self.bias = torch.randn(out_features) if bias else None

    def __call__(self, x):
        self.out = x @ self.weight
        if self.bias is not None:
            self.out += self.bias
        return self.out

    def parameters(self):
        if self.bias is not None:
            return [self.weight, self.bias]
        else:
            return [self.weight]

    def zero_grad(self):
        for param in self.parameters():
            param.grad = None


class Sigmoid:
    def __call__(self, x):
        self.out = 1 / (1 + torch.exp(-x))
        return self.out

    def parameters(self):
        return []

    def zero_grad(self):
        pass


class Sequential:
    def __init__(self, *modules):
        self.modules = modules

    def __call__(self, x):
        for module in self.modules:
            x = module(x)
        self.out = x
        return self.out

    def parameters(self):
        return [param for module in self.modules for param in module.parameters()]

    def zero_grad(self):
        for module in self.modules:
            module.zero_grad()


def cross_entropy(y_pred, y_true):
    # Softmax
    y_pred = torch.exp(y_pred) / torch.sum(torch.exp(y_pred), dim=1, keepdim=True)
    # One-hot encoding
    y_one_hot = torch.zeros_like(y_pred)
    y_one_hot.scatter_(1, y_true.long(), 1)
    # NLLLoss
    return -torch.sum(y_one_hot * torch.log(y_pred))


def accuracy(y_pred, y_true):
    y_pred = torch.argmax(y_pred, dim=1, keepdim=True)
    return torch.sum(y_pred == y_true).item() / len(y_true)


if __name__ == "__main__":
    # 生成训练数据
    X, y = make_circles(n_samples=1000, noise=0.1, factor=0.5, random_state=42)
    X = (X - X.mean(axis=0)) / X.std(axis=0)
    X, y = (
        torch.tensor(X, dtype=torch.float32),
        torch.tensor(y, dtype=torch.float32).unsqueeze(1),
    )

    # 定义多层感知机
    model = Sequential(
        Linear(2, 4),
        Sigmoid(),
        Linear(4, 8),
        Sigmoid(),
        Linear(8, 4),
        Sigmoid(),
        Linear(4, 2),  # Softmax -> NLLLoss（交叉熵在二分类时退化为二元交叉熵）
    )
    for param in model.parameters():
        param.requires_grad = True

    max_epochs = 1000
    batch_size = 100
    learning_rate = 0.1
    print_interval = 100
    dataset = TensorDataset(X, y)
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(max_epochs):
        loss_avg = 0.0
        for batch_idx, (X_batch, y_batch) in enumerate(dataloader):
            y_pred = model(X_batch)
            loss = cross_entropy(y_pred, y_batch)
            loss = loss / len(X_batch)
            loss_avg += loss.item()
            model.zero_grad()
            loss.backward()
            for param in model.parameters():
                param.data -= learning_rate * param.grad
        if epoch % print_interval == 0:
            with torch.no_grad():
                loss_avg /= len(dataloader)
                acc = accuracy(model(X), y)
                print(f"Epoch {epoch}, Loss: {loss_avg*1e3:.2f}, Acc: {acc*1e2:.2f}%")
