# type: ignore[all]

import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import (
    Linear,
    LayerNorm,
    BatchNorm2d,
    Sigmoid,
    ReLU,
    LeakyReLU,
    Tanh,
    Sequential,
    Conv2d,
    MaxPool2d,
    AdaptiveAvgPool2d,
    CrossEntropyLoss,
)
from torch.optim.adam import Adam
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision.datasets import MNIST
from torchvision.transforms import transforms
from torchmetrics import Accuracy, Precision, Recall, F1Score, AveragePrecision, AUROC

import os, sys, shutil
from enum import Enum, unique
import logging
import argparse
from easydict import EasyDict as edict
from tqdm import tqdm

from time import time
from datetime import datetime
from tensorboardX import SummaryWriter

import random
import numpy as np
from matplotlib import pyplot as plt


def viz_mnist_pred(images, preds, targets, num_images=36):
    """
    images: numpy.ndarray, shape: (N, 1, 28, 28)
    preds: numpy.ndarray, shape: (N, 10)
    targets: numpy.ndarray, shape: (N,)
    returns: plt.figure
    """
    num_imgs = min(num_images, images.shape[0])
    images = images[:num_imgs]
    preds = preds[:num_imgs]  # logits
    targets = targets[:num_imgs]

    if preds.ndim == 2:
        confidences = np.max(
            np.exp(preds) / np.sum(np.exp(preds), axis=1, keepdims=True), axis=1
        )
        preds = np.argmax(preds, axis=1)

    grid_size = int(np.sqrt(num_imgs))
    fig = plt.figure(figsize=(grid_size * 2.5, grid_size * 2.5))

    for idx in range(num_imgs):
        ax = fig.add_subplot(grid_size, grid_size, idx + 1)
        img = images[idx].squeeze()
        pred = preds[idx]
        gt = targets[idx]
        is_correct = pred == gt
        color = "green" if is_correct else "red"

        ax.imshow(img, cmap="gray")

        ax.set_title(
            f"Pred: {pred} ({confidences[idx]*100:.0f}%)",
            color=color,
            fontsize=12,
            fontweight="bold",
        )
        ax.axis("off")

    plt.tight_layout()
    return fig


def save_checkpoint(cfg, epoch, model, optim, scheduler, best_metrics, metrics=None):
    file_name = "ckpt-epoch-%03d" % epoch
    labels = []
    if metrics is not None and metrics.better_than(best_metrics):
        labels.append("best")
        best_metrics = metrics
    if epoch == cfg.TRAIN.EPOCHS - 1:
        labels.append("last")
    if len(labels) > 0:
        file_name += "-" + "-".join(labels) + ".pth"
    else:
        file_name += ".pth"
    file_path = os.path.join(cfg.CONST.CKPT_DIR, file_name)

    checkpoint = {
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optim.state_dict(),
        "scheduler": scheduler.state_dict(),
        "best_metrics": best_metrics.state_dict(),
    }
    th.save(checkpoint, file_path)
    logging.info(f"Saved checkpoint to {file_path}")


def to_device(data, device, non_blocking=False):
    if isinstance(data, th.Tensor):
        if device.type == "cuda":
            data = data.cuda(device=device, non_blocking=non_blocking)
        else:
            data = data.to(device)
        return data
    elif isinstance(data, list):
        return [to_device(i, device, non_blocking) for i in data]
    elif isinstance(data, dict):
        return {k: to_device(v, device, non_blocking) for k, v in data.items()}
    else:
        raise ValueError(f"Invalid data type: {type(data)}")


class AverageMeter:
    def __init__(self, items=None):
        self.items = items
        self.n_items = 1 if items is None else len(items)
        self.reset()

    def reset(self):
        self._val = [0] * self.n_items
        self._sum = [0] * self.n_items
        self._count = [0] * self.n_items

    def update(self, values):
        if isinstance(values, list):
            for i, v in enumerate(values):
                self._val[i] = v
                self._sum[i] += v
                self._count[i] += 1
        else:
            self._val[0] = values
            self._sum[0] += values
            self._count[0] += 1

    def val(self, idx=None):
        if idx is None:
            return self._val[0] if self.n_items == 1 else self._val
        else:
            return self._val[idx]

    def count(self, idx=None):
        if idx is None:
            return self._count[0] if self.n_items == 1 else self._count
        else:
            return self._count[idx]

    def avg(self, idx=None):
        if idx is None:
            return (
                self._sum[0] / self._count[0]
                if self.n_items == 1
                else [s / c for s, c in zip(self._sum, self._count)]
            )
        else:
            return self._sum[idx] / self._count[idx]

    def __str__(self):
        if self.n_items == 1:
            return f"{self.val():.3f}"
        else:
            return ", ".join([f"{v:.3f}" for v in self.val()])


class Metrics:
    TORCHMETRICS_PARAMS = {
        "task": "multiclass",
        "num_classes": 10,  # MNIST
        "average": "weighted",
    }
    TORCHMETRICS_CONFIG = {
        "acc": {
            "enabled": True,
            "metric": Accuracy(**TORCHMETRICS_PARAMS),
            "is_greater_better": True,
            "init_value": 0,
        },
        "precision": {
            "enabled": True,
            "metric": Precision(**TORCHMETRICS_PARAMS),
            "is_greater_better": True,
            "init_value": 0,
        },
        "recall": {
            "enabled": True,
            "metric": Recall(**TORCHMETRICS_PARAMS),
            "is_greater_better": True,
            "init_value": 0,
        },
        "f1_score": {
            "enabled": True,
            "metric": F1Score(**TORCHMETRICS_PARAMS),
            "is_greater_better": True,
            "init_value": 0,
        },
        "average_precision": {
            "enabled": True,
            "metric": AveragePrecision(**TORCHMETRICS_PARAMS),
            "is_greater_better": True,
            "init_value": 0,
        },
        "auroc": {
            "enabled": True,
            "metric": AUROC(**TORCHMETRICS_PARAMS),
            "is_greater_better": True,
            "init_value": 0,
        },
    }

    ITEMS = [
        {
            "name": name,
            "enabled": True,
            "eval_func": config["metric"],  # or 'cls._compute_my_metric'
            "is_greater_better": config["is_greater_better"],
            "init_value": config["init_value"],
        }
        for name, config in TORCHMETRICS_CONFIG.items()
        if config["enabled"]
    ]

    @classmethod
    def items(cls):
        return [i for i in cls.ITEMS if i["enabled"]]

    @classmethod
    def names(cls):
        _items = cls.items()
        return [i["name"] for i in _items]

    @classmethod
    def get(cls, pred, gt):
        _items = cls.items()
        _values = []
        for i, item in enumerate(_items):
            eval_func = item["eval_func"]
            _values.append(
                eval(eval_func)(pred, gt)
                if isinstance(eval_func, str)
                else eval_func(pred, gt).item()
            )
        return _values

    def __init__(self, best_metric_name, values):
        self._items = Metrics.items()
        self._values = [i["init_value"] for i in self._items]
        self.best_metric_name = best_metric_name
        self.metric_names = [i["name"] for i in self._items]

        if isinstance(values, list):
            self._values = values
        elif isinstance(values, dict):
            for k, v in values.items():
                if k not in self.metric_names:
                    logging.warning(f"Metric {k} not found, ignored")
                    continue
                self._values[self.metric_names.index(k)] = v
        else:
            raise ValueError(f"Invalid values type: {type(values)}")

    @classmethod
    def _compute_my_metric(cls, pred, gt):
        pass

    def state_dict(self):
        _dict = {}
        for i in range(len(self._items)):
            item = self._items[i]["name"]
            value = self._values[i]
            _dict[item] = value
        return _dict

    def __repr__(self):
        return str(self.state_dict())

    def better_than(self, other=None):
        if other is None:
            return True

        _index = self.metric_names.index(self.best_metric_name)
        if _index == -1:
            raise ValueError(f"Metric {self.best_metric_name} not found")

        _metric = self._items[_index]
        _value = self._values[_index]
        other_value = other._values[_index]
        return (
            _value > other_value
            if _metric["is_greater_better"]
            else _value < other_value
        )

    @classmethod
    def to(cls, device):
        _items = cls.items()
        for i in range(len(_items)):
            item = _items[i]
            item["eval_func"] = item["eval_func"].to(device)


def count_parameters(model):
    # numel() returns the number of elements in a tensor
    return sum(p.numel() for p in model.parameters())


def init_weights(m):
    if isinstance(m, Linear):
        # '_' denotes in-place operation
        nn.init.normal_(m.weight, std=0.01)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, Conv2d):
        nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
        if m.bias is not None:
            nn.init.zeros_(m.bias)


class ResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding="same",
            bias=False,
        )
        self.bn1 = BatchNorm2d(out_channels)
        self.relu = ReLU(inplace=True)
        self.conv2 = Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            padding="same",
            bias=False,
        )
        self.bn2 = BatchNorm2d(out_channels)

        self.shortcut = nn.Identity()
        if in_channels != out_channels:
            self.shortcut = Sequential(
                Conv2d(
                    in_channels, out_channels, kernel_size=1, stride=stride, bias=False
                ),
                BatchNorm2d(out_channels),
            )

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        identity = self.shortcut(identity)
        out = out + identity
        out = self.relu(out)
        return out


class ResNet18(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.in_channels = 64
        self.relu = ReLU(inplace=True)
        self.before_resnet = Sequential(
            Conv2d(1, 64, kernel_size=3, stride=1, padding="same", bias=False),
            BatchNorm2d(64),
            self.relu,
        )
        self.maxpool = MaxPool2d(kernel_size=2, stride=2)
        self.layer1 = self._make_layer(64, 2, 1)
        self.layer2 = self._make_layer(128, 2, 1)
        self.layer3 = self._make_layer(256, 2, 1)
        self.layer4 = self._make_layer(512, 2, 1)
        self.avgpool = AdaptiveAvgPool2d((1, 1))
        self.fc = Linear(512, 10)

    def forward(self, x):
        x = self.before_resnet(x)  # B, 64, 28, 28
        x = self.layer1(x)  # B, 64, 14, 14
        x = self.layer2(x)  # B, 128, 7, 7
        x = self.layer3(x)  # B, 256, 4, 4
        x = self.layer4(x)  # B, 512, 2, 2
        x = self.avgpool(x)  # B, 512, 1, 1
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

    def _make_layer(self, out_channels, num_blocks, stride=1):
        layers = []
        layers.append(ResNetBlock(self.in_channels, out_channels, stride))
        self.in_channels = out_channels
        for _ in range(num_blocks - 1):
            layers.append(ResNetBlock(out_channels, out_channels, stride))
        layers.append(self.maxpool)
        return Sequential(*layers)


class MLP(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.hidden_dim = cfg.MODEL.MLP.HIDDEN_DIM
        self.net = Sequential(
            Linear(28 * 28, self.hidden_dim),
            LayerNorm(self.hidden_dim),
            Sigmoid(),
            Linear(self.hidden_dim, self.hidden_dim // 2),
            LayerNorm(self.hidden_dim // 2),
            Sigmoid(),
            Linear(self.hidden_dim // 2, 10),
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.net(x)


def get_model(cfg):
    MODEL_MAP = {
        "mlp": MLP,
        "resnet18": ResNet18,
    }

    return MODEL_MAP[cfg.MODEL.NAME](cfg)


@unique
class DatasetSubset(Enum):
    TRAIN = 0
    VAL = 1
    TEST = 2


class MY_DATASET(Dataset):
    def __init__(self, cfg, subset: DatasetSubset, transform=None):
        self.cfg = cfg
        self.subset = subset
        self.transform = transform
        if self.cfg.CONST.DATASET == "MNIST":
            self.data = MNIST(
                root=self.cfg.CONST.DATA_DIR,
                train=self.subset == DatasetSubset.TRAIN,
                transform=None,
                download=True,
            )
        else:
            raise ValueError(f"Dataset {self.cfg.CONST.DATASET} not supported")
        logging.info(
            f"Dataset {self.cfg.CONST.DATASET} loaded with {len(self.data)} samples"
        )

    def __len__(self):
        len_debug = 10 if self.subset == DatasetSubset.TRAIN else 36
        return len_debug if self.cfg.DEBUG else len(self.data)

    def __getitem__(self, index):
        img, target = self.data[index]
        if self.transform is not None:
            img = self.transform(img)
        return img, target

    def _load_data(self, **kwargs):
        pass


class MY_MNIST:
    def __init__(self, cfg):
        self.cfg = cfg

    def get_dataset(self, subset: DatasetSubset):
        return MY_DATASET(self.cfg, subset, self._get_transform())

    def _get_transform(self):
        return transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )

    def _get_file_list(self, **kwargs):
        pass


def get_data_loader(cfg):
    DATASET_LOADER_MAP = {
        "MNIST": MY_MNIST,
    }

    return DATASET_LOADER_MAP[cfg.CONST.DATASET](cfg)


def train_net(cfg):
    # Set up DataLoader
    train_dataset = get_data_loader(cfg).get_dataset(DatasetSubset.TRAIN)
    val_dataset = get_data_loader(cfg).get_dataset(DatasetSubset.VAL)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=cfg.TRAIN.BATCH_SIZE,
        shuffle=True,
        num_workers=cfg.CONST.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
        worker_init_fn=worker_init_fn,
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=cfg.VAL.BATCH_SIZE,
        shuffle=False,
        num_workers=cfg.CONST.NUM_WORKERS,
        pin_memory=True,
        drop_last=False,
        worker_init_fn=worker_init_fn,
    )

    # Set up folders for logs and checkpoints
    output_dir = os.path.join(
        cfg.CONST.OUT_DIR, cfg.MODEL.NAME, cfg.CONST.DATASET, "%s"
    )
    cfg.CONST.CKPT_DIR = output_dir % "checkpoints"
    cfg.CONST.LOG_DIR = output_dir % "logs"
    os.makedirs(cfg.CONST.CKPT_DIR, exist_ok=True)

    # Clean up the log directory
    if os.path.exists(cfg.CONST.LOG_DIR):
        shutil.rmtree(cfg.CONST.LOG_DIR)
        logging.warning(f"Log directory {cfg.CONST.LOG_DIR} cleaned up")

    # Set up writer for TensorBoard
    train_writer = SummaryWriter(os.path.join(cfg.CONST.LOG_DIR, "train"))
    val_writer = SummaryWriter(os.path.join(cfg.CONST.LOG_DIR, "val"))

    # Create and init the model
    model = get_model(cfg)
    model.apply(init_weights)
    logging.info(
        f"Model {cfg.MODEL.NAME} created with {count_parameters(model)} parameters"
    )

    # Move the model to GPU if available
    model = model.to(cfg.CONST.DEVICE)
    Metrics.to(cfg.CONST.DEVICE)

    # Set up optimizer
    optim = Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=cfg.TRAIN.LR)
    scheduler = ExponentialLR(optim, gamma=cfg.TRAIN.GAMMA)

    # Set up loss functions
    loss = CrossEntropyLoss()

    # Load pretrained model if exists
    init_epoch = 0
    best_metrics = None
    if "WEIGHTS" in cfg.MODEL and os.path.exists(cfg.MODEL.WEIGHTS):
        checkpoint = th.load(cfg.MODEL.WEIGHTS, map_location=cfg.CONST.DEVICE)
        best_metrics = Metrics(cfg.CONST.BEST_METRIC_NAME, checkpoint["best_metrics"])
        model.load_state_dict(checkpoint["model"])
        init_epoch = checkpoint["epoch"] + 1  # start from next epoch
        if "optimizer" in checkpoint:
            optim.load_state_dict(checkpoint["optimizer"])
        if "scheduler" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler"])
        logging.info(f"Loaded checkpoint from {cfg.MODEL.WEIGHTS}")

    # Training loop
    for epoch in range(init_epoch, cfg.TRAIN.EPOCHS):
        epoch_start_time = time()

        # Set up average meters
        batch_time = AverageMeter()  # time for one batch
        data_time = AverageMeter()  # time for loading data
        _loss_meter = AverageMeter()  # _loss_meter = AverageMeter(['loss1', 'loss2'])

        # record lr
        train_writer.add_scalar("LR/Epoch", optim.param_groups[0]["lr"], epoch)

        model.train()

        batch_end_time = time()
        n_batches = len(train_dataloader)
        # batch loop
        for batch_idx, (img, target) in enumerate(train_dataloader):
            data_time.update(time() - batch_end_time)

            # move to GPU
            img, target = to_device(
                [img, target], th.device(cfg.CONST.DEVICE), non_blocking=True
            )

            # forward & backward
            pred = model(img)
            _loss = loss(pred, target)
            model.zero_grad()
            _loss.backward()
            optim.step()

            loss_value = _loss.item() * 1e3

            # record loss per batch
            _loss_meter.update(loss_value)
            train_writer.add_scalar(
                "Loss/Batch", loss_value, epoch * n_batches + batch_idx
            )

            batch_time.update(time() - batch_end_time)
            batch_end_time = time()

            # batch logging
            step_idx = epoch * n_batches + batch_idx
            if (step_idx + 1) % cfg.TRAIN.LOGGING_INTERVAL == 0:
                logging.info(
                    "Epoch %d/%d, Batch %d/%d, BatchTime: %.3fs, DataTime: %.3fs, Loss: %.3f"
                    % (
                        epoch,
                        cfg.TRAIN.EPOCHS,
                        batch_idx,
                        n_batches,
                        batch_time.val(),
                        data_time.val(),
                        loss_value,
                    )
                )

        scheduler.step()

        epoch_end_time = time()
        epoch_time = epoch_end_time - epoch_start_time

        # epoch logging
        logging.info(
            "Epoch %d/%d, EpochTime: %.3fs, Loss: %.3f"
            % (
                epoch,
                cfg.TRAIN.EPOCHS,
                epoch_time,
                _loss_meter.avg(),
            )
        )

        # validation
        if (epoch + 1) % cfg.VAL.VAL_INTERVAL == 0:
            metrics = test_net(cfg, epoch, val_dataloader, val_writer, model)

        # save checkpoint
        if (epoch + 1) % cfg.TRAIN.SAVE_INTERVAL == 0:
            save_checkpoint(cfg, epoch, model, optim, scheduler, best_metrics, metrics)

    # save final checkpoint
    save_checkpoint(
        cfg, cfg.TRAIN.EPOCHS, model, optim, scheduler, best_metrics, metrics
    )

    # close writer
    train_writer.close()
    val_writer.close()


def test_net(cfg, epoch=-1, test_dataloader=None, test_writer=None, model=None):
    if test_dataloader is None:
        test_dataset = get_data_loader(cfg).get_dataset(DatasetSubset.TEST)
        test_dataloader = DataLoader(
            test_dataset,
            batch_size=cfg.TEST.BATCH_SIZE,
            shuffle=False,
            num_workers=cfg.CONST.NUM_WORKERS,
            pin_memory=True,
            drop_last=False,
        )
    if model is None:
        model = get_model(cfg)
        model = model.to(cfg.CONST.DEVICE)
        if "WEIGHTS" in cfg.MODEL and os.path.exists(cfg.MODEL.WEIGHTS):
            checkpoint = th.load(cfg.MODEL.WEIGHTS, map_location=cfg.CONST.DEVICE)
            model.load_state_dict(checkpoint["model"])
            logging.info(f"Loaded checkpoint from {cfg.MODEL.WEIGHTS}")

    model.eval()

    # set up loss functions
    loss = CrossEntropyLoss()  # log_softmax + nll_loss

    # set up average meters
    _loss_meter = AverageMeter()
    metrics_meter = AverageMeter(Metrics.names())

    # set up lists for viz
    imgs_viz = []
    preds_viz = []
    targets_viz = []

    # test loop
    with th.no_grad():
        n_imgs = 0
        n_batches = len(test_dataloader)
        for batch_idx, (img, target) in enumerate(test_dataloader):
            img, target = to_device(
                [img, target], th.device(cfg.CONST.DEVICE), non_blocking=True
            )

            pred = model(img)
            _loss = loss(pred, target)

            loss_value = _loss.item() * 1e3
            _loss_meter.update(loss_value)

            metrics = Metrics.get(pred, target)
            metrics_meter.update(metrics)

            logging.info(
                "Epoch %d/%d, TestBatch %d/%d, Loss: %.3f, Metrics: %s"
                % (
                    epoch,
                    cfg.TRAIN.EPOCHS,
                    batch_idx,
                    n_batches,
                    _loss_meter.val(),
                    metrics_meter,
                )
            )

            # add to lists for viz
            if n_imgs < 36:
                imgs_viz.append(img.cpu().numpy())
                preds_viz.append(pred.cpu().numpy())
                targets_viz.append(target.cpu().numpy())
                n_imgs += len(img)

        print("== TEST RESULTS ==")
        for name, value in zip(Metrics.names(), metrics_meter.avg()):
            print(f"{name}: {value:.3f}", end="  ")
        print("\n==================")

        if test_writer is not None:
            test_writer.add_scalar("Loss/Epoch", _loss_meter.avg(), epoch)
            for name, value in zip(Metrics.names(), metrics_meter.avg()):
                test_writer.add_scalar(f"Metrics/{name}/Epoch", value, epoch)

            # viz
            imgs_viz = np.concatenate(imgs_viz, axis=0)
            preds_viz = np.concatenate(preds_viz, axis=0)
            targets_viz = np.concatenate(targets_viz, axis=0)
            fig = viz_mnist_pred(imgs_viz, preds_viz, targets_viz)
            test_writer.add_figure("Predictions/Epoch", fig, epoch)

    return Metrics(cfg.CONST.BEST_METRIC_NAME, metrics_meter.avg())


def infer_net(cfg, **kwargs):
    pass


def worker_init_fn(worker_id):
    """
    set random seed for each worker
    """
    worker_seed = th.initial_seed() % 2**32 + worker_id
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    th.manual_seed(worker_seed)


def set_random_seed(seed, deterministic=True):
    random.seed(seed)
    np.random.seed(seed)
    th.manual_seed(seed)
    th.cuda.manual_seed(seed)
    th.cuda.manual_seed_all(seed)  # multi-GPU

    if deterministic:
        th.backends.cudnn.benchmark = False
        th.backends.cudnn.deterministic = True
    else:
        th.backends.cudnn.benchmark = True
        th.backends.cudnn.deterministic = False


def get_args(cfg):
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=str, default=cfg.CONST.DEVICE)
    parser.add_argument("--root", type=str, default=cfg.CONST.ROOT)
    parser.add_argument("--test", action="store_true", default=False)
    parser.add_argument("--infer", action="store_true", default=False)
    parser.add_argument("--weights", type=str, default=None)
    parser.add_argument("--debug", action="store_true", default=cfg.DEBUG)

    args = parser.parse_args()

    # args.weights = "/root/tmp/out/mlp/MNIST/checkpoints/ckpt-epoch-003-best.pth"  # DEBUG

    return args


def get_config():
    _C = edict()
    cfg = _C

    _C.DEBUG = False

    _C.CONST = edict()
    _C.CONST.DEVICE = str(th.device("cuda:0" if th.cuda.is_available() else "cpu"))
    _C.CONST.ROOT = os.path.expanduser("~/tmp")
    _C.CONST.DATA_DIR = os.path.join(_C.CONST.ROOT, "datasets")
    _C.CONST.OUT_DIR = os.path.join(_C.CONST.ROOT, "out")
    _C.CONST.SEED = 1024
    _C.CONST.NUM_WORKERS = 4
    _C.CONST.DATASET = "MNIST"
    _C.CONST.BEST_METRIC_NAME = "acc"

    _C.MODEL = edict()
    _C.MODEL.NAME = "resnet18"  # mlp / resnet18

    _C.MODEL.MLP = edict()
    _C.MODEL.MLP.HIDDEN_DIM = 128

    _C.MODEL.RESNET = edict()

    _C.TRAIN = edict()
    _C.TRAIN.BATCH_SIZE = 2 if _C.DEBUG else 16
    _C.TRAIN.EPOCHS = 10 if _C.DEBUG else 100
    _C.TRAIN.LR = 1e-1
    _C.TRAIN.GAMMA = 0.99
    _C.TRAIN.LOGGING_INTERVAL = 1 if _C.DEBUG else 100  # based on step
    _C.TRAIN.SAVE_INTERVAL = 1 if _C.DEBUG else 20  # based on epoch

    _C.VAL = edict()
    _C.VAL.BATCH_SIZE = 2 if _C.DEBUG else 16
    _C.VAL.VAL_INTERVAL = 1 if _C.DEBUG else 10  # based on epoch

    _C.TEST = edict()
    _C.TEST.BATCH_SIZE = 2 if _C.DEBUG else 16

    return cfg


def main():
    cfg = get_config()
    args = get_args(cfg)
    if args.gpu is not None:
        cfg.CONST.DEVICE = args.gpu
    if args.root is not None:
        cfg.CONST.ROOT = args.root
    if args.weights is not None:
        cfg.MODEL.WEIGHTS = args.weights
    cfg.DEBUG = args.debug

    logging.info(f"Use config: {cfg}")

    if cfg.CONST.DEVICE.startswith("cuda"):
        os.environ["CUDA_VISIBLE_DEVICES"] = cfg.CONST.DEVICE.split(":")[-1]

    # reproducibility
    set_random_seed(cfg.CONST.SEED)
    th.set_num_threads(cfg.CONST.NUM_WORKERS)

    if not args.test and not args.infer:
        train_net(cfg)
    else:
        if "WEIGHTS" not in cfg.MODEL or not os.path.exists(cfg.MODEL.WEIGHTS):
            logging.error(f"Weights file not found: {cfg.MODEL.WEIGHTS}")
            exit(1)
        if args.test:
            test_net(cfg)
        elif args.infer:
            infer_net(cfg)


if __name__ == "__main__":
    logging.basicConfig(
        format="[%(levelname)s] %(asctime)s %(message)s", level=logging.INFO
    )
    main()
