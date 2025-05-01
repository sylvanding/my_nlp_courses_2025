import os

from matplotlib import pyplot as plt


def draw_loss(loss_list: list[float], save_path: str, title: str = "Loss"):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.plot(loss_list)
    plt.title(title)
    plt.savefig(save_path)
    plt.close()
