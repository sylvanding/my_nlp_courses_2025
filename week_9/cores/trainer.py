import logging
import os

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from utils.loss_drawer import draw_loss
from utils.Tokenizer import CharacterTokenizer
from utils.training_data_creator import rnn_training_data_creator


def train_model_rnn_batch(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    dataloader: DataLoader,
    loss_fn: nn.Module,
    num_epochs: int = 1000,
    print_every: int = 100,
    save_path: str = "./outputs",
):
    model.train()
    num_batches = len(dataloader)
    epoch_loss_list = []
    for epoch in range(num_epochs):
        epoch_loss = 0
        for batch_idx, data in enumerate(dataloader):
            inputs, labels = data["inputs"], data["labels"]
            logits = model(inputs)  # (B, T, V)
            loss = loss_fn(logits.transpose(-2, -1), labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_loss = round(loss.item() * 1000, 3)

            if batch_idx % print_every == 0:
                logging.info(
                    f"Epoch {epoch}/{num_epochs}, Batch {batch_idx}/{num_batches}, Loss: {batch_loss}"
                )

            epoch_loss += batch_loss

        epoch_loss /= num_batches
        epoch_loss_list.append(epoch_loss)

    model_saver(model, os.path.join(save_path, "checkpoints", "model.pth"))
    draw_loss(
        epoch_loss_list,
        os.path.join(save_path, "logs", "loss.png"),
        title="RNN Batch Loss",
    )


def train_model_rnn(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    dataloader: list[str],
    tokenizer: CharacterTokenizer,
    loss_fn: nn.Module,
    num_epochs: int = 1000,
    print_every: int = 100,
    save_path: str = "./outputs",
):
    model.train()
    num_batches = len(dataloader)
    epoch_loss_list = []
    device = next(model.parameters()).device
    for epoch in range(num_epochs):
        epoch_loss = 0
        for batch_idx, data in enumerate(dataloader):
            inputs, labels = rnn_training_data_creator(data, tokenizer)
            inputs = torch.tensor(inputs, device=device).unsqueeze(0)  # (1, ...)
            labels = torch.tensor(labels, device=device).unsqueeze(0)  # (1, ...)
            hidden = None
            loss = torch.tensor(0.0, device=device)
            for i in range(inputs.shape[1]):
                output, hidden = model(inputs[:, i], hidden)
                loss = loss + loss_fn(output, labels[:, i])
            loss = loss / inputs.shape[1]
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_loss = round(loss.item() * 1000, 3)

            if batch_idx % print_every == 0:
                logging.info(
                    f"Epoch {epoch}/{num_epochs}, Batch {batch_idx}/{num_batches}, Loss: {batch_loss}"
                )

            epoch_loss += batch_loss

        epoch_loss /= num_batches
        epoch_loss_list.append(epoch_loss)

    model_saver(model, os.path.join(save_path, "checkpoints", "model.pth"))
    draw_loss(
        epoch_loss_list,
        os.path.join(save_path, "logs", "loss.png"),
        title="RNN Loss",
    )


def model_saver(model: nn.Module, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)
    logging.info(f"Model saved to {path}")


def train_model_mlp(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    dataloader: DataLoader,
    loss_fn: nn.Module,
    num_epochs: int = 1000,
    print_every: int = 100,
    save_path: str = "./outputs",
):
    model.train()
    num_batches = len(dataloader)
    epoch_loss_list = []
    for epoch in range(num_epochs):
        epoch_loss = 0
        for batch_idx, data in enumerate(dataloader):
            inputs, labels = data["inputs"], data["labels"]
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_loss = round(loss.item() * 1000, 3)

            if batch_idx % print_every == 0:
                logging.info(
                    f"Epoch {epoch}/{num_epochs}, Batch {batch_idx}/{num_batches}, Loss: {batch_loss}"
                )

            epoch_loss += batch_loss

        epoch_loss /= num_batches
        epoch_loss_list.append(epoch_loss)

    model_saver(model, os.path.join(save_path, "checkpoints", "model.pth"))
    draw_loss(
        epoch_loss_list,
        os.path.join(save_path, "logs", "loss.png"),
        title="MLP Loss",
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    model = nn.Linear(10, 10).to("cuda")
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    class SimpleDataset(Dataset):
        def __init__(self, inputs, labels):
            self.inputs = inputs
            self.labels = labels

        def __len__(self):
            return len(self.inputs)

        def __getitem__(self, idx):
            return {"inputs": self.inputs[idx], "labels": self.labels[idx]}

    dataset = SimpleDataset(
        torch.randn(100, 10).to("cuda"), torch.randn(100, 10).to("cuda")
    )
    dataloader = DataLoader(dataset, batch_size=10, shuffle=True)
    loss_fn = nn.MSELoss()
    train_model_mlp(
        model,
        optimizer,
        dataloader,
        loss_fn,
        num_epochs=10,
        print_every=1,
    )
