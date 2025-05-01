import logging

import torch
from cores.generator import generate_text_mlp
from cores.trainer import train_model_mlp
from models.CharMLP import CharMLP
from torch import nn
from torch.utils.data import DataLoader
from utils.load_training_dataset import load_training_dataset
from utils.Tokenizer import CharacterTokenizer
from utils.training_data_creator import autoregressive_training_data_creator


def run(device: str = "cuda" if torch.cuda.is_available() else "cpu"):
    dataset = load_training_dataset()

    tokenizer = CharacterTokenizer(dataset["whole_func_string"])
    tokenized_dataset = dataset.map(
        lambda x: autoregressive_training_data_creator(x, tokenizer),
        batched=True,
        remove_columns=dataset.column_names,
        # load_from_cache_file=False,
    )
    tokenized_dataset.set_format(type="torch", device=device)

    logging.info(
        f"Shape of tokenized dataset: {tokenized_dataset['inputs'].shape}, {tokenized_dataset['labels'].shape}"
    )  # torch.Size([56006, 10]), torch.Size([56006])

    dataloader = DataLoader(tokenized_dataset, batch_size=512, shuffle=True)

    model = CharMLP(vocab_size=len(tokenizer)).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    loss_fn = nn.CrossEntropyLoss()

    train_model_mlp(
        model,
        optimizer,
        dataloader,
        loss_fn,
        num_epochs=100,
        print_every=10,
        save_path="./outputs",
    )

    generated_text = generate_text_mlp(
        model,
        tokenizer,
        torch.zeros((1, 10), device=device, dtype=torch.long),
        # model_path="./outputs/checkpoints/model.pth",
    )
    print("".join(generated_text))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run()
