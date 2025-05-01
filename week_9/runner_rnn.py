import logging

import torch
from cores.generator import generate_text_rnn
from cores.trainer import train_model_rnn
from models.CharRNN import CharRNN
from torch import nn
from utils.load_training_dataset import load_training_dataset
from utils.Tokenizer import CharacterTokenizer


def run(device: str = "cuda" if torch.cuda.is_available() else "cpu"):
    dataset = load_training_dataset()["whole_func_string"]

    tokenizer = CharacterTokenizer(dataset)

    logging.info(f"Shape of dataset: {len(dataset)}")

    model = CharRNN(vocab_size=len(tokenizer)).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)

    loss_fn = nn.CrossEntropyLoss()

    train_model_rnn(
        model,
        optimizer,
        dataset,
        tokenizer,
        loss_fn,
        num_epochs=10,
        print_every=10,
        save_path="./outputs",
    )

    generated_text = generate_text_rnn(
        model,
        tokenizer,
        model_path="./outputs/checkpoints/model.pth",
    )
    print("".join(generated_text))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run()
