import logging
import os

import torch
from torch.nn import functional as F
from utils.Tokenizer import CharacterTokenizer


@torch.no_grad()
def generate_text_rnn_batch(
    model: torch.nn.Module,
    tokenizer: CharacterTokenizer,
    context: torch.Tensor,
    sequence_length: int = 10,
    max_length: int = 100,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    model_path: str | None = None,
):
    if model_path is not None:
        assert os.path.exists(model_path), f"Model path {model_path} does not exist"
        logging.info(f"Loading model from {model_path}")
        model.load_state_dict(
            torch.load(model_path, map_location=device, weights_only=True)
        )
    model.eval()
    context = context.to(device)
    output = [] if context is None else context.tolist()[0]
    for _ in range(max_length):
        logits = model(context[..., -sequence_length:])[:, -1, :]
        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        output.append(next_token.item())
        if next_token.item() == tokenizer.eos_index:
            break
        context = torch.cat([context, next_token], dim=-1)

    return tokenizer.decode(output)


@torch.no_grad()
def generate_text_rnn(
    model: torch.nn.Module,
    tokenizer: CharacterTokenizer,
    max_length: int = 100,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    model_path: str | None = None,
):
    if model_path is not None:
        assert os.path.exists(model_path), f"Model path {model_path} does not exist"
        logging.info(f"Loading model from {model_path}")
        model.load_state_dict(
            torch.load(model_path, map_location=device, weights_only=True)
        )
    model.eval()
    outputs = []
    next_token = torch.tensor((tokenizer.encode("d")[0],), device=device)
    outputs.append(next_token.item())
    hidden = None
    for _ in range(max_length):
        output, hidden = model(next_token, hidden)
        probs = F.softmax(output, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1).squeeze(0)
        outputs.append(next_token.item())
        if next_token.item() == tokenizer.eos_index:
            break

    return tokenizer.decode(outputs)


@torch.no_grad()
def generate_text_mlp(
    model: torch.nn.Module,
    tokenizer: CharacterTokenizer,
    context: torch.Tensor,  # shape: (1, 10)
    max_length: int = 100,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    model_path: str | None = None,
):
    if model_path is not None:
        assert os.path.exists(model_path), f"Model path {model_path} does not exist"
        logging.info(f"Loading model from {model_path}")
        model.load_state_dict(
            torch.load(model_path, map_location=device, weights_only=True)
        )
    model.eval()
    context = context.to(device)
    output = []
    for _ in range(max_length):
        logits = model(context)
        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        output.append(next_token.item())
        if next_token.item() == tokenizer.eos_index:
            break
        context = torch.cat([context[..., 1:], next_token], dim=-1)

    return tokenizer.decode(output)


if __name__ == "__main__":
    pass
