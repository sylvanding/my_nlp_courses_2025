import logging

from datasets import DatasetDict

from utils.Tokenizer import CharacterTokenizer


def rnn_batch_text_processor(
    text: str,
    tokenizer: CharacterTokenizer,
    sequence_length: int = 10,
) -> tuple[list[list[int]], list[int]]:
    inputs, labels = [], []
    eos_index = tokenizer.eos_index
    encoded_text = tokenizer.encode(text)
    x = encoded_text + [eos_index]
    if len(x) >= sequence_length:
        for i in range(len(x) - sequence_length):
            inputs.append(x[i : i + sequence_length])
            labels.append(x[i + 1 : i + sequence_length + 1])
    return inputs, labels


def rnn_batch_training_data_creator(
    dataset: DatasetDict | list[str] | str,
    tokenizer: CharacterTokenizer,
    sequence_length: int = 10,
) -> dict[str, list[int]]:
    if not isinstance(dataset, str):
        texts = dataset["whole_func_string"]
    if isinstance(texts, str):
        inputs, labels = rnn_batch_text_processor(texts, tokenizer, sequence_length)
        return {"inputs": inputs, "labels": labels}
    inputs, labels = [], []
    for text in texts:
        input, label = rnn_batch_text_processor(text, tokenizer, sequence_length)
        inputs.extend(input)
        labels.extend(label)
    return {"inputs": inputs, "labels": labels}


def rnn_training_data_creator(
    text: str,
    tokenizer: CharacterTokenizer,
) -> tuple[list[int], list[int]]:
    tokenized = tokenizer.encode(text)
    # inputs = [tokenizer.bos_index] + tokenized
    # labels = tokenized + [tokenizer.eos_index]
    inputs = tokenized
    labels = tokenized[1:] + [tokenizer.eos_index]
    return inputs, labels


def autoregressive_text_processor(
    text: str,
    tokenizer: CharacterTokenizer,
    context_length: int = 10,
) -> tuple[list[list[int]], list[int]]:
    inputs, labels = [], []
    bos_index = tokenizer.bos_index
    eos_index = tokenizer.eos_index
    encoded_text = tokenizer.encode(text)
    x = [bos_index] * context_length + encoded_text + [eos_index]
    for i in range(len(x) - context_length):
        inputs.append(x[i : i + context_length])
        labels.append(x[i + context_length])
    return inputs, labels


def autoregressive_training_data_creator(
    dataset: DatasetDict | list[str] | str,
    tokenizer: CharacterTokenizer,
    context_length: int = 10,
) -> dict[str, list[int]]:
    if not isinstance(dataset, str):
        texts = dataset["whole_func_string"]
    if isinstance(texts, str):
        inputs, labels = autoregressive_text_processor(texts, tokenizer, context_length)
        return {"inputs": inputs, "labels": labels}
    inputs, labels = [], []
    for text in texts:
        input, label = autoregressive_text_processor(text, tokenizer, context_length)
        inputs.extend(input)
        labels.extend(label)
    return {"inputs": inputs, "labels": labels}


if __name__ == "__main__":
    text = "Hello, world!"
    tokenizer = CharacterTokenizer(text)
    inputs, labels = autoregressive_text_processor(text, tokenizer)
    for context, label in zip(inputs, labels):
        print("-->".join(["".join(tokenizer.decode(context)), tokenizer.decode(label)]))

    data = autoregressive_training_data_creator(text, tokenizer)
    print(data)
