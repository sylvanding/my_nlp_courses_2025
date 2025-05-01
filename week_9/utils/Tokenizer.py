class CharacterTokenizer:
    def __init__(self, dataset):
        self.chars = set()
        self.char_to_index = {}
        self.index_to_char = {}
        self.bos_token = "<bos>"
        self.eos_token = "<eos>"
        self.bos_index = 0
        self.eos_index = 1

        self.build_vocab(dataset)

    def build_vocab(self, dataset):
        self.chars.update(char for char in "".join(dataset))
        chars_sorted = sorted(self.chars)
        self.char_to_index[self.bos_token] = self.bos_index
        self.char_to_index[self.eos_token] = self.eos_index
        self.char_to_index.update({char: i + 2 for i, char in enumerate(chars_sorted)})
        self.index_to_char = {i: char for char, i in self.char_to_index.items()}

    def encode(self, text):
        return [self.char_to_index[char] for char in text]

    def decode(self, indices):
        if isinstance(indices, int):
            return self.index_to_char[indices]
        elif isinstance(indices, list):
            return [self.index_to_char[i] for i in indices]
        else:
            raise ValueError(f"Invalid input type: {type(indices)}")

    def __len__(self):
        return len(self.char_to_index)


if __name__ == "__main__":
    dataset = ["hello", "world", "hello", "world"]
    tokenizer = CharacterTokenizer(dataset)
    print(tokenizer.encode("hello"))
    print(tokenizer.decode([0, 1, 2, 3]))
