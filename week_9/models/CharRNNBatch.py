import torch
import torch.nn as nn


class RNNCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        combined_size = self.input_size + self.hidden_size
        self.i2h = nn.Sequential(
            nn.Linear(combined_size, hidden_size),
            nn.SiLU(),
            nn.LayerNorm(hidden_size),
        )

    def forward(self, input, hidden=None):
        B, T, C = input.shape
        input = input.transpose(0, 1)  # (T, B, C)
        if hidden is None:
            hidden = torch.zeros((B, self.hidden_size), device=input.device)
        outputs = []
        for i in range(T):
            combined = torch.cat((input[i], hidden), dim=-1)  # (B, H + C)
            hidden = self.i2h(combined)  # (B, H)
            outputs.append(hidden)
        outputs = torch.stack(outputs, dim=0)  # (T, B, H)
        return outputs.transpose(0, 1)  # (B, T, H)


class CharRNNBatch(nn.Module):
    def __init__(self, vocab_size, embedding_dim=32, hidden_size=64):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.dropout = nn.Dropout(0.1)
        # multi-layer RNN
        self.rnn1 = RNNCell(embedding_dim, hidden_size)
        self.rnn2 = RNNCell(hidden_size, hidden_size)
        self.mlp_ho = nn.Linear(hidden_size, vocab_size)

    def forward(self, input):
        embedded = self.embedding(input)  # (B, T) -> (B, T, E)
        hidden = self.dropout(self.rnn1(embedded))  # (B, T, H)
        hidden = self.dropout(self.rnn2(hidden))  # (B, T, H)
        outputs = self.mlp_ho(hidden)  # (B, T, V)
        return outputs


if __name__ == "__main__":
    model = CharRNNBatch(vocab_size=10, embedding_dim=32, hidden_size=64).to("cuda")
    input = torch.randint(0, 10, (10, 5)).to("cuda")
    output = model(input)
    print(output.shape)
