import torch
import torch.nn as nn


class RNNCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        combined_size = self.input_size + self.hidden_size
        self.mlp_ih = nn.Linear(combined_size, hidden_size)
        self.activation = nn.Tanh()
        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, input, hidden=None):
        if hidden is None:
            hidden = torch.zeros((1, self.hidden_size), device=input.device)
        combined = torch.cat((input, hidden), dim=-1)  # (1, H + I)
        hidden = self.norm(self.activation(self.mlp_ih(combined)))  # (1, H)
        return hidden


class CharRNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim=32, hidden_size=64):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn_cell = RNNCell(embedding_dim, hidden_size)
        self.mlp_ho = nn.Linear(hidden_size, vocab_size)

    def forward(self, input, hidden=None):
        embedded = self.embedding(input)  # (1) -> (1, E)
        hidden = self.rnn_cell(embedded, hidden)  # (1, H)
        output = self.mlp_ho(hidden)  # (1, V)
        return output, hidden


if __name__ == "__main__":
    rnn = CharRNN(vocab_size=10, embedding_dim=15, hidden_size=20).to("cuda")
    input = torch.randint(0, 10, (1,)).to("cuda")
    output, hidden = rnn(input)
    print(output.shape, hidden.shape)
