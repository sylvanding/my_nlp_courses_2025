import torch
import torch.nn as nn
import torch.nn.functional as F


class LSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        combined_size = self.input_size + self.hidden_size
        # 4 * hidden_size: forget gate, input gate, candidate cell state, output gate
        self.linear = nn.Linear(combined_size, 4 * self.hidden_size)
        self.init_weight()

    def forward(self, input, state=None):
        B, _ = input.shape  # (B, C)
        if state is None:
            state = self.init_state(B, input.device)
        cs, hs = state  # (B, H)
        combined = torch.cat((input, hs), dim=-1)  # (B, C + H)
        gates_linear = self.linear(combined)  # (B, 4H)
        # forget gate, input gate, candidate cell state, output gate
        f, i, g, o = gates_linear.chunk(4, dim=-1)  # ((B, H) * 4)
        forget_gate = F.sigmoid(f)
        input_gate = F.sigmoid(i)
        candidate_cs = F.tanh(g)
        output_gate = F.sigmoid(o)
        cs = forget_gate * cs + input_gate * candidate_cs
        hs = output_gate * F.tanh(cs)
        return cs, hs

    def init_weight(self):
        # Tanh/Sigmoid vanishing gradients can be solved with Xavier initialization
        stdv = 1.0 / torch.sqrt(torch.tensor(self.hidden_size))
        for weight in self.parameters():
            nn.init.uniform_(weight, -stdv, stdv)

        # set bias of forget gate to 1 to keep the cell state
        if hasattr(self.linear, "bias") and self.linear.bias is not None:
            self.linear.bias.data[: self.hidden_size] = 1.0

    def init_state(self, B, device):
        cs = torch.zeros((B, self.hidden_size), device=device)
        hs = torch.zeros((B, self.hidden_size), device=device)
        return cs, hs


class CharLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim=32, hidden_size=64):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = LSTMCell(embedding_dim, hidden_size)
        self.mlp_ho = nn.Linear(hidden_size, vocab_size)

    def forward(self, input):
        embedded = self.embedding(input)  # (B, T) -> (B, T, E)
        B, T, C = embedded.shape
        embedded = embedded.transpose(0, 1)  # (T, B, C)
        outputs = []
        state = None
        for i in range(T):
            state = self.lstm(embedded[i], state)
            outputs.append(state[1])
        outputs = torch.stack(outputs, dim=0)  # (T, B, H)
        outputs = self.mlp_ho(outputs)  # (T, B, V)
        return outputs.transpose(0, 1)  # (B, T, V)


if __name__ == "__main__":
    model = CharLSTM(vocab_size=10, embedding_dim=32, hidden_size=64).to("cuda")
    input = torch.randint(0, 10, (10, 5)).to("cuda")
    output = model(input)
    print(output.shape)
