import torch
import torch.nn as nn


class GRUCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        combined_size = self.input_size + self.hidden_size
        self.linear_update_gate = nn.Sequential(
            nn.Linear(combined_size, self.hidden_size),
            nn.Sigmoid(),
        )
        self.linear_reset_gate = nn.Sequential(
            nn.Linear(combined_size, self.hidden_size),
            nn.Sigmoid(),
        )
        self.linear_new_gate = nn.Sequential(
            nn.Linear(combined_size, self.hidden_size), nn.Tanh()
        )

    def forward(self, input, hidden=None):
        B, _ = input.shape  # (B, C)
        if hidden is None:  # (B, H)
            hidden = self.init_state(B, input.device)
        combined = torch.cat((input, hidden), dim=-1)  # (B, C + H)
        z_t = self.linear_update_gate(combined)  # (B, H)
        r_t = self.linear_reset_gate(combined)  # (B, H)
        h_c = self.linear_new_gate(torch.cat((input, r_t * hidden), dim=-1))  # (B, H)
        h_t = (1 - z_t) * hidden + z_t * h_c
        return h_t

    def init_weight(self):
        # Tanh/Sigmoid vanishing gradients can be solved with Xavier initialization
        stdv = 1.0 / torch.sqrt(torch.tensor(self.hidden_size))
        for weight in self.parameters():
            nn.init.uniform_(weight, -stdv, stdv)

    def init_state(self, B, device):
        state = torch.zeros((B, self.hidden_size), device=device)
        return state


class CharGRU(nn.Module):
    def __init__(
        self, vocab_size, embedding_dim=32, hidden_size=64, num_layers=3, dropout=0.0
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        # multi-layer GRU
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            cur_input_size = embedding_dim if i == 0 else hidden_size
            self.layers.append(GRUCell(cur_input_size, hidden_size))
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None
        self.mlp_ho = nn.Linear(hidden_size, vocab_size)

    def forward(self, input):
        embedded = self.embedding(input)  # (B, T) -> (B, T, E)
        B, T, C = embedded.shape
        embedded = embedded.transpose(0, 1)  # (T, B, C)
        hidden = None
        current_input = embedded  # (T, B, C)
        for i, gru in enumerate(self.layers):
            layer_output = []
            for t in range(T):
                hidden = gru(current_input[t], hidden)
                layer_output.append(hidden)
            current_input = torch.stack(layer_output, dim=0)  # (T, B, H)
            hidden = None

            if self.dropout and i < len(self.layers) - 1:
                current_input = self.dropout(current_input)
        outputs = self.mlp_ho(current_input)  # (T, B, V)
        return outputs.transpose(0, 1)  # (B, T, V)


if __name__ == "__main__":
    model = CharGRU(vocab_size=10, embedding_dim=32, hidden_size=64).to("cuda")
    input = torch.randint(0, 10, (10, 5)).to("cuda")
    output = model(input)
    print(output.shape)
