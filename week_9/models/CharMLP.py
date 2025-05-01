import torch
from torch import nn


class CharMLP(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 32,
        hidden_dim: int = 512,
        context_length: int = 10,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim * context_length, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, vocab_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embedding(x)  # (batch_size, context_length, embedding_dim)
        x = x.view(x.size(0), -1)  # (batch_size, context_length * embedding_dim)
        return self.mlp(x)  # (batch_size, vocab_size)


if __name__ == "__main__":
    model = CharMLP(vocab_size=20)
    x = torch.randint(0, 20, (1, 10))
    print(model(x).shape)
