import torch
import torch.nn as nn
from torch.nn import functional as F
from torch import tensor

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def get_batch(data: tensor, batch_size: int, device: str = device) -> tuple[tensor, tensor]:
    ids = torch.randint(0, len(data), (batch_size,))
    X = data[ids].T[:-1].T.long()
    y = data[ids].T[-1]
    X, y = X.to(device), y.to(device)
    return X, y


def estimate_loss(model, data: tensor, batch_size: int, steps: int, device: str = device) -> float:
    loss = 0
    for _ in range(steps):
        X, y = get_batch(data, batch_size)
        loss += model.loss(model(X), y)

    loss /= steps
    return loss


class Head(nn.Module):
    """One head of self attention"""

    def __init__(self, head_size: int, n_embed: int, context_size: int, dropout: float) -> None:
        super().__init__()
        # Queries, Keys and Values
        self.query = nn.Linear(n_embed, head_size, bias=False)
        self.key = nn.Linear(n_embed, head_size, bias=False)
        self.value = nn.Linear(n_embed, head_size, bias=False)

        # Other variables/layers
        self.dropout = nn.Dropout(dropout)
        self.register_buffer('tril', torch.tril(torch.ones(context_size, context_size)))

    def forward(self, x: tensor):
        # Input of size (batch, context_size, n_embed)
        # Output of size (batch, context_size, head_size)

        Q = self.query(x)   # (B, T, H)
        K = self.key(x)     # (B, T, H)

        # Compute attention scores (Affinities)
        W = Q @ K.transpose(-2, -1) * K.shape[-1]**-0.5 # (B, T, H) @ (B, H, T) -> (B, T, T)
        W = W.masked_fill(self.tril == 0, float('-inf')) # (B, T, T)
        W = F.softmax(W, dim=-1) # (B, T, T)
        W = self.dropout(W)

        # Weighted aggregation
        V = self.value(x) # (B, T, H)
        out = W @ V # (B, T, T) @ (B, T, H) -> (B, T, H)
        return out
    
class MultiHeadAttention(nn.Module):
    """Multiple heads of self-attention in parallel"""

    def __init__(self, n_heads: int, head_size: int, n_embed: int, context_size: int, dropout: float) -> None:
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size, n_embed, context_size, dropout) for _ in range(n_heads)])
        self.proj = nn.Linear(head_size * n_heads, n_embed)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedFoward(nn.Module):
    """ a simple linear layer followed by a non-linearity """

    def __init__(self, n_embed: int, dropout: float, n_output: int = None) -> None:
        super().__init__()
        if not n_output:
            n_output = n_embed
        self.net = nn.Sequential(
            nn.Linear(n_embed, 4 * n_embed),
            nn.ReLU(),
            nn.Linear(4 * n_embed, n_output),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self, n_heads: int, n_embed: int, context_size: int, dropout: float) -> None:
        # n_embed: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        head_size = max(n_embed // n_heads, 1)
        self.sa = MultiHeadAttention(n_heads, head_size, n_embed, context_size, dropout)
        self.ffwd = FeedFoward(n_embed, dropout)
        self.ln1 = nn.LayerNorm(n_embed)
        self.ln2 = nn.LayerNorm(n_embed)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class Transformer(nn.Module):

    def __init__(self, context_size: int, n_embed: int, n_heads: int, n_layers: int, dropout: float, learning_rate: float = 1e-4) -> None:
        super().__init__()
        # Create embedding for attention
        self.embedding = nn.Embedding(context_size, n_embed)

        # Transformer layers
        self.blocks = nn.Sequential(*[Block(n_heads, n_embed, context_size, dropout) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(n_embed) # final layer norm
        self.lm_head = nn.Linear(n_embed, 1)
        self.ffwd = FeedFoward(context_size, 0.2, 1)

        # Optimizer
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        self.loss = nn.MSELoss()

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, embed_x: tensor) -> tuple[tensor, tensor]:
        # idx and targets are both (B, T) tensor of integers

        x = self.embedding(embed_x) # (B, T, C)
        x = self.blocks(x) # (B, T, C)
        x = self.ln_f(x) # (B, T, C)
        pred = torch.squeeze(self.lm_head(x)) # (B, T)
        pred = torch.squeeze(self.ffwd(pred)) # (B)

        return pred
    
    def fit(self, data: tensor, steps: int, batch_size: int, eval_iter: int = 100) -> None:

        # Split into train and validation data
        n = int(0.9 * len(data))
        train = data[:n]
        validation = data[n:]

        for iter in range(steps):
            self.optimizer.zero_grad(set_to_none=True)

            # Sample a batch
            X, y = get_batch(train, batch_size)

            # Evaluate the loss
            loss = self.loss(self(X), y)
            loss.backward()
            self.optimizer.step()


            # Every once in a while evaluate the loss in validation
            if iter % eval_iter == 0:
                train_loss = estimate_loss(self, train, batch_size, 10)
                val_loss = estimate_loss(self, validation, batch_size, 10)
                print(f"step {iter}: train loss {train_loss:.4f}, val loss {val_loss:.4f}")