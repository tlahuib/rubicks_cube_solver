import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from torch import tensor


class Head(nn.Module):
    """One head of self attention"""

    def __init__(self, n_embed: int, head_size: int, dropout: float) -> None:
        super().__init__()
        # Queries, Keys and Values
        self.query = nn.Linear(n_embed, head_size, bias=False)
        self.key = nn.Linear(n_embed, head_size, bias=False)
        self.value = nn.Linear(n_embed, head_size, bias=False)

        # Other variables/layers
        self.dropout = nn.Dropout(dropout)
        # self.register_buffer('tril', torch.tril(torch.ones(context_size, context_size)))

    def forward(self, x: tensor):
        # Input of size (batch, context_size, n_embed)
        # Output of size (batch, n_embed, head_size)

        Q = self.query(x)   # (B, C, H)
        K = self.key(x)     # (B, C, H)

        # Compute attention scores (Affinities)
        W = Q @ K.transpose(-2, -1) * K.shape[-1]**-0.5 # (B, C, H) @ (B, H, C) -> (B, C, C)
        # W = W.masked_fill(self.tril == 0, float('-inf')) # (B, T, T)
        W = F.softmax(W, dim=-1) # (B, C, C)
        W = self.dropout(W)

        # Weighted aggregation
        V = self.value(x) # (B, C, H)
        out = W @ V # (B, C, C) @ (B, C, H) -> (B, C, H)
        return out
    
class MultiHeadAttention(nn.Module):
    """Multiple heads of self-attention in parallel"""

    def __init__(self, n_embed: int, head_size: int, n_heads: int, dropout: float) -> None:
        super().__init__()
        self.heads = nn.ModuleList([Head(n_embed, head_size, dropout) for _ in range(n_heads)])
        self.proj = nn.Linear(head_size * n_heads, n_embed)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedFoward(nn.Module):
    """ a simple linear layer followed by a non-linearity """

    def __init__(self, n_input: int, dropout: float, n_output: int = None) -> None:
        super().__init__()
        if not n_output:
            n_output = n_input
        self.net = nn.Sequential(
            nn.Linear(n_input, 4 * n_input),
            nn.ReLU(),
            nn.Linear(4 * n_input, n_output),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self, n_embed: int, n_heads: int, dropout: float) -> None:
        # n_embed: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        head_size = max(n_embed // n_heads, 1)
        self.sa = MultiHeadAttention(n_embed, head_size, n_heads, dropout)
        self.ffwd = FeedFoward(n_embed, dropout)
        self.ln1 = nn.LayerNorm(n_embed)
        self.ln2 = nn.LayerNorm(n_embed)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x
    

class Preprocessor(nn.Module):

    def __init__(self,  location_size: int, color_size: int, n_embed: int, n_heads: int, n_layers: int, dropout: float) -> None:
        super().__init__()

        # Create embedding for attention
        self.loc_embedding = nn.Embedding(location_size, n_embed)
        self.color_embedding = nn.Embedding(color_size, n_embed)

        # Transformer layers
        self.blocks = nn.Sequential(
            *[Block(n_embed, n_heads, dropout) for _ in range(n_layers)]
        )

    def forward(self, loc_embed: tensor, color_embed: tensor):
        x = torch.cat((
            self.loc_embedding(loc_embed), 
            self.color_embedding(color_embed)
        ), 1)  # (B, C, E)
        x = self.blocks(x) # (B, C, E)
        return x


class Transformer(nn.Module):

    def __init__(self, location_size: int, color_size: int, n_embed: int, n_heads: int, n_layers: int, dropout: float) -> None:
        super().__init__()
        
        self.shared_layers = Preprocessor(location_size, color_size, n_embed, n_heads, n_layers, dropout) # (B, C, E)

        self.policy_layers = nn.Sequential(
            nn.LayerNorm(n_embed), # (B, C, E)
            nn.Linear(n_embed, 1), # (B, C, E)
            nn.Flatten(), # (B, C)
            FeedFoward(location_size + color_size, 0.2, 12) # (B, 12)
        )

        self.value_layers = nn.Sequential(
            nn.LayerNorm(n_embed), # (B, C, E)
            nn.Linear(n_embed, 1), # (B, C, 1)
            nn.Flatten(), # (B, C)
            FeedFoward(location_size + color_size, 0.2, 1), # (B, 1)
        )

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)


    def forward(self, loc_embed: tensor, color_embed: tensor):

        preprocessed_x = self.shared_layers(loc_embed, color_embed)

        policy_logits = self.policy_layers(preprocessed_x)

        value_logits = self.value_layers(preprocessed_x)

        return policy_logits, value_logits
