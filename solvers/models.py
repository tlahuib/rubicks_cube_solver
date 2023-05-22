import argparse
import json
import logging
import os
import sys
from time import time
from datetime import timedelta

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
from torch import tensor

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))


class Head(nn.Module):
    """One head of self attention"""

    def __init__(self, context_size: int, n_embed: int, head_size: int, dropout: float) -> None:
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

    def __init__(self, context_size: int, n_embed: int, head_size: int, n_heads: int, dropout: float) -> None:
        super().__init__()
        self.heads = nn.ModuleList([Head(context_size, n_embed, head_size, dropout) for _ in range(n_heads)])
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

    def __init__(self, context_size: int, n_embed: int, n_heads: int, dropout: float) -> None:
        # n_embed: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        head_size = max(n_embed // n_heads, 1)
        self.sa = MultiHeadAttention(context_size, n_embed, head_size, n_heads, dropout)
        self.ffwd = FeedFoward(n_embed, dropout)
        self.ln1 = nn.LayerNorm(n_embed)
        self.ln2 = nn.LayerNorm(n_embed)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        # x = x + self.sa(x)
        # x = x + self.ffwd(x)
        return x


class Transformer(nn.Module):

    def __init__(self, context_size: int, n_embed: int, n_heads: int, n_layers: int, dropout: float) -> None:
        super().__init__()
        # Create embedding for attention
        self.embedding = nn.Linear(1, n_embed)

        # Transformer layers
        self.blocks = nn.Sequential(*[Block(context_size, n_embed, n_heads, dropout) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(n_embed) # final layer norm
        self.lm_head = nn.Linear(n_embed, 1)
        self.ffwd_result = FeedFoward(context_size, 0.2, 1)

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, embed_x: tensor):
        # embed_x (B, C)
        x = embed_x[:, :, None] # (B, C, 1)
        x = self.embedding(x) # (B, C, E)
        x = self.blocks(x) # (B, C, E)
        x = self.ln_f(x) # (B, C, E)
        pred = torch.squeeze(self.lm_head(x)) # (B, C)
        pred = torch.squeeze(self.ffwd_result(pred)) # (B)

        return pred


def _get_feature_length(file):
    sample = np.loadtxt(file, delimiter=',', max_rows=1)
    return sample.shape[0]


def _get_batch(X, y, batch_size, rng):
    batch = rng.choice(len(y), batch_size, replace=False)
    return X[batch], y[batch]


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")

    # set the seed for generating random numbers
    torch.manual_seed(args.seed)
    
    features_file = f"{args.data_dir}/solves_features.csv"
    labels_file = f"{args.data_dir}/solves_labels.csv"
    model_args = dict(
        context_size = _get_feature_length(features_file),
        n_embed = args.n_embed,
        n_heads = args.n_heads,
        n_layers = args.n_layers,
        dropout = args.dropout
    )
    logger.info(f"Creating transformer with hyperparameters:\n{model_args}")
    model = Transformer(**model_args).to(device)
    # model = nn.DataParallel(model)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    with open(labels_file) as f:
        len_data = sum(1 for _ in f)
    epoch_size = len_data // args.epochs
    rng = np.random.default_rng()

    total_steps = args.data_loops * args.epochs * args.steps
    tic = time()
    for checkpoint in range(args.data_loops):
        for epoch in range(1, args.epochs + 1):
            logger.info(f"Starting epoch {epoch}")
            
            # Read data
            X = np.loadtxt(features_file, delimiter=',', max_rows=epoch_size, skiprows=(epoch - 1) * epoch_size)
            y = np.loadtxt(labels_file, delimiter=',', max_rows=epoch_size, skiprows=(epoch - 1) * epoch_size)

            shuffle = rng.choice(len(y), len(y), replace=False)
            X = tensor(X[shuffle], dtype=torch.float, device=device)
            y = tensor(y[shuffle], dtype=torch.float, device=device)

            # Train test split
            n = int(0.9 * len(y))

            model.train()
            agg_mse = 0
            agg_l1 = 0
            agg_lens = 0
            for step in range(1, args.steps + 1):
                X_train, y_train = _get_batch(X[:n], y[:n], args.batch_size, rng)
                optimizer.zero_grad()
                output = model(X_train)
                loss = F.mse_loss(output, y_train)
                agg_mse += F.mse_loss(output, y_train, reduction='sum').item()
                agg_l1 += F.l1_loss(output, y_train, reduction="sum").item()
                agg_lens += len(output)
                loss.backward()
                optimizer.step()
                if step % args.log_interval == 0:
                    current_steps = checkpoint * args.epochs * args.steps + (epoch - 1) * args.steps + step
                    pending_steps = total_steps - current_steps
                    time_elapsed = time() - tic
                    eta = (time_elapsed / current_steps) * pending_steps
                    logger.info(
                        "{}: \tEpoch: {} [{}/{} ({:.0f}%)] MSE: {:.2f}, L1: {:.2f}\t Elpased Steps: {} Pending Steps: {}\t ETA: {}".format(
                            timedelta(seconds=int(time() - tic)),
                            epoch,
                            step,
                            args.steps,
                            100.0 * step / args.steps,
                            agg_mse / agg_lens,
                            agg_l1 / agg_lens,
                            current_steps,
                            pending_steps,
                            timedelta(seconds=int(eta))
                        )
                    )
                    agg_mse = 0
                    agg_l1 = 0
                    agg_lens = 0
            test(model, X[n:], y[n:], args.batch_size)
        save_model(model, args.model_dir, f"checkpoint_{checkpoint}.pth")

    logger.info(f"Saving the model in {args.model_dir}")
    save_model(model, args.model_dir)


def test(model, X, y, batch_size):
    model.eval()
    mse_loss = 0
    l1_loss = 0
    steps = len(y) // batch_size
    with torch.no_grad():
        for step in range(steps):
            X_test = X[step * batch_size: min((step + 1) * batch_size, len(y))]
            y_test = y[step * batch_size: min((step + 1) * batch_size, len(y))]
            pred = model(X_test)
            mse_loss += F.mse_loss(pred, y_test, reduction="sum").item()  # sum up batch loss
            l1_loss += F.l1_loss(pred, y_test, reduction="sum").item()  # sum up batch loss

    mse_loss /= len(y)
    l1_loss /= len(y)
    logger.info("Test set: MSE: {:.4f}, L1:{:.4f}\n".format(mse_loss, l1_loss))


def model_fn(model_dir, model_args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")
    # model = torch.nn.DataParallel(Transformer(**model_args))
    model = Transformer(**model_args)
    with open(os.path.join(model_dir, "model.pth"), "rb") as f:
        model.load_state_dict(torch.load(f))
    return model.to(device)


def save_model(model, model_dir, model_name="model.pth"):
    logger.info("Saving the model.")
    path = os.path.join(model_dir, model_name)
    # recommended way from http://pytorch.org/docs/master/notes/serialization.html
    torch.save(model.cpu().state_dict(), path)
    logger.info(f"Model saved in {path}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Data and model checkpoints directories
    parser.add_argument(
        "--batch-size",
        type=int,
        default=200,
        metavar="N",
        help="input batch size for training (default: 64)",
    )
    parser.add_argument(
        "--data-loops",
        type=int,
        default=10,
        metavar="N",
        help="number of times to loop through the whole data (default: 2)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        metavar="N",
        help="number of epochs to train (default: 10)",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=1000,
        metavar="N",
        help="number of steps to train per epoch (default: 10)",
    )
    parser.add_argument(
        "--lr", type=float, default=1e-6, metavar="LR", help="learning rate (default: 1e-6)"
    )
    parser.add_argument("--seed", type=int, default=1, metavar="S", help="random seed (default: 1)")
    parser.add_argument(
        "--log-interval",
        type=int,
        default=100,
        metavar="N",
        help="how many batches to wait before logging training status",
    )

    # Model arguments
    parser.add_argument(
        "--n_embed",
        type=int,
        default=4,
        metavar="N",
        help="how many embeddings per feature",
    )
    parser.add_argument(
        "--n-heads",
        type=int,
        default=6,
        metavar="N",
        help="how many parallel heads of self attention",
    )
    parser.add_argument(
        "--n-layers",
        type=int,
        default=6,
        metavar="N",
        help="how many blocks of multi-headed attention in sequence",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.2,
        metavar="N",
        help="percent of nodes disconected for dropout layers",
    )


    # Container environment
    try:
        parser.add_argument("--hosts", type=list, default=json.loads(os.environ["SM_HOSTS"]))
        parser.add_argument("--current-host", type=str, default=os.environ["SM_CURRENT_HOST"])
        parser.add_argument("--model-dir", type=str, default=os.environ["SM_MODEL_DIR"])
        parser.add_argument("--data-dir", type=str, default=os.environ["SM_CHANNEL_TRAINING"])
        parser.add_argument("--num-gpus", type=int, default=os.environ["SM_NUM_GPUS"])
    except KeyError:
        parser.add_argument("--model-dir", type=str, default=os.getcwd() + "/constructor/solves")
        parser.add_argument("--data-dir", type=str, default=os.getcwd() + "/constructor/solves")
        parser.add_argument("--num-gpus", type=int, default=0)

    train(parser.parse_args())
