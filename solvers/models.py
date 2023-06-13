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
import sqlalchemy as db

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
    

class MLP_layer(nn.Module):
    def __init__(self, size) -> None:
        super().__init__()
        self.linear = nn.Linear(size, size)
        self.activation = nn.Sigmoid()

    def forward(self, x):
        x = self.linear(x)
        x = self.activation(x)
        return x
    

class MLP(nn.Module):
    def __init__(self, n_input, embed_multiplier, n_hidden_layers) -> None:
        super().__init__()
        self.hidden_layer_size = n_input * embed_multiplier
        self.input_layer = nn.Linear(n_input, n_input * embed_multiplier)
        self.hidden_layers = nn.Sequential(*[MLP_layer(self.hidden_layer_size) for _ in range(n_hidden_layers)])
        self.output_layer = nn.Linear(self.hidden_layer_size, 2)
        self.softmax = nn.Softmax(1)

    # def _init_weights(self, module):
    #     if isinstance(module, nn.Linear):
    #         torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    #         if module.bias is not None:
    #             torch.nn.init.zeros_(module.bias)

    def forward(self, embed_x: tensor):

        x = self.input_layer(embed_x)
        x = self.hidden_layers(x)
        x = self.output_layer(x)
        x = self.softmax(x)
        return x


def connectDB(config_dir: str) -> db.Engine:
    with open(config_dir + 'config.json') as f:
        config = json.load(f)

    engine = db.create_engine(
        f"postgresql://{config['user']}:{config['pass']}@{config['host']}:{config['port']}/{config['db']}"
    )
    return engine


def getBatch(engine: db.Engine, batch: int, batch_size: int):
    X, y = [], []

    with engine.connect() as conn:
        
        for row in conn.execute(db.text(f"select embed_diff, (move_diff < 0)::int from rubik.differences where id between {batch * batch_size} and {(batch + 1) * batch_size}")):
            X.append(row[0])
            y.append(row[1])
        
    return X, y


def getShape(engine: db.Engine):
    X, _ = getBatch(engine, 0, 1)

    with engine.connect() as conn:
        for row in conn.execute(db.text(f"select count(*) from rubik.differences")):
            size = row[0]
        for row in conn.execute(db.text(f"select count(*) from rubik.differences where move_diff = -1")):
            ratio = row[0] / size

    return size, ratio, len(X[0])


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")

    # set the seed for generating random numbers
    torch.manual_seed(args.seed)
    
    engine = connectDB(args.data_dir)
    sample_size, ratio, feature_size = getShape(engine)
    weights = tensor([ratio, 1-ratio], dtype=torch.float, device=device)
    n_batches = sample_size // args.batch_size
    logger.info(f"Performing {n_batches} batches per epoch")
    model_args = dict(
        context_size = feature_size,
        n_embed = args.n_embed,
        n_heads = args.n_heads,
        n_layers = args.n_layers,
        dropout = args.dropout
    )
    logger.info(f"Creating transformer with hyperparameters:\n{model_args}")
    # model = Transformer(**model_args).to(device)
    model = MLP(model_args['context_size'], args.n_embed, args.n_layers).to(device)
    # model = nn.DataParallel(model)

    if args.load:
        path = args.model_dir + args.model_file
        logger.info(f"Loading model from {path}")
        model.load_state_dict(torch.load(path))

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # rng = np.random.default_rng()

    total_steps = args.epochs * n_batches
    tic = time()
    for epoch in range(1, args.epochs + 1):
        logger.info(f"Starting epoch {epoch}")

        model.train()
        X_test = tensor([], dtype=torch.float, device=device)
        y_test = tensor([], dtype=torch.long, device=device)

        unweighted_loss = 0
        weighted_loss = 0
        n_loss = 0
        for step in range(n_batches):

            # Read data
            X, y = getBatch(engine, step, args.batch_size)
            # shuffle = rng.choice(len(y), len(y), replace=False)
            X = tensor(X, dtype=torch.float, device=device)
            y = tensor(y, dtype=torch.long, device=device)

            # Train test split
            n = int(0.9 * len(y))
            X_train, y_train = X[:n], y[:n]
            X_test = torch.cat((X_test, X[n:]))
            y_test = torch.cat((y_test, y[n:]))

            optimizer.zero_grad()
            output = model(X_train)
            loss = F.cross_entropy(output, y_train, weight=weights)
            unweighted_loss += F.cross_entropy(output, y_train, reduction="sum")
            weighted_loss += F.cross_entropy(output, y_train, reduction="sum", weight=weights)
            n_loss += len(y_train)
            loss.backward()
            optimizer.step()
            if step % args.log_interval == 0:
                current_steps = (epoch - 1) * n_batches + step + 1
                pending_steps = total_steps - current_steps
                time_elapsed = time() - tic
                eta = (time_elapsed / current_steps) * pending_steps
                log_text = "{}: \tEpoch: {} / {} ({:.2f}%) Loss: ({:.2f}, {:.2f})\t Elpased Steps: {} Pending Steps: {}\t ETA: {}\ty: {}  output: {}"
                logger.info(
                    log_text.format(
                        timedelta(seconds=int(time() - tic)),
                        epoch,
                        args.epochs,
                        step / n_batches * 100,
                        unweighted_loss / n_loss,
                        weighted_loss / n_loss,
                        current_steps,
                        pending_steps,
                        timedelta(seconds=int(eta)),
                        y_train[0],
                        output.cpu().detach().numpy()[0]
                    )
                )
                unweighted_loss = 0
                weighted_loss = 0
                n_loss = 0
                test(model, X_test, y_test, args.batch_size, weights)
                X_test = tensor([], dtype=torch.float, device=device)
                y_test = tensor([], dtype=torch.long, device=device)
        save_model(model, args.model_dir, f"checkpoint_{epoch}.pth")

    logger.info(f"Saving the model in {args.model_dir}")
    save_model(model, args.model_dir)


def test(model, X, y, batch_size, weights):
    model.eval()
    unweighted_loss = 0
    weighted_loss = 0
    steps = len(y) // batch_size
    with torch.no_grad():
        for step in range(steps):
            X_test = X[step * batch_size: min((step + 1) * batch_size, len(y))]
            y_test = y[step * batch_size: min((step + 1) * batch_size, len(y))]
            pred = model(X_test)
            unweighted_loss += F.cross_entropy(pred, y_test, reduction="sum").item()
            weighted_loss += F.cross_entropy(pred, y_test, reduction="sum", weight=weights).item()

    unweighted_loss /= len(y)
    weighted_loss /= len(y)

    logger.info("Test Loss: ({:.4f}, {:.4f})\n".format(unweighted_loss, weighted_loss))


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
    torch.save(model.state_dict(), path)
    logger.info(f"Model saved in {path}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Data and model checkpoints directories
    parser.add_argument(
        "--batch-size",
        type=int,
        default=5000,
        metavar="N",
        help="input batch size for training (default: 64)",
    )
    parser.add_argument(
        "--data-loops",
        type=int,
        default=25,
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
    # parser.add_argument(
    #     "--steps",
    #     type=int,
    #     default=1000,
    #     metavar="N",
    #     help="number of steps to train per epoch (default: 10)",
    # )
    parser.add_argument(
        "--lr", type=float, default=1e-5, metavar="LR", help="learning rate (default: 1e-5)"
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
        default=12,
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
    parser.add_argument(
        "--load",
        type=bool,
        default=False, 
        metavar="N",
        help="Whether to load or not a pretrained model (default: False)",
    )
    parser.add_argument(
        "--model-file",
        type=str,
        default="model.pth",
        metavar="N",
        help="Which checkpoint to load.",
    )


    # Container environment
    try:
        parser.add_argument("--hosts", type=list, default=json.loads(os.environ["SM_HOSTS"]))
        parser.add_argument("--current-host", type=str, default=os.environ["SM_CURRENT_HOST"])
        parser.add_argument("--model-dir", type=str, default=os.environ["SM_MODEL_DIR"])
        parser.add_argument("--data-dir", type=str, default=os.environ["SM_CHANNEL_TRAINING"])
        parser.add_argument("--num-gpus", type=int, default=os.environ["SM_NUM_GPUS"])
    except KeyError:
        parser.add_argument("--model-dir", type=str, default=os.getcwd() + "/constructor/solves/v8/Checkpoints/")
        parser.add_argument("--data-dir", type=str, default=os.getcwd() + "/constructor/solves/v8/")
        parser.add_argument("--num-gpus", type=int, default=0)

    train(parser.parse_args())
