import argparse
import json
import logging
import os
import sys

#import sagemaker_containers
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
from torch import tensor
from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))


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

    def __init__(self, context_size: int, n_embed: int, n_heads: int, n_layers: int, dropout: float) -> None:
        super().__init__()
        # Create embedding for attention
        self.embedding = nn.Embedding(context_size, n_embed)

        # Transformer layers
        self.blocks = nn.Sequential(*[Block(n_heads, n_embed, context_size, dropout) for _ in range(n_layers)])
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

    def forward(self, embed_x: tensor) -> tuple[tensor, tensor]:
        # idx and targets are both (B, T) tensor of integers

        x = self.embedding(x) # (B, T, C)
        x = self.blocks(x) # (B, T, C)
        x = self.ln_f(x) # (B, T, C)
        pred = torch.squeeze(self.lm_head(x)) # (B, T)
        pred = torch.squeeze(self.ffwd_result(pred)) # (B)

        return pred
    

class SolvesDataset(Dataset):
    def __init__(self, file) -> None:
        super().__init__()
        self.file = file

    def __len__(self):
        try:
            return self.len
        except AttributeError:
            with open(self.file) as f:
                self.len = sum(1 for _ in f)
            return self.len

    def __getitem__(self, idx):
        row = np.loadtxt(self.file, delimiter=',', dtype=int, max_rows=1, skiprows=idx)

        X = torch.as_tensor(row[:-1], dtype=torch.long)
        y = torch.as_tensor(row[-1], dtype=torch.float)

        return X, y


def _get_feature_length(training_dir):
    dataset = SolvesDataset(training_dir, train=True)
    sample, _ = dataset[0]
    return len(sample)


def _get_train_data_loader(batch_size, training_dir, **kwargs):
    logger.info("Get train data loader")
    dataset = SolvesDataset(training_dir, train=True)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        **kwargs
    )


def _get_test_data_loader(test_batch_size, testing_dir, **kwargs):
    logger.info("Get test data loader")
    dataset = SolvesDataset(testing_dir, train=False)
    return DataLoader(
        dataset,
        batch_size=test_batch_size,
        shuffle=True,
        **kwargs
    )


def train(args):
    use_cuda = args.num_gpus > 0
    logger.debug("Number of gpus available - {}".format(args.num_gpus))
    kwargs = {"num_workers": 1, "pin_memory": True} if use_cuda else {}
    device = torch.device("cuda" if use_cuda else "cpu")

    # set the seed for generating random numbers
    torch.manual_seed(args.seed)
    if use_cuda:
        torch.cuda.manual_seed(args.seed)

    train_loader = _get_train_data_loader(args.batch_size, args.data_dir, **kwargs)
    test_loader = _get_test_data_loader(args.test_batch_size, args.test_data_dir, **kwargs)

    logger.debug(
        "Processes {}/{} ({:.0f}%) of train data".format(
            len(train_loader.sampler),
            len(train_loader.dataset),
            100.0 * len(train_loader.sampler) / len(train_loader.dataset),
        )
    )

    logger.debug(
        "Processes {}/{} ({:.0f}%) of test data".format(
            len(test_loader.sampler),
            len(test_loader.dataset),
            100.0 * len(test_loader.sampler) / len(test_loader.dataset),
        )
    )

    
    model_args = dict(
        context_size = _get_feature_length(args.data_dir),
        n_embed = args.n_embed,
        n_heads = args.n_heads,
        n_layers = args.n_layers,
        dropout = args.dropout
    )
    model = Transformer(**model_args).to(device)
    model = nn.DataParallel(model)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(1, args.epochs + 1):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader, 1):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = F.mse_loss(output, target)
            loss.backward()
            optimizer.step()
            if batch_idx % args.log_interval == 0:
                logger.info(
                    "Train Epoch: {} [{}/{} ({:.0f}%)] Loss: {:.6f}".format(
                        epoch,
                        batch_idx * len(data),
                        len(train_loader.sampler),
                        100.0 * batch_idx / len(train_loader),
                        loss.item(),
                    )
                )
        test(model, test_loader, device)
    save_model(model, args.model_dir)


def test(model, test_loader, device):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            pred = model(data)
            test_loss += F.mse_loss(pred, target, size_average=False).item()  # sum up batch loss

    test_loss /= len(test_loader.dataset)
    logger.info("Test set: Average loss: {:.4f}\n".format(test_loss))


def model_fn(model_dir, model_args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.nn.DataParallel(Transformer(**model_args))
    with open(os.path.join(model_dir, "model.pth"), "rb") as f:
        model.load_state_dict(torch.load(f))
    return model.to(device)


def save_model(model, model_dir):
    logger.info("Saving the model.")
    path = os.path.join(model_dir, "model.pth")
    # recommended way from http://pytorch.org/docs/master/notes/serialization.html
    torch.save(model.cpu().state_dict(), path)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Data and model checkpoints directories
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        metavar="N",
        help="input batch size for training (default: 64)",
    )
    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=1000,
        metavar="N",
        help="input batch size for testing (default: 1000)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        metavar="N",
        help="number of epochs to train (default: 10)",
    )
    parser.add_argument(
        "--lr", type=float, default=1e-4, metavar="LR", help="learning rate (default: 1e-4)"
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
        default=8,
        metavar="N",
        help="how many embeddings per feature",
    )
    parser.add_argument(
        "--n-heads",
        type=int,
        default=100,
        metavar="N",
        help="how many parallel heads of self attention",
    )
    parser.add_argument(
        "--n-layers",
        type=int,
        default=100,
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
    parser.add_argument("--hosts", type=list, default=json.loads(os.environ["SM_HOSTS"]))
    parser.add_argument("--current-host", type=str, default=os.environ["SM_CURRENT_HOST"])
    parser.add_argument("--model-dir", type=str, default=os.environ["SM_MODEL_DIR"])
    parser.add_argument("--data-dir", type=str, default=os.environ["SM_CHANNEL_TRAINING"])
    parser.add_argument("--test-data-dir", type=str, default=os.environ["SM_CHANNEL_TESTING"])
    parser.add_argument("--num-gpus", type=int, default=os.environ["SM_NUM_GPUS"])

    train(parser.parse_args())

