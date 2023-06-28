import models
import numpy as np
import torch
import utils


def initialize_model(n_embed: int = 3, n_heads: int = 6, n_layers: int = 6, dropout: float = 1):

    _, loc_embed, color_embed = utils.receiveRandomCubes(np.array([0]))

    model = models.Transformer(len(loc_embed[0]), len(color_embed[0]), n_embed, n_heads, n_layers, dropout)

    return model

def train(batch_size, u_bound: int):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = initialize_model().to(device)

    cube, loc_embed, color_embed = utils.receiveRandomCubes(np.random.randint(1, u_bound, batch_size), device)
    output = model(loc_embed, color_embed)

    return output

if __name__ ==  '__main__':
    print(train(2, 2))


