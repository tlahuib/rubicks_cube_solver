import models
import numpy as np
import torch
import utils


def initialize_model(n_embed: int = 3, n_heads: int = 6, n_layers: int = 6, dropout: float = 1):

    _, loc_embed, color_embed = utils.receiveRandomCubes(np.array([0]))

    model = models.Transformer(len(loc_embed[0]), len(color_embed[0]), n_embed, n_heads, n_layers, dropout)

    return model

def compute_advantage(model: torch.nn.Module, cube: str, loc_embed: torch.tensor, color_embed: torch.tensor, device: str = 'cpu'):
    nMoves = []
    # TODO: Paralellize
    for new_cube, new_loc_embed, new_color_embed in zip(*utils.getPossiblePositions(cube, device)):
        nMoves.append(utils.followHeuristic(model, np.array([new_cube]), new_loc_embed[None, :], new_color_embed[None, :], device=device))
    
    return nMoves



def train(batch_size, u_bound: int):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = initialize_model().to(device)

    cube, loc_embed, color_embed = utils.receiveRandomCubes(np.random.randint(1, u_bound, batch_size), device)
    output = model(loc_embed, color_embed)

    return output

if __name__ ==  '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = initialize_model().to(device)

    cube, loc_embed, color_embed = utils.receiveRandomCubes(np.array([1]), device)

    nMoves = compute_advantage(model, cube[0], loc_embed[0], color_embed[0], device)
    print(nMoves)


