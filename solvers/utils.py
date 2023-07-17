import os
import subprocess
import numpy as np
import torch
from torch import tensor

wd = os.getcwd()

solvedLoc = tensor([list(range(20))], dtype=torch.long)
solvedCol = tensor([list(range(48))], dtype=torch.long)

def runGo(function: str, *args):
    result = subprocess.run(['go', 'run', f'{wd}/solvers/wrapper.go', function, *args], stdout=subprocess.PIPE)

    return result.stdout.decode('utf-8')

def decodeCubes(raw: str, device: str = 'cpu'):
    rawSplit = raw.replace('[', '').replace(']', '').split('\n')[:-1]

    cubes = []
    loc_embeds = []
    color_embeds = []
    is_solved = []
    for item in rawSplit:
        itemSplit = item.split("|")
        cubes.append(itemSplit[0])
        loc_embeds.append(itemSplit[1].split(' '))
        color_embeds.append(itemSplit[2].split(' '))
        is_solved.append(itemSplit[3] == 'true')

    cubes = np.array(cubes)
    loc_embeds = tensor(np.array(loc_embeds).astype(int), dtype=torch.long, device=device, requires_grad=False)
    color_embeds = tensor(np.array(color_embeds).astype(int), dtype=torch.long, device=device, requires_grad=False)
    is_solved = np.array(is_solved)

    return cubes, loc_embeds, color_embeds, is_solved


def receiveRandomCubes(nMoves: np.array, device: str = 'cpu'):
    raw = runGo('receiveRandomCubes', *nMoves.astype(str))
    cubes, loc_embeds, color_embeds, is_solved = decodeCubes(raw, device)
    return cubes, loc_embeds, color_embeds, is_solved


def moveCubes(cubes: np.array, moves: tensor, device: str = 'cpu'):
    args = []
    for item in  zip(cubes, moves.cpu().detach().numpy()):
        args.append(item[0] + "|" + str(item[1]))

    raw = runGo('moveCubes', *args)
    cubes, loc_embeds, color_embeds, is_solved = decodeCubes(raw, device)
    return cubes, loc_embeds, color_embeds, is_solved


def getPossiblePositions(cube: str, device: str = 'cpu'):
    raw = runGo('getPossiblePositions', cube)
    cubes, loc_embeds, color_embeds, is_solved = decodeCubes(raw, device)
    return cubes, loc_embeds, color_embeds, is_solved, list(range(len(cubes)))


# def followHeuristic(model, cube: np.array, loc_embeds: tensor, color_embeds: tensor, nMoves: int = 0, maxMoves=50, device: str = 'cpu'):

#     if torch.equal(loc_embeds, solvedLoc.to(device)) and torch.equal(color_embeds, solvedCol.to(device)):
#         return nMoves

#     if nMoves >= maxMoves:
#         return maxMoves

#     probs = model(loc_embeds, color_embeds)
#     move = torch.reshape(torch.argmax(probs), (1, 1))
#     _cube, _loc_embeds, _color_embeds, _is_solved = moveCubes(cube, move[None,:], device)
    
#     return followHeuristic(model, _cube, _loc_embeds, _color_embeds, nMoves +1, device=device)




if __name__ == '__main__':
    cubes, loc_embeds, color_embeds, is_solved = receiveRandomCubes(np.array([1]))
   
    # cubes, loc_embeds, color_embeds, is_solved = moveCubes(cubes, tensor([0, 1, 5], dtype=torch.long, device='cuda'))

    print(is_solved)
    print(loc_embeds)
    print(color_embeds)