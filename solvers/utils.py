import os
import subprocess
import numpy as np
import torch
from torch import tensor

wd = os.getcwd()

def runGo(function: str, *args):
    result = subprocess.run(['go', 'run', f'{wd}/solvers/wrapper.go', function, *args], stdout=subprocess.PIPE)

    return result.stdout.decode('utf-8')

def decodeCubes(raw: str, device: str = 'cpu'):
    rawSplit = raw.replace('[', '').replace(']', '').split('\n')[:-1]

    cubes = []
    loc_embeds = []
    color_embeds = []
    for item in rawSplit:
        itemSplit = item.split("|")
        cubes.append(itemSplit[0])
        loc_embeds.append(itemSplit[1].split(' '))
        color_embeds.append(itemSplit[2].split(' '))

    loc_embeds = tensor(np.array(loc_embeds).astype(int), dtype=torch.long, device=device, requires_grad=False)
    color_embeds = tensor(np.array(color_embeds).astype(int), dtype=torch.long, device=device, requires_grad=False)

    return cubes, loc_embeds, color_embeds


def receiveRandomCubes(nMoves: np.array, device: str = 'cpu'):
    raw = runGo('receiveRandomCubes', *nMoves.astype(str))
    cubes, loc_embeds, color_embeds = decodeCubes(raw, device)
    return cubes, loc_embeds, color_embeds

if __name__ == '__main__':
    cubes, loc_embeds, color_embeds = receiveRandomCubes(np.array([1, 1, 5]))
    print(loc_embeds)