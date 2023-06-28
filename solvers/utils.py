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
    embeds = []
    diff = np.array([])
    for item in rawSplit:
        itemSplit = item.split("|")
        cubes.append(itemSplit[0])
        embeds.append(np.array(itemSplit[1].split(' ')).astype(int))

        try:
            diff = np.append(diff, np.array(itemSplit[2].split(' ')).astype(int))
        except IndexError:
            pass
    
    diff = diff.reshape((-1, 282))
    diff = tensor(diff, dtype=torch.float, device=device, requires_grad=False)

    return cubes, embeds, diff


def receiveRandomCube(nMoves: int):
    raw = runGo('receiveRandomCube', str(nMoves))
    cubes, embeds, _ = decodeCubes(raw)
    return cubes[0], embeds[0]

def getPossibleMoves(cube: str, **kwargs):
    raw = runGo('getPossibleMoves', cube)
    cubes, embeds, diff = decodeCubes(raw, **kwargs)
    return cubes, embeds, diff