import os
import subprocess
import numpy as np
import torch
from torch import tensor

from models import MLP

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

if __name__ == '__main__':
    # Setup the inference model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MLP(282, 12, 6).to(device)
    model.load_state_dict(torch.load(f'{wd}/constructor/solves/v8/Checkpoints/model.pth'))
    model.eval()

    # Get a cube to solve
    # nMoves = np.random.randint(1, 25)
    nMoves = 5
    originalCube, originalEmbed = receiveRandomCube(nMoves)

    if all(originalEmbed[-6:] == [1, 1, 1, 1, 1, 1]):
        print('Original cube is already solved')

    # Initialize queue
    cubeQueue = [[originalCube]]
    probQueue = [[0]]
    positionDict = {''.join(originalEmbed.astype(str)): 0}

    # Try moves until solved
    while True:
        # Get the position to expand
        while True:
            if not probQueue: break
            if not probQueue[-1]:
                probQueue = probQueue[:-1]
                cubeQueue = cubeQueue[:-1]
            else: break
        if not probQueue: break
        foundSolve = False
        queuePos = -1
        index = np.argmin(probQueue[queuePos])
        cubes, embeds, diff = getPossibleMoves(cubeQueue[-1][index], device=device)
        probs = model(diff).cpu().detach().numpy().T
        
        probBuffer = []
        cubeBuffer = []
        for i in range(18):
            try:
                positionDict[''.join(embeds[i].astype(str))]
            except KeyError:
                positionDict[''.join(embeds[i].astype(str))] = len(probQueue)
                probBuffer.append(probs[0][i])
                cubeBuffer.append(cubes[i])

                if all(embeds[i][-6:] == [1, 1, 1, 1, 1, 1]):
                    print(f'Solved cube in {len(probQueue)} moves.')
                    nMoves = len(probQueue)
                    probQueue = probQueue[:-2]
                    cubeQueue = cubeQueue[:-2]
                    foundSolve = True
                    break
        
        if foundSolve: continue

        if probBuffer and len(probQueue) < nMoves:
            probQueue.append(probBuffer)
            cubeQueue.append(cubeBuffer)
            queuePos = -2

        probQueue[queuePos].pop(index)
        cubeQueue[queuePos].pop(index)

        if not probQueue: break



    


        

