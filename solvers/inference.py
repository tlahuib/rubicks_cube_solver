import numpy as np
import torch
import utils
from models import MLP

if __name__ == '__main__':
    # Setup the inference model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MLP(282, 12, 6).to(device)
    model.load_state_dict(torch.load(f'{utils.wd}/constructor/solves/v8/Checkpoints/model.pth'))
    model.eval()

    # Get a cube to solve
    # nMoves = np.random.randint(1, 25)
    nMoves = 5
    originalCube, originalEmbed = utils.receiveRandomCube(nMoves)

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
        cubes, embeds, diff = utils.getPossibleMoves(cubeQueue[-1][index], device=device)
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



    


        

