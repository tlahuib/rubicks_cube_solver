import os
import subprocess
import numpy as np

wd = os.getcwd()

def runGo(function: str, *args):
    result = subprocess.run(['go', 'run', f'{wd}/solvers/wrapper.go', function, *args], stdout=subprocess.PIPE)

    return result.stdout.decode('utf-8')

def decodeCubes(raw: str):
    rawSplit = raw.replace('[', '').replace(']', '').split('\n')[:-1]

    cubes = []
    embeds = []
    diff = []
    for item in rawSplit:
        itemSplit = item.split("|")
        cubes.append(itemSplit[0])
        embeds.append(np.array(itemSplit[1].split(' ')).astype(int))

        try:
            diff.append(np.array(itemSplit[2].split(' ')).astype(int))
        except IndexError:
            pass


    return cubes, embeds, diff


def receiveRandomCube(nMoves: int):
    raw = runGo('receiveRandomCube', str(nMoves))
    cubes, embeds, _ = decodeCubes(raw)
    return cubes[0], embeds[0]

def getPossibleMoves(cube: str):
    raw = runGo('getPossibleMoves', cube)
    cubes, embeds, diff = decodeCubes(raw)
    return cubes, embeds, diff

if __name__ == '__main__':
    nMoves = np.random.randint(1, 25)
    originalCube, originalEmbed = receiveRandomCube(nMoves)
    print(originalEmbed)

    queue = [[originalEmbed]]

    cubes, embeds, diff = getPossibleMoves(originalCube)
    print(len(embeds))
    print(len(diff[0]))

