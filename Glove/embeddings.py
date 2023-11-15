import numpy as np
from numpy.linalg import norm
from typing import Callable

embeds = dict()

with open('vectors.txt') as file:
    for line in file:
        tokens = line.rstrip().split()
        #print(tokens)
        embeds[tokens[0]] = np.array([float(tok) for tok in tokens[1:]])
        #print(embeds[tokens[0]])

#print(embeds.keys())
#print(embeds['cat'])
#print(embeds['dog'])

def embedNorm(arr1 : np.ndarray, arr2 : np.ndarray):
    return norm(arr1 - arr2)

def embedCos(arr1 : np.ndarray, arr2 : np.ndarray):
    return np.dot(arr1, arr2) / (norm(arr1) * norm(arr2))

def closestWordsNorm(embedDict: dict , arr: np.ndarray, n: int = 5):
    topWords = [('NA', 10000) for l in range(n+1)]
    for word, array in embedDict.items():
        dist = embedNorm(arr, array)
        if dist < topWords[-1][1]:
            newLow = (word, dist)
            topWords[-1] = newLow
            topWords = sorted(topWords, key= lambda x: x[1])
    return topWords[1:]

def farthestWordsNorm(embedDict: dict , arr: np.ndarray, n: int = 5):
    bottomWords = [('NA', 0) for l in range(n)]
    for word, array in embedDict.items():
        dist = embedNorm(arr, array)
        if dist > bottomWords[0][1]:
            newHigh = (word, dist)
            bottomWords[0] = newHigh
            bottomWords = sorted(bottomWords, key= lambda x: x[1])
    return bottomWords

cat = embeds['cat']
dog = embeds['dog']

print(f'norm: {norm(cat - dog)}')
cosine = np.dot(cat, dog) / (norm(cat) * norm(dog))
print(f'cosine: {cosine}')

top = closestWordsNorm(embeds,cat)
bot = farthestWordsNorm(embeds, cat)
print(top)
print(bot)