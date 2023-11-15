import numpy as np
from numpy.linalg import norm
from typing import Callable
import csv

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

if False:
    print(f'norm: {norm(cat - dog)}')
    cosine = np.dot(cat, dog) / (norm(cat) * norm(dog))
    print(f'cosine: {cosine}')

if False:
    top = closestWordsNorm(embeds,cat)
    bot = farthestWordsNorm(embeds, cat)
    print(top)
    print(bot)

genderPairs = [('female', 'male'), ('her', 'his')]
associatedWords = []

with open('GenderPairs.csv', 'r', encoding='utf-8') as file:
    reader = csv.reader(file)

    for row in reader:
        associationWord = row[0]
        #print(associationWord)
        associatedWords.append(associationWord)

for pair in genderPairs:
    for word in associatedWords:
        g1 = pair[0]
        g2 = pair[1]

        g1Embed = embeds[g1]
        g2Embed = embeds[g2]
        if word in embeds.keys():
            wordEmbed = embeds[word]

            dist1 = embedCos(g1Embed, wordEmbed)
            dist2 = embedCos(g2Embed, wordEmbed)

            print(f'Pair: {g1}, {g2} | Dist-- {word}: {dist1}, {dist2}')
        else:
            continue