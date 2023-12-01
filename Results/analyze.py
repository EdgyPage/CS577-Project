import numpy as np
from numpy.linalg import norm
from typing import Callable
import csv
import matplotlib.pyplot as plt


def embeds(path: str):
    embeds = dict()
    with open(path) as file:
        for line in file:
            tokens = line.rstrip().split()
            #print(tokens)
            embeds[tokens[0]] = np.array([float(tok) for tok in tokens[1:]])
            #print(embeds[tokens[0]])
    return embeds

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

if False:
    print(f'norm: {norm(cat - dog)}')
    cosine = np.dot(cat, dog) / (norm(cat) * norm(dog))
    print(f'cosine: {cosine}')

if False:
    top = closestWordsNorm(embeds,cat)
    bot = farthestWordsNorm(embeds, cat)
    print(top)
    print(bot)

#genderPairs = [('female', 'male'), ('her', 'his')]

def spotlightWords(spotlightPath: str):
    spotlightWords = []
    with open(spotlightPath, 'r', encoding='utf-8-sig') as file:
        reader = csv.reader(file)

        for row in reader:
            associationWord = row[0]
            #print(associationWord)
            spotlightWords.append(associationWord)
    return spotlightWords

def genderPairs(pairsPath: str):
    tuplePairs = []
    with open(pairsPath, 'r', encoding='utf-8-sig') as file:
        reader = csv.reader(file)
        for row in reader:
            p1 = row[0]
            p2 = row[1]
            tuplePairs.append((p1, p2))
    return tuplePairs

def distanceVector(spotlightWords: [str], pairs : [(str, str)], embeds: dict):
    for pair in pairs:
        g1Dists = []
        g2Dists = []
        g1 = pair[0]
        g2 = pair[1]
        for word in spotlightWords:
            g1Embed = embeds[g1]
            g2Embed = embeds[g2]
            if word in embeds.keys():
                wordEmbed = embeds[word]

                dist1 = round(embedCos(g1Embed, wordEmbed), 4)
                dist2 = round(embedCos(g2Embed, wordEmbed), 4)

                g1Dists.append(dist1)
                g2Dists.append(dist2)

            else:
                continue
        g1Dists = np.array(g1Dists)
        g2Dists = np.array(g2Dists)
        embeds['DistanceVector'] = {}
        embeds['DistanceVector'][g1] = g1Dists
        embeds['DistanceVector'][g2] = g2Dists

def plotter(pairs:[(str, str)], spotlightWords: [str], neutralEmbeds: dict, genderEmbeds: dict, emphasize:[str] = []):
    
    for g1, g2 in pairs:
        plt.scatter(genderEmbeds[g1], neutralEmbeds[g1], label = 'Male', color = 'blue')
        plt.scatter(genderEmbeds[g2], neutralEmbeds[g2], label = 'Female', color = 'red')
        plt.xlabel('Gendered Cosine Distance')
        plt.ylabel('Neutral Cosine Distance')
        plt.title(f'{g1}/{g2} Distances From Word List')
        indices = []
        for empWord in emphasize:
            indices.append(spotlightWords.index(empWord))
        
        for index in indices:
            plt.scatter(genderEmbeds[g1][index], neutralEmbeds[g1][index], marker = '^', label = f'{spotlightWords[index]}', color = 'blue')
            plt.scatter(genderEmbeds[g2][index], neutralEmbeds[g2][index], marker = '^', label = f'{spotlightWords[index]}', color = 'red')
    plt.show()