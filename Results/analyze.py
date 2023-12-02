import numpy as np
from numpy.linalg import norm
from typing import Callable
import csv
import matplotlib.pyplot as plt


def embeds(path: str):
    embeds = dict()
    with open(path, 'r', encoding= 'utf-8') as file:
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

def closestWordsDistances(spotlightWords: [str], embedGDict: dict , embedNDict: dict , n: int = 5):
    
    distancesG = []
    distancesN = []
    for word in spotlightWords:
        if word in embedGDict.keys() and word in embedNDict.keys():
            topGWords = closestWordsNorm(embedGDict, embedGDict[word], n)
            topNWords = closestWordsNorm(embedNDict, embedNDict[word], n)
            for i in range(n):
                if topGWords[i][0] in embedGDict.keys() and topGWords[i][0] in embedNDict.keys():
                    distancesG.append(embedCos(embedGDict[topGWords[i][0]], embedGDict[word]))
                    #print(distancesG.append(embedCos(embedGDict[topGWords[i][0]], embedGDict[word])))
                    distancesN.append(embedCos(embedNDict[topGWords[i][0]], embedNDict[word]))
    if sum(distancesG) == 0:
        percentDiff = 0
    else:
        percentDiff = round(sum(distancesN) /sum(distancesG), 3)

    return percentDiff


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
        if g1 in embeds.keys() and g2 in embeds.keys():
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

def plotter(pairs:[(str, str)], spotlightWords: [str], neutralEmbeds: dict, genderEmbeds: dict, title: str, emphasize:[str] = []):
    
    gkeys = genderEmbeds.keys()
    nkeys = neutralEmbeds.keys()

    for g1, g2 in pairs:
        if g1 in gkeys and g1 in nkeys and g2 in gkeys and g2 in nkeys:
            plt.scatter(genderEmbeds[g1], neutralEmbeds[g1], s= .5, label = 'Male', color = 'blue')
            plt.scatter(genderEmbeds[g2], neutralEmbeds[g2], s= .5, label = 'Female', color = 'red')
            plt.xlabel('Gendered Cosine Distance')
            plt.ylabel('Neutral Cosine Distance')
            plt.title(f'{g1}-{g2} Distances From Word List in {title}')
            plt.legend()
            """
            indices = []
            for empWord in emphasize:
            indices.append(spotlightWords.index(empWord))
            """
        


            x_values = np.linspace(min(min(genderEmbeds[g1]), min(neutralEmbeds[g1]), min(genderEmbeds[g2]), min(neutralEmbeds[g2])),\
                                max(max(genderEmbeds[g1]), max(neutralEmbeds[g1]), max(genderEmbeds[g2]), max(neutralEmbeds[g2])), 1000)
            plt.plot(x_values, x_values, color='black', alpha = .2, linestyle='-')

            """
            for i, index in enumerate(indices):
                plt.scatter(genderEmbeds[spotlightWords[index]][index], neutralEmbeds[spotlightWords[index]][index], marker = '^', label = f'{spotlightWords[index]}', color = 'blue')
                plt.scatter(genderEmbeds[g2][index], neutralEmbeds[g2][index], marker = '^', label = f'{spotlightWords[index]}', color = 'red')
            """
            plt.savefig(f'{g1}-{g2} Distances From Word List in {title}.png')
            plt.show()