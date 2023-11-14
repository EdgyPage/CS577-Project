import numpy as np

embeds = dict()
with open('vectors.txt') as file:
    for line in file:
        tokens = line.rstrip().split('\t')
        print(tokens)
        embeds[tokens[0]] = np.array([float(tok) for tok in tokens[1:]])
        print(embeds[tokens[0]])
