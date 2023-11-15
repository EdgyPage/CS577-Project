import numpy as np
from numpy.linalg import norm
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

def ex():
    pass

cat = embeds['cat']
dog = embeds['dog']

print(f'norm: {norm(cat - dog)}')
cosine = np.dot(cat, dog) / (norm(cat) * norm(dog))
print(f'cosine: {cosine}')
