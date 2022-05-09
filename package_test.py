import livia.triplet as triplet

import numpy as np
embedding_path = 'data/se_mak_100d.csv'
# load sentence embedding 100d
sentence_embeddings = np.loadtxt(embedding_path, delimiter=',', usecols=range(1,101))
# load museum ids
ids = np.loadtxt(embedding_path, delimiter=',', usecols=0, dtype=str)

n = 100

#help(triplet.generate_triplets)

triplet.generate_triplets(method="brute-force", 
                          sentence_embeddings=sentence_embeddings, 
                          ids=ids,
                          n=n)

