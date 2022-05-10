import livia.triplet as tr

import numpy as np
embedding_path = 'data/se_wm_100d.csv'
# load sentence embedding 100d
sentence_embeddings = np.loadtxt(embedding_path, delimiter=',', usecols=range(1,101))
# load museum ids
ids = np.loadtxt(embedding_path, delimiter=',', usecols=0, dtype=str)

n = 10

# create instance of class Tiplet
triplet = tr.Triplet(sentence_embeddings, ids)

## generate triplets
#triplet.generate_triplets(method="clustering",
#                          n=n)

## compare precision: brute-force vs clustering algo
#triplet.precision_comparison_histogram(n=n,
#                                       n_clusters=5, 
#                                       k_farthest_clusters=3, 
#                                       n_random_samples=3000)

## compare performance: brute-force vs clustering algo
#triplet.performance_comparison_plot(n_list=[1,10,100,300])