import livia.triplet as triplet

import numpy as np
embedding_path = 'data/se_wm_100d.csv'
# load sentence embedding 100d
sentence_embeddings = np.loadtxt(embedding_path, delimiter=',', usecols=range(1,101))
# load museum ids
ids = np.loadtxt(embedding_path, delimiter=',', usecols=0, dtype=str)

n = 100

## generate triplets
#triplet.generate_triplets(method="brute-force", 
#                          sentence_embeddings=sentence_embeddings, 
#                          ids=ids,
#                          n=n)

## compare precision: brute-force vs clustering algo
#triplet.precision_comparison_histogram(sentence_embeddings=sentence_embeddings, 
#                                        ids=ids,
#                                        n=n,
#                                        n_clusters=5, 
#                                        k_farthest_clusters=3, 
#                                        n_random_samples=3000)

## compare performance: brute-force vs clustering algo
#triplet.performance_comparison_plot(sentence_embeddings=sentence_embeddings, 
#                                    ids=ids,
#                                    n_list=[1,10,100,300])