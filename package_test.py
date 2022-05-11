from pyrsistent import freeze
import livia
import livia.embedding as emb
import livia.triplet as tr

from pip._internal.operations import freeze

#######################################
# triplets

import numpy as np
import livia.triplet as tr

embedding_path = 'data/se_wm_100d.csv'
# load sentence embedding 100d
sentence_embeddings = np.loadtxt(embedding_path, delimiter=',', usecols=range(1,101))
# load museum ids
ids = np.loadtxt(embedding_path, delimiter=',', usecols=0, dtype=str)

n = 10

# create instance of class Tiplet
triplet = tr.Triplet(sentence_embeddings, ids)

# generate triplets
triplet.generate_triplets(method="clustering",
                          n=n)

# compare precision: brute-force vs clustering algo
triplet.precision_comparison_histogram(n=n,
                                       n_clusters=5, 
                                       k_farthest_clusters=3, 
                                       n_random_samples=3000)

# compare performance: brute-force vs clustering algo
triplet.performance_comparison_plot(n_list=[1,10,100,300])
#######################################


#######################################
## embedding
import pandas as pd
import livia.embedding as emb

embedding = emb.Embedding()

wm_data = pd.read_csv("data/belvedere.csv")
emb_columns = ["Title", "Description"]
id_column = "Identifier"
embedding_dimensions = 100
embedding.generate_embedding(wm_data, emb_columns, id_column, embedding_dimensions)
