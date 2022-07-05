import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

# load full sentence embedding
dataset_name = "bel" # wm, mak , bel
path = f"data/sentence_embeddings_{dataset_name}.csv"
sentence_embeddings = np.loadtxt(path, delimiter=",")
#print("loaded")

# standardize data
stand_sentence_embeddings = (sentence_embeddings - np.mean(sentence_embeddings, axis=0)) / np.std(sentence_embeddings, axis=0)

# project to d dimensions
dimensions = 100
pca = PCA(n_components=dimensions)
embeddings_dp = pca.fit_transform(stand_sentence_embeddings)

# include museum id in sentence embedding array

if dataset_name == "wm":
    wm_data = pd.read_csv("data/wien_museum.csv")
    identifier = wm_data["id"].to_numpy().reshape((-1,1))
 
if dataset_name == "mak":
    mak_1 = pd.read_csv("data/mak_1.csv")
    mak_2 = pd.read_csv("data/mak_2.csv")
    mak_3 = pd.read_csv("data/mak_3.csv")
    mak = pd.concat([mak_1, mak_2, mak_3])
    identifier = mak["priref"].to_numpy().reshape((-1,1))

if dataset_name == "bel":
    bel = pd.read_csv("data/belvedere.csv").reset_index()
    identifier = bel["Identifier"].to_numpy().reshape((-1,1))

# concatenate the index column in the first column of the array
embeddings_dp = np.concatenate([identifier, embeddings_dp], axis=1)

# small format hack otherwise it does not work for all 3 datasets
format = ['%s']
format += ['%.18e']*dimensions
# save down projected sentence embedding
np.savetxt(f'data/se_{dataset_name}_{dimensions}d.csv', embeddings_dp, delimiter=',', fmt=format)
