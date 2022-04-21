#imports
import numpy as np
import pandas as pd
import utility_functions.utility_functions as utils
from sklearn.decomposition import PCA

# data loading and preprocessing
dataset_name = "BEL"

if dataset_name == "WM":
    # load and prepare wien museum
    wm_original = pd.read_csv("data/wien_museum.csv") 
    wm_filtered = wm_original[wm_original.columns[[0,3,4,5,6,7,8]]]
    #wm_filtered = wm_filtered.assign(full_text = wm_filtered[wm_filtered.columns[1:]].apply(lambda x: ' '.join(x.dropna().astype(str)),axis=1))
    #wm_preprocessed = utils.preprocessing(wm_filtered, "full_text")
    sentence_embeddings = np.loadtxt('data/se_wm_100d.csv', delimiter=',')[:, 1:]
    meta_data = wm_filtered
    # dataset specific column names used for coloring and hover labels
    color = "classifications" # column used for coloring
    hover_name = "title"
    identifier = "id"

if dataset_name == "MAK":
    # load and prepare mak
    mak_1 = pd.read_csv("data/mak_1.csv")
    mak_2 = pd.read_csv("data/mak_2.csv")
    mak_3 = pd.read_csv("data/mak_3.csv")
    mak = pd.concat([mak_1, mak_2, mak_3])
    mak_filtered = mak[mak.columns[[0,2,4,5,15,16,17,21,28,29,31,34,36,38]]]
    mak_filtered.reset_index(drop=True, inplace=True)
    #mak_filtered = mak_filtered.assign(full_text = mak_filtered[mak_filtered.columns[1:]].apply(lambda x: ' '.join(x.dropna().astype(str)),axis=1))
    #mak_preprocessed = utils.preprocessing(mak_filtered, "full_text")
    sentence_embeddings = np.loadtxt('data/se_mak_100d.csv', delimiter=',')[:, 1:]
    meta_data = mak_filtered
    # dataset specific column names used for coloring and hover labels
    color = "object_name" # "collection", "object_name"
    identifier = "priref"
    hover_name = "title"

if dataset_name == "BEL":
    # load and prepare belvedere
    bel = pd.read_csv("data/belvedere.csv").reset_index()
    bel_filtered = bel[bel.columns[[0,10,1,2,3,4,11,12,14,16,21,22]]]
    #bel_filtered = bel_filtered.assign(full_text = bel_filtered[bel_filtered.columns[1:]].apply(lambda x: ' '.join(x.dropna().astype(str)),axis=1))
    #bel_preprocessed = utils.preprocessing(bel_filtered, "full_text")
    sentence_embeddings = np.loadtxt('data/se_bel_100d.csv', delimiter=',', dtype= object)[:, 1:]
    meta_data = bel_filtered
    # dataset specific column names used for coloring and hover labels
    color = "Collection" # ObjectClass, Collection
    identifier = "Identifier"
    hover_name = "Title"

print(sentence_embeddings.shape)

# randomly sample a subset of the sentence embeddings (62591 = all samples) 
n = 5000
rng = np.random.default_rng()
sample_ids = rng.integers(low=0, high=len(sentence_embeddings), size=n)
samples = sentence_embeddings[sample_ids]

# perform PCA: down project to 3 dimensional vector
pca_3d = PCA(n_components=3)
embeddings_3d = pca_3d.fit_transform(samples)

# plot results 
title = f"{n} Random Samples from {dataset_name} Data"
utils.plot(meta_data = meta_data, 
           sample_ids = sample_ids,
           embeddings = embeddings_3d,  
           color = color,
           identifier = identifier,
           hover_name = hover_name, 
           title = title)
