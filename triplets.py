# imports
import numpy as np
import pandas as pd
from utility_functions import utility_functions as utils
import sklearn
from sklearn.decomposition import PCA
import sklearn.neighbors as neighbors

# load wien museum data 
wm_original = pd.read_csv("data/wien_museum.csv") # note  file location
# take only interesting columns
wm_filtered = wm_original[wm_original.columns[[0,3,4,5,6,7,8]]]
# merge text data of all columns into one 
wm_filtered = wm_filtered.assign(full_text = wm_filtered[wm_filtered.columns[1:]].apply(lambda x: ' '.join(x.dropna().astype(str)),axis=1))
# load sentence embedding
sentence_embeddings = np.loadtxt('data/sentence_embeddings_wien_museum.csv', delimiter=',')

# standardize data
stand_sentence_embeddings = (sentence_embeddings - np.mean(sentence_embeddings, axis=0)) / np.std(sentence_embeddings, axis=0)
stand_pca = PCA(n_components=3)
coordinates_3d = stand_pca.fit_transform(stand_sentence_embeddings)

# create a dataframe that contains the needed meta data (mostly for plotting)
triplet_dataframe = wm_filtered[["id","classifications", "subjects", "full_text"]]
triplet_dataframe  = triplet_dataframe.assign(title = wm_filtered["title"][:].apply(lambda x: x[:75] if len(x)>75 else x))
triplet_dataframe  = triplet_dataframe.assign(x = coordinates_3d[:,0])
triplet_dataframe  = triplet_dataframe.assign(y = coordinates_3d[:,1])
triplet_dataframe  = triplet_dataframe.assign(z = coordinates_3d[:,2])

# sample n query samples randomly 
n = 30
rng = np.random.default_rng()
query_ids = rng.integers(low=0, high=len(stand_sentence_embeddings), size=n)
queries = stand_sentence_embeddings[query_ids]

full_information, triplets = utils.create_triplets(queries, query_ids, stand_sentence_embeddings, "cosine", 15)

# displays one randomly chosen triplet
#utils.display_one_triplet(triplets, triplet_dataframe)

# displays all triplets
# utils.display_all_triplets(triplets, triplet_dataframe)

# displays first sample with all k simialar and dissimilar samples
utils.display_all_dis_similar(query_ids, full_information, triplet_dataframe)
