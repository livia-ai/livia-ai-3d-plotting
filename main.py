import numpy as np
import pandas as pd
import utility_functions.utility_functions as utils
from sklearn.decomposition import PCA

# import nltk
# nltk.download('stopwords')

# data loading and preprocessing
# load wien museum data 
wm_original = pd.read_csv("wien_museum.csv") # note  file location
# take only interesting columns (id, title, artistsProducers, classifications, dates, districts, subjects)
wm_filtered = wm_original[wm_original.columns[[0,3,4,5,6,7,8]]]
# merge text data of all columns into one 
# wm_filtered = wm_filtered.assign(full_text = wm_filtered[wm_filtered.columns[1:]].apply(lambda x: ' '.join(x.dropna().astype(str)),axis=1))
# apply preprocessing pipeline
# wm_preprocessed = utils.preprocessing(wm_filtered, "full_text")
print("Preprocessing: Done")

# load sentence embedding
sentence_embeddings = np.loadtxt('sentence_embeddings_wien_museum.csv', delimiter=',')
print("Loading: Done")

# create subset of sentence embeddings (62591 = all samples)
nr_samples = 10000 
sub_sample = sentence_embeddings[:nr_samples]

# perform PCA: down project to 3 dimensional vector
pca = PCA(n_components=3)
embeddings_pca = pca.fit_transform(sub_sample)

# plot results 
sent_vec_generation_method = "Sentence Transformer" # only for plot title 
dim_reduction_method = "PCA" # only for plot title
color = "classifications" # column used for coloring
utils.plot(meta_data = wm_filtered, embeddings = embeddings_pca, 
           nr_samples = nr_samples,  color = color,
           sent_vec_gen_method = sent_vec_generation_method, 
           dim_red_method = dim_reduction_method)
