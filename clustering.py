from distutils import dir_util
import numpy as np
import pandas as pd
import os
import utility_functions.utility_functions as utils
from sklearn.decomposition import PCA

from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.cluster import OPTICS
from sklearn.mixture import GaussianMixture
from sklearn.cluster import Birch


############### data loading and preprocessing ###############

# load wien museum data 
wm_original = pd.read_csv("data/wien_museum.csv") # note  file location
# take only interesting columns
wm_filtered = wm_original[wm_original.columns[[0,3,4,5,6,7,8]]]
# merge text data of all columns into one 
wm_filtered = wm_filtered.assign(full_text = wm_filtered[wm_filtered.columns[1:]].apply(lambda x: ' '.join(x.dropna().astype(str)),axis=1))

## apply preprocessing pipeline
# args = {"text_data": wm_filtered, "column_name": "full_text"}
# wm_preprocessed, time_pre = utils.time_function(utils.preprocessing, args)
#print(f"Preprocessing: {round(time_pre,4)}s")

# load sentence embedding
args = {"fname": "data/sentence_embeddings_wien_museum.csv", "delimiter" : ","}
sentence_embeddings, time_emb = utils.time_function(np.loadtxt, args)
print(f"Loading: {round(time_emb,4)}s \n\n")

# create subset of sentence embeddings (62591 = all samples)
nr_samples = 62591 
sub_sample = sentence_embeddings[:nr_samples]
# standardize data
sub_sample = (sub_sample - np.mean(sub_sample, axis=0)) / np.std(sub_sample, axis=0)

# for visualization project data to 3 dimensions
emb_3d = PCA(n_components=3).fit_transform(sub_sample)


########### Down-project to n dimensions then do clustering ##########

# specify number of clusters
nr_clusters = 11
print("Number of clusters:", nr_clusters)
# for nr_clusters in [11]:

# initialize variable path
path = ""

# if wanted specify folder to store charts and dataframe
#dir_path = f"break_clustering"
#path += dir_path + "/"
#if not os.path.isdir(dir_path): 
#     os.mkdir(dir_path)

# specify number of dimenions to use for clustering
n_components = 512
# for n_components in [50,40,30,20,10,5,3,2,1]:

# PCA
pca = PCA(n_components=n_components)
emb_clustering = pca.fit_transform(sub_sample)
print(f"Explained variance with {n_components} compononents: {np.sum(pca.explained_variance_ratio_)}")

# Clustering
args = {"X":emb_clustering}
cluster_algo = KMeans(n_clusters=nr_clusters)
cluster_algo, time_clust = utils.time_function(cluster_algo.fit, args)
print(f"Clustering: {round(time_clust,4)}s")
labels = cluster_algo.labels_

# create dataframe that contains all the necessary inforamtion for plotting
# add information
df = wm_filtered[:nr_samples][["id", "classifications", "subjects"]]
# title gets cut off after "length" characters, otherwise hoverlabel is too long
length = 75
df["title"] = wm_filtered["title"][:nr_samples].apply(lambda x: x[:length] if len(x)>length else x)
df["x"] = emb_3d[:,0]
df["y"] = emb_3d[:,1]
df["z"] = emb_3d[:,2]
df["label"] = labels
df.fillna('NaN', inplace=True)
# print(df.head())

########### stacked barplot -> distribution of classifications amoing clusters ############
# get the distribution of classifications among different clusters

counts = utils.get_counts(dataframe=df, column_to_count="classifications", nr_clusters=nr_clusters)
#print(counts.head())

# create bar chart with the ungrouped classifications
# utils.plot_counts_full(counts)

# create stacked barplot -> distribution of classifications amoing clusters
chart_path = f"bar_{nr_clusters}c_{n_components}d"
path+= chart_path
print(path)
utils.plot_counts_clusters(counts=counts, nr_clusters=nr_clusters, path=path)
# store csv of dataframe for later use
df.to_csv(path + ".csv")

# plot 3 dimenional down-projection colored by  assigned cluster 
df = df.astype({"label": "str"}) # cool color selection feature of plotly does not work with numerical labels
utils.plot_clustering(df, "Sentence Transformer", "PCA", f"K-means - {nr_clusters} clusters - {n_components}d")


########### plot without clustering ##################
## perform PCA: down project to 3 dimensional vector 
#pca = PCA(n_components=3)
#embeddings_pca = pca.fit_transform(sub_sample)
## plot
#sent_vec_generation_method = "Sentence Transformer" # only for plot title 
#dim_reduction_method = "PCA" # only for plot title
#color = "classifications" # column used for coloring
#utils.plot(meta_data = wm_filtered, embeddings = embeddings_pca, 
#           nr_samples = nr_samples,  color = color,
#           sent_vec_gen_method = sent_vec_generation_method, 
#           dim_red_method = dim_reduction_method)
