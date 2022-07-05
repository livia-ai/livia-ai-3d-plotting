# imports
import sys
sys.path.append('.')
import time
import numpy as np
import pandas as pd
from utility_functions import utility_functions as utils
from sklearn.decomposition import PCA

#############################################
#### prepare data ####
start_prep = time.time()

# load wien museum data 
wm_original = pd.read_csv("data/wien_museum.csv") # note  file location
# take only interesting columns
wm_filtered = wm_original[wm_original.columns[[0,1,3,4,5,6,7,8]]]
# merge text data of all columns into one 
#wm_filtered = wm_filtered.assign(full_text = wm_filtered[wm_filtered.columns[2:]].apply(lambda x: ' '.join(x.dropna().astype(str)),axis=1))
# load sentence embedding 100d
sentence_embeddings = np.loadtxt('data/sentence_embeddings_wm.csv', delimiter=',')[:, 1:]
# load sentence embedding 512d
#sentence_embeddings = np.loadtxt('data/sentence_embeddings_wm.csv', delimiter=',')

# standardize data
stand_sentence_embeddings = (sentence_embeddings - np.mean(sentence_embeddings, axis=0)) / np.std(sentence_embeddings, axis=0)
# calculate 3d projection for visualization
stand_pca = PCA(n_components=3)
coordinates_3d = stand_pca.fit_transform(stand_sentence_embeddings)

#####
#TODO: function that does this for every dataset
# create a dataframe that contains the needed meta data (mostly for plotting)
triplet_dataframe = wm_filtered[["id","classifications", "subjects", "url"]]
triplet_dataframe  = triplet_dataframe.assign(title = wm_filtered["title"][:].apply(lambda x: x[:75] if len(x)>75 else x))
triplet_dataframe  = triplet_dataframe.assign(x = coordinates_3d[:,0])
triplet_dataframe  = triplet_dataframe.assign(y = coordinates_3d[:,1])
triplet_dataframe  = triplet_dataframe.assign(z = coordinates_3d[:,2])
#####

n = 1000 
#sample n queries randomly
rng = np.random.default_rng()
query_ids = rng.integers(low=0, high=len(stand_sentence_embeddings), size=n)
queries = stand_sentence_embeddings[query_ids]

end_prep = time.time()
print(f"Time needed for data preparation: {round(end_prep - start_prep, 2)}s")
#############################################


############################################
# Performance comparison
#n_list = [1, 10, 100]
#utils.performance_comparison_triplets(n_list, stand_sentence_embeddings, triplet_dataframe)
############################################


############################################
##### Brute-Force FN Algo for Evaluation ####
#k_bf = 1000 #this has to be quite large to get accurate positions in evaluation
#start_bf = time.time()
#true_results, triplets = utils.triplets_brute_force(queries, query_ids, stand_sentence_embeddings, "cosine", k_bf)
#end_bf = time.time()
#time_bf = round(end_bf - start_bf, 2)
############################################


############################################
#### Clustering NN FN Algo ####
n_clusters = 5
k_farthest = 3
n_random_samples = 3000
results_clustering_algo , time_clustering = utils.nn_fn_clustering(query=queries, data=stand_sentence_embeddings, df=triplet_dataframe,
                                                                    nr_clusters=n_clusters, nr_farthest=k_farthest, n_random_sample=n_random_samples)

#tr_start = time.time()
triplets = utils.triplets_clustering(results_clustering_algo, query_ids)
#tr_end = time.time()
#triplets_time = (tr_end - tr_start)
#time_cl = round(time_clustering +  triplets_time, 2)

###########################################
# Evaluate Triplets
#utils.display_one_triplet(triplets, triplet_dataframe)
utils.meta_data_triplets(triplets, triplet_dataframe)
###########################################


###########################################
## Evaluate Results Distance Calculations
#mean_dists = list()
#mean_positions = list()
#for i in range(n):
#    max_ids_cl = results_clustering_algo[i][2]
#    max_dists_cl = results_clustering_algo[i][3]

#    max_ids_bf = true_results[i][2]
#    max_dists_bf = true_results[i][3]

#    performance_cluster_algo = [np.argwhere(max_ids_bf == idx) for idx in max_ids_cl]
#    #print(performance_cluster_algo)
#    sum_pos = 0
#    for i in performance_cluster_algo:
#        if len(i) == 0:
#            sum_pos -= 10
#        else:
#            sum_pos += i[0][0]

#    mean_positions.append(abs(sum_pos/len(performance_cluster_algo) - k_bf))
#    mean_dists.append((np.mean(max_dists_cl),np.mean(max_dists_bf[-6:]) ))


#import matplotlib.pyplot as plt
#plt.hist(mean_positions)
#plt.title(f"Performance: n_clusters={n_clusters}, k_farthest={k_farthest}, n_random_samples={n_random_samples} \n time needed cluster={time_clustering}s, , time needed bf={time_bf}s")
#plt.savefig(f'visualizations/performance_hist_{n_clusters}ncl_{k_farthest}kf_{n_random_samples}nrs.png')

#plt.clf()
######################################