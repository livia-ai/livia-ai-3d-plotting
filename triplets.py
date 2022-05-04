# imports
from re import S
import string
import time
import numpy as np
import pandas as pd
from utility_functions import utility_functions as utils
import sklearn
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import sklearn.neighbors as neighbors

#############################################
#### prepare data ####
start_prep = time.time()

# load wien museum data 
wm_original = pd.read_csv("data/wien_museum.csv") # note  file location
# take only interesting columns
wm_filtered = wm_original[wm_original.columns[[0,3,4,5,6,7,8]]]
# merge text data of all columns into one 
# wm_filtered = wm_filtered.assign(full_text = wm_filtered[wm_filtered.columns[1:]].apply(lambda x: ' '.join(x.dropna().astype(str)),axis=1))
# load sentence embedding 100d
#sentence_embeddings = np.loadtxt('data/se_wm_100d.csv', delimiter=',')[:, 1:]
# load sentence embedding 512d
sentence_embeddings = np.loadtxt('data/sentence_embeddings_wm.csv', delimiter=',')

# standardize data
stand_sentence_embeddings = (sentence_embeddings - np.mean(sentence_embeddings, axis=0)) / np.std(sentence_embeddings, axis=0)
# calculate 3d projection for visualization
stand_pca = PCA(n_components=3)
coordinates_3d = stand_pca.fit_transform(stand_sentence_embeddings)

#####
#TODO: function that does this for every dataset
# create a dataframe that contains the needed meta data (mostly for plotting)
triplet_dataframe = wm_filtered[["id","classifications", "subjects"]]#, "full_text"]]
triplet_dataframe  = triplet_dataframe.assign(title = wm_filtered["title"][:].apply(lambda x: x[:75] if len(x)>75 else x))
triplet_dataframe  = triplet_dataframe.assign(x = coordinates_3d[:,0])
triplet_dataframe  = triplet_dataframe.assign(y = coordinates_3d[:,1])
triplet_dataframe  = triplet_dataframe.assign(z = coordinates_3d[:,2])
#####

#sample n queries randomly
n = 1000
rng = np.random.default_rng()
query_ids = rng.integers(low=0, high=len(stand_sentence_embeddings), size=n)
queries = stand_sentence_embeddings[query_ids]

end_prep = time.time()
print(f"Time needed for data preparation: {round(end_prep - start_prep, 2)}s")
#############################################


############################################
#### Brute-Force FN Algo for Evaluation ####
k_bf = 1000 #this has to be quite large to get accurate positions in evaluation
start_bf = time.time()
true_results, triplets = utils.triplets_brute_force(queries, query_ids, stand_sentence_embeddings, "cosine", k_bf)
end_bf = time.time()
time_bf = round(end_bf - start_bf, 2)
print(f"Time needed for brute force nn&fn: {time_bf}s")
print()
############################################

for index in range():
    ############################################
    #### Clustering FN Algo ####
    n_clusters = 5
    k_farthest = 3
    n_random_samples = 3000
    results_clustering_algo , time_clustering = utils.triplets_clustering(query=queries, data=stand_sentence_embeddings, df=triplet_dataframe,
                                                                        nr_clusters=n_clusters, nr_farthest=k_farthest, n_random_sample=n_random_samples)
    print(f"Time needed for clustering nn&fn: {time_clustering}s")
    ###########################################


    ###########################################
    # Evaluate Results
    mean_dists = list()
    mean_positions = list()
    for i in range(n):
        max_ids_cl = results_clustering_algo[i][2]
        max_dists_cl = results_clustering_algo[i][3]

        max_ids_bf = true_results[i][2]
        max_dists_bf = true_results[i][3]

        #print(max_dists_cl)
        #print(max_dists_bf)

        performance_cluster_algo = [np.argwhere(max_ids_bf == idx) for idx in max_ids_cl]
        #print(performance_cluster_algo)
        sum_pos = 0
        for i in performance_cluster_algo:
            if len(i) == 0:
                sum_pos -= 10
            else:
                sum_pos += i[0][0]

        mean_positions.append(abs(sum_pos/len(performance_cluster_algo) - k_bf))
        mean_dists.append((np.mean(max_dists_cl),np.mean(max_dists_bf[-6:]) ))

    #print(mean_positions)
    #print(mean_dists) 

    import matplotlib.pyplot as plt
    plt.hist(mean_positions)
    plt.title(f"512d_{index}\n time needed cluster={time_clustering}s, , time needed bf={time_bf}s")
    plt.savefig(f'performance_hist_512d_{index}.png')

    plt.clf()
    print()



#print("query:", query_ids)
#print("sklearn:\n", indices[0][1:])
#print("my brute force:\n", full_information[0][0])

# create triplets with clustering and gt tree
# displays one randomly chosen triplet
#utils.display_one_triplet(triplets, triplet_dataframe)

# displays all triplets
# utils.display_all_triplets(triplets, triplet_dataframe)

# displays first sample with all k simialar and dissimilar samples
#utils.display_all_dis_similar(query_ids, full_information, triplet_dataframe)









#nr_clusters = 5
#print(f"################ {nr_clusters} #############")
#start_sk = time.time()

#cluster_algo = KMeans(n_clusters=nr_clusters)
#cluster_algo = cluster_algo.fit(stand_sentence_embeddings)
#labels = cluster_algo.labels_

#triplet_dataframe["label"] = labels
#triplet_dataframe = triplet_dataframe.astype({"label": "str"})

#end_clustering = time.time()

#print("Min number of samples in any cluster:", min(triplet_dataframe["label"].value_counts()))
#print(f"Time needed for clustering: {round(end_clustering - start_sk, 2)}s")

##utils.plot_clustering(triplet_dataframe,"Sentence_Transformer", "PCA", f"KMeans - {nr_clusters} Clusters")

##### sklearn nn ####
#n_neighbors = 5

#nn = neighbors.NearestNeighbors(n_neighbors=n_neighbors+1, metric="cosine")
#nn.fit(stand_sentence_embeddings)

#nn_dists, nn_indices = nn.kneighbors(queries)


###### compute cluster centers with maxium distances ####
#dists_to_cluster_centers = cluster_algo.transform(queries)

#argsort = np.argsort(dists_to_cluster_centers, axis=1)

##shuffled = rng.permuted(argsort[:,-3:], axis=1)
##random_clusters = shuffled[:,0]
#nr_farthest = 3
#farthest_clusters = argsort[:,-nr_farthest:]

#farthest_clustering = list()

#for i in range(n):
#    string_labels = [str(cl) for cl in farthest_clusters[i]]
#    all_rows_of_cluster = triplet_dataframe.loc[triplet_dataframe["label"].isin(string_labels)]

#    # indices_farthest_cluster = [np.array(triplet_dataframe.loc[triplet_dataframe["label"] == str(cluster)].index) for cluster in farthest_clusters]
#    random_samples = 3000
#    indices_farthest_cluster = np.array(all_rows_of_cluster.index)
#    random_sampled = rng.permuted(indices_farthest_cluster)[:random_samples]

#    clustering_info = utils.calc_distances(queries[i].reshape(1,-1), stand_sentence_embeddings[random_sampled], "cosine", n_neighbors)

#    max_k_ids = clustering_info[0][2].astype("int")
#    max_k_dists = clustering_info[0][3]

#    max_k_ids_df = random_sampled[max_k_ids]

#    farthest_clustering.append((max_k_ids_df, max_k_dists, max_k_ids))

#end_sk = time.time()
#print(f"Time needed for sklearn nn&fn + clustering {round(end_sk - start_sk, 2)}s")
#print("results cluster center fn")
#for i in farthest_clustering:
#    print(i)
#print()
#print(triplet_dataframe.loc[max_k_ids_df])

#print("results brute force fn")
#for i in true_full_information:
#    print(i[2])
#    print(i[3])