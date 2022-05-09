import numpy as np
import pandas as pd
from time import time

import sklearn
from sklearn.cluster import KMeans
import sklearn.neighbors as neighbors

import pip

def generate_triplets(method:str, sentence_embeddings:np.array, ids:np.array, n:int):
    """
    :method: the method used to calculate the triplets -> "clustering" or "brute-force"
    :sentence_embeddings: the embeddings used to calculate the distances
    :ids: museum ids in the same order as the sentence embeddings
    :n: how many triplets should be calculated

    saves: the generated triplets in a txt file
    returns: the generated triplets as a list of tuples
    """


    #sample n queries randomly
    rng = np.random.default_rng()
    query_ids = rng.integers(low=0, high=len(sentence_embeddings), size=n)
    query_ses = sentence_embeddings[query_ids]

    if method == "clustering":

        print(f"{n} triplets are generated. This may take a while ... ")
        triplets = triplets_clustering(query=query_ses, 
                                    query_ids=query_ids,
                                    data=sentence_embeddings, 
                                    ids=ids,
                                    n=n,
                                    rng=rng)
        
    elif method == "brute-force":
        print(f"{n} triplets are generated. This may take a while ... ")

        triplets = triplets_brute_force(query=query_ses,
                                        query_ids=query_ids,
                                        embeddings=sentence_embeddings,
                                        ids=ids,
                                        rng=rng)
    
    else:
        print("Please specify a valid triplet generation method!")
    
    with open("triplets.txt", "w") as txt:
        txt.write("sample_id,similar_id,dissimilar_id\n")
        for triplet in triplets:
            ori, sim, dis = triplet
            txt.write(f"{ori},{sim},{dis}\n")

    print("Done!")

    return triplets
def triplets_clustering(query, query_ids, data, ids, n, rng, nr_clusters=5, nr_farthest=3, n_random_sample=3000):
    
    ################################
    # clustering
    start_clustering = time()

    cluster_algo = KMeans(n_clusters=nr_clusters)
    cluster_algo = cluster_algo.fit(data)
    labels = cluster_algo.labels_

    df = pd.DataFrame({"index":ids, "label":labels})
    df = df.astype({"label": "str"})
    df = df.astype({"index": "str"})

    end_clustering = time()
    clustering_time = round(end_clustering - start_clustering, 2)
    ################################


    ################################
    # nearest neighbors - sklearn built-in function
    n_neighbors = 5 # rather small n_neighbors because of randomness & approximations
    nn = neighbors.NearestNeighbors(n_neighbors=n_neighbors+1, metric="cosine")
    nn.fit(data)
    nn_dists, nn_indices = nn.kneighbors(query)
    ################################


    ################################
    # farthest neighbors

    # compute distances to cluster centers
    dists_to_cluster_centers = cluster_algo.transform(query)
    # choose k clusters with maximal distance
    argsort = np.argsort(dists_to_cluster_centers, axis=1)
    k_farthest_clusters = argsort[:,-nr_farthest:]

    # select only the data that belongs to the k-farthest clusters
    string_labels = k_farthest_clusters.astype(str)
    df_clusters = [df.loc[df["label"].isin(string_label)] for string_label in string_labels]
    # extract row indices from subsample
    indices_clusters = [np.array(df.index) for df in df_clusters]
    # select a random sample form the subsample
    random_subsample = np.concatenate([[rng.permuted(inds)[:n_random_sample] for inds in indices_clusters]])
    
    # calculate distances between query and random subsampls
    results = list()
    museum_id_results = list()
    for i in range(n):

        distance_magic_clustering = calc_distances(query[i].reshape(1,-1), data[random_subsample[i]], "cosine", n_neighbors)

        max_k_ids_subsample = distance_magic_clustering[0][2].astype("int")
        max_k_dists = distance_magic_clustering[0][3]
        max_k_ids_df = random_subsample[i][max_k_ids_subsample]


        results.append((nn_indices[i][1:], nn_dists[i][1:], max_k_ids_df, max_k_dists))
        museum_id_results.append((ids[nn_indices[i][1:]], nn_dists[i][1:] ,ids[max_k_ids_df], max_k_dists))

    end = time()
    ################################


    ################################
    #create triplets
    museum_query_ids = ids[query_ids]
    museum_id_results = np.array(museum_id_results)
    sim = rng.permuted(museum_id_results[:,0], axis=1)[:,0]
    disim = museum_id_results[:,2][:,-1]
    #print(df_clusters)
    ################################

    return list(zip(museum_query_ids, sim, disim))

def triplets_brute_force(query, query_ids, embeddings, ids,  rng, k=5):
    
    neighborhoods = np.array(calc_distances(query, embeddings, "cosine", k))

    res = list()
    triplets = list()
    
    i = 0
    for min_k_ids, min_k_dists, max_k_ids, max_k_dists in neighborhoods:
        min_id = rng.integers(k)
        max_id = rng.integers(k)

        res.append((int(min_k_ids[min_id]), min_k_dists[min_id],
                    int(max_k_ids[max_id]), max_k_dists[max_id]))
        
        triplets.append([ids[query_ids[i]],ids[int(min_k_ids[min_id])], ids[int(max_k_ids[max_id])]])
        
        i += 1
        
    return triplets

def calc_distances(query, data, metric, k):
    
    if query.shape[0] == 1:
        results = list()
        # claculate distance of query to all data points in data
        dists = sklearn.metrics.pairwise_distances(query, data, metric=metric)[0]
        # zip ids to distances for sorting
        zipped_dists = np.array(list(zip(range(len(dists)),dists)))
        # sort according to distances
        sorted_dists = np.array(sorted(zipped_dists, key  = lambda x: x[1]))

        min_k_ids = sorted_dists[1:k+1, 0]
        min_k_dists = sorted_dists[1:k+1, 1]

        max_k_ids = sorted_dists[-k:, 0]
        max_k_dists = sorted_dists[-k:, 1]

        results.append((min_k_ids, min_k_dists, max_k_ids, max_k_dists))
        return results
    
    else:
        # claculate distance of query to all data points in data
        dists_matrix = sklearn.metrics.pairwise_distances(query, data, metric=metric)
        #print(dists_matrix.shape)
        
        results = list()
        for dists in dists_matrix:
            # zip ids to distances for sorting
            zipped_dists = np.array(list(zip(range(len(dists)),dists)))
            # sort according to distances
            sorted_dists = np.array(sorted(zipped_dists, key  = lambda x: x[1]))

            min_k_ids = sorted_dists[1:k+1, 0]
            min_k_dists = sorted_dists[1:k+1, 1]
            
            max_k_ids = sorted_dists[-k:, 0]
            max_k_dists = sorted_dists[-k:, 1]

            results.append((min_k_ids, min_k_dists, max_k_ids, max_k_dists))

        return results
