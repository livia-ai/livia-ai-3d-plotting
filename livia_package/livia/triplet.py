from time import time
import warnings
import numpy as np
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
from sklearn.cluster import KMeans
import sklearn.neighbors as neighbors

from livia.embedding import Embedding


def sample_n_queries(embedding:np.array, n:int, rng):
    query_ids = rng.integers(low=0, high=len(embedding), size=n)
    query_ses = embedding[query_ids]
    return query_ids, query_ses

def generate_triplets(embedding:Embedding, method:str, n:int, seed:int=None):
    """
    :method: the method used to calculate the triplets -> "clustering" or "brute-force"
    :sentence_embeddings: the embeddings used to calculate the distances
    :ids: museum ids in the same order as the sentence embeddings
    :n: how many triplets should be calculated

    saves: the generated triplets in a txt file
    returns: the generated triplets as a list of tuples
    """

    # create random number generator
    rng = np.random.default_rng(seed=seed)

    #sample n queries randomly
    query_ids, query_ses = sample_n_queries(embedding.embedding, n, rng)

    if method == "clustering":

        print(f"{n} triplets are generated. This may take a while ... ")
        triplets, _, _ = triplets_clustering(embedding=embedding,
                                             query=query_ses, 
                                             query_ids=query_ids,
                                             n=n,
                                             rng=rng)
                    
    elif method == "brute-force":

        print(f"{n} triplets are generated. This may take a while ... ")
        triplets, _, _ = triplets_brute_force(embedding=embedding,
                                              query=query_ses,
                                              query_ids=query_ids,
                                              rng=rng)
                
    else:
        print("Please specify a valid triplet generation method!")

    #with open("triplets.txt", "w") as txt:
    #    txt.write("sample_id,similar_id,dissimilar_id\n")
    #    for triplet in triplets:
    #        ori, sim, dis = triplet
    #        txt.write(f"{ori},{sim},{dis}\n")
    print("Done!")

    return triplets

def triplets_clustering(embedding, query, query_ids, n, rng, nr_clusters=5, nr_farthest=3, n_random_sample=3000):
    
    ################################
    # clustering
    start = time()

    cluster_algo = KMeans(n_clusters=nr_clusters)
    cluster_algo = cluster_algo.fit(embedding.embedding)
    labels = cluster_algo.labels_

    df = pd.DataFrame({"index":embedding.identifier, "label":labels})
    df = df.astype({"label": "str"})
    df = df.astype({"index": "str"})
    ################################


    ################################
    # nearest neighbors - sklearn built-in function
    n_neighbors = 5 # rather small n_neighbors because of randomness & approximations
    nn = neighbors.NearestNeighbors(n_neighbors=n_neighbors+1, metric="cosine")
    nn.fit(embedding.embedding)
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

        distance_magic_clustering = calc_distances(query[i].reshape(1,-1), embedding.embedding[random_subsample[i]], "cosine", n_neighbors)

        max_k_ids_subsample = distance_magic_clustering[0][2].astype("int")
        max_k_dists = distance_magic_clustering[0][3]
        max_k_ids_df = random_subsample[i][max_k_ids_subsample]


        results.append((nn_indices[i][1:], nn_dists[i][1:], max_k_ids_df, max_k_dists))
        museum_id_results.append((embedding.identifier[nn_indices[i][1:]], nn_dists[i][1:] ,embedding.identifier[max_k_ids_df], max_k_dists))

    ################################

    ################################
    #create triplets
    museum_query_ids = embedding.identifier[query_ids]
    museum_id_results = np.array(museum_id_results)
    sim = rng.permuted(museum_id_results[:,0], axis=1)[:,0]
    disim = museum_id_results[:,2][:,-1]
    #print(df_clusters)
    ################################

    end = time()
    time_needed = round(end - start, 2)

    return list(zip(museum_query_ids, sim, disim)), results, time_needed

def triplets_brute_force(embedding, query, query_ids, rng, k=5):
    
    start = time()

    neighborhoods = np.array(calc_distances(query, embedding.embedding, "cosine", k))

    res = list()
    triplets = list()
    
    i = 0
    for min_k_ids, min_k_dists, max_k_ids, max_k_dists in neighborhoods:
        min_id = rng.integers(k)
        max_id = rng.integers(k)

        res.append((int(min_k_ids[min_id]), min_k_dists[min_id],
                    int(max_k_ids[max_id]), max_k_dists[max_id]))

        triplets.append([embedding.identifier[query_ids[i]], embedding.identifier[int(min_k_ids[min_id])], embedding.identifier[int(max_k_ids[max_id])]])
        
        i += 1

    end = time()
    time_needed = round(end - start, 2)

    return triplets, neighborhoods, time_needed

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

def precision_comparison_histogram(embedding, n:int, n_clusters:int = 5, k_farthest_clusters:int = 3, n_random_samples:int = 3000, seed:int=None):
    
    # create random number generator
    rng = np.random.default_rng(seed=seed)

    #sample n queries randomly
    query_ids, query_ses = sample_n_queries(embedding.embedding, n, rng)

    ###########################################
    # perform nn/fn clustering algorithm
    print(f"{n} triplets are generated using the clustering algorithm")
    triplets_cl, res_cl, time_cl = triplets_clustering(embedding=embedding,
                                                        query=query_ses, 
                                                        query_ids=query_ids,
                                                        n=n,
                                                        rng=rng,
                                                        nr_clusters=n_clusters,
                                                        nr_farthest=k_farthest_clusters,
                                                        n_random_sample=n_random_samples)
    ###########################################


    ###########################################
    # perform brute-force algorithm
    k_bf = 1000
    print(f"{n} triplets are generated using the brute-force algorithm")
    triplets_bf, res_bf, time_bf = triplets_brute_force(embedding=embedding,
                                                        query=query_ses,
                                                        query_ids=query_ids,
                                                        rng=rng,
                                                        k=k_bf)
    ###########################################


    ###########################################
    # Evaluate Results Distance Calculations
    print("Compare Results...")
    mean_dists = list()
    mean_positions = list()
    for i in range(n):
        max_ids_cl = res_cl[i][2]
        max_dists_cl = res_cl[i][3]

        max_ids_bf = res_bf[i][2]
        max_dists_bf = res_bf[i][3]

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
    ######################################

    # plot results
    fig, ax = plt.subplots(1,1, figsize=(20,10))

    plt.hist(mean_positions)
    plt.suptitle(f"Precision Comparison Histogram - Farthest Neighbor Problem", fontsize=16)
    plt.title(f"Clustering Algorithm Parameters: n_clusters={n_clusters},  k_farthest={k_farthest_clusters}, n_random_samples={n_random_samples} \n Performance: Clustering={time_cl}s, Brute-Force={time_bf}s")
    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
    ax.set_xlabel("Clustering Method - Average Deviation from Optimal Farthest Neighbor")
    ax.set_ylabel("Frequency")
    #plt.savefig(f'precision_comparison_histogram.png')
    plt.show()

    return plt

def performance_comparison_plot(embedding, n_list:list=[1,10,100,500], n_clusters:int = 5, k_farthest_clusters:int = 3, n_random_samples:int=3000, seed:int=None):
    
    #  create random number generator
    rng = np.random.default_rng(seed=seed)

    performance_bf = list()
    performance_cl = list()
    for n in n_list:
        print(f"Calculating n={n}")
        #sample n queries randomly
        query_ids, query_ses = sample_n_queries(embedding.embedding, n, rng)

        ###########################################
        # perform nn/fn clustering algorithm
        #print(f"{n} triplets are generated using the clustering algorithm")
        triplets_cl, res_cl, time_cl = triplets_clustering(embedding=embedding,
                                                            query=query_ses, 
                                                            query_ids=query_ids,
                                                            n=n,
                                                            rng=rng,
                                                            nr_clusters=n_clusters,
                                                            nr_farthest=k_farthest_clusters,
                                                            n_random_sample=n_random_samples)
        ###########################################


        ###########################################
        k_bf = 5
        #print(f"{n} triplets are generated using the brute-force algorithm")
        triplets_bf, res_bf, time_bf = triplets_brute_force(embedding=embedding,
                                                            query=query_ses,
                                                            query_ids=query_ids,
                                                            rng=rng,
                                                            k=k_bf)
        ###########################################
        
        performance_bf.append(time_bf)
        performance_cl.append(time_cl)


    ######################################
    # plot results
    barwidth = 0.25
    x_axis = np.arange(len(n_list))
    cl_bar_pos = [x + barwidth/2 for x in x_axis]
    bf_bar_pos = [x - barwidth/2 for x in x_axis]

    fig, ax = plt.subplots(1,1, figsize=(10,10))
    ax.bar(cl_bar_pos, performance_cl, color ='C0', width = barwidth,
            edgecolor ='grey', label ='clustering')
    ax.bar(bf_bar_pos, performance_bf, color ='C1', width = barwidth,
            edgecolor ='grey', label ='brute force')

    ax.set_xlabel('Number of samples [n]')
    ax.set_ylabel('Time [s]')
    ax.set_xticks(x_axis,n_list)

    for bars in ax.containers:
        ax.bar_label(bars)

    ax.legend()
    ax.set_title('Performance Comparison: Brute-Force vs Clustering', fontsize=16)

    #plt.savefig(f'performance_comparison_plot.png')
    plt.show()

    return plt
    ######################################
