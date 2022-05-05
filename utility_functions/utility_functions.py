from random import random
import string
from time import time
import pandas as pd
import numpy as np
import nltk

#nltk.download('stopwords')
from nltk.corpus import stopwords

import plotly.express as px
import altair as alt
import altair_saver as asav

import sklearn
from sklearn.cluster import KMeans
import sklearn.neighbors as neighbors
import scipy.spatial.distance as dist


def preprocessing(text_data: pd.DataFrame, column_name: str) -> pd.DataFrame:

    helper = text_data.copy(deep = True)
    helper = helper.assign(pre_text = helper[column_name]) 
    tokenizer = nltk.RegexpTokenizer(r"\w+")
    cachedStopWords = stopwords.words('german')
    
    # noch kein Stemmer -> könnt ma auch probieren hab online ein bisschen geschaut und die
    # Qualität von deutschen Stemmern ist relativ bescheiden
    
    # iterate over all documents and tokenize each text
    i = 0
    for text in helper[column_name]:
        # text = text.lower()
        # remove special characters and do tokenization
        text = np.array(tokenizer.tokenize(text))
        #remove stopwords
        text = [word for word in text if not word in cachedStopWords]
        
        helper.at[i,"pre_text"] = text
        i += 1
        
    return helper

def plot(meta_data, sample_ids, embeddings, color, identifier, hover_name, title):

    #print(meta_data.columns)
    #print(color, hover_name)
    df = meta_data.loc[sample_ids][[color, identifier]]
    #print(df)
    # title gets cut off after "length" characters, otherwise hoverlabel is too long
    length = 75
    df["title"] = meta_data[hover_name].loc[sample_ids].apply(lambda x: str(x)[:length] if len(str(x))>length else str(x))

    df["x"] = embeddings[:,0]
    df["y"] = embeddings[:,1]
    df["z"] = embeddings[:,2]
    df.fillna('NaN', inplace=True)

    fig = px.scatter_3d(df, 
                        x='x', y='y', z='z', 
                        color=color, 
                        hover_name="title", # what to show when hovered over
                        hover_data=[identifier],
                        width=2500, height=1250, # adjust height and width
                        title=title)

    # make set size for legend and hover label
    fig.update_layout(showlegend=True,
                     legend = dict(
                            font = dict(size = 10)
                            ), 
                    hoverlabel=dict(
                            font_size=10,
                            )
                    )

    # set marker size
    fig.update_traces(marker_size = 3)
    fig.show()

def plot_clustering(df, sent_vec_gen_method, dim_red_method, cluster_algo):

    title = f"Visualization of: {sent_vec_gen_method} + {dim_red_method} + {cluster_algo}"

    fig = px.scatter_3d(df, 
                        x="x", y="y", z="z", 
                        color="label", 
                        hover_name="label",
                        hover_data=["id", "title", "classifications"], # what to show when hovered over
                        width=2500, height=1250, # adjust height and width
                        title=title)

    # make set size for legend and hover label
    fig.update_layout(showlegend=True,
                     legend = dict(
                            font = dict(size = 10)
                            ), 
                    hoverlabel=dict(
                            font_size=9,
                            )
                    )
                    
    # set marker size
    fig.update_traces(marker_size = 3)
    fig.show()

def get_counts(dataframe, column_to_count, nr_clusters):

    counts = dataframe[column_to_count].value_counts().to_frame(name = "counts_full")
    counts.index.name = column_to_count
    counts.reset_index(inplace=True)

    for i in range(nr_clusters):
        only_one_cluster = dataframe.loc[dataframe['label'] == i]
        counts_cluster = only_one_cluster[column_to_count].value_counts().to_frame(name = f"counts_{i}")
        counts_cluster.index.name = column_to_count
        counts_cluster.reset_index(inplace=True)
        counts = pd.merge(counts, counts_cluster, on=column_to_count, how='left')
    counts = counts.fillna(0)

    return counts

def plot_counts_full(counts, path, column_to_count):
    alt.data_transformers.disable_max_rows()

    bars = alt.Chart(counts).mark_bar().encode(
        x=alt.X(column_to_count, sort='-y'),
        y='counts:Q',
    )


    text = bars.mark_text(
        #align='left',
        #baseline='middle',
        dy=-3,  # Nudges text up 
        size = 8
    ).encode(
        text='counts:Q'
    )

    plot = (bars + text).configure_axis(labelLimit=180, labelOffset=30)

    asav.save(plot, f'{path}.pdf')

def plot_counts_clusters(counts, nr_clusters, path):
    alt.data_transformers.disable_max_rows()
    alt.renderers.enable('altair_viewer')
    columns = [f'counts_{x}' for x in range(nr_clusters)]
    bars = alt.Chart(counts).transform_fold(
        columns
    ).mark_bar().encode(
        alt.X('classifications', sort='-y', axis=alt.Axis(labelAngle=-90)),
        y='value:Q',
        color=alt.Color('key:N', legend=alt.Legend(
            orient='none',
            legendX=130, legendY=-40,
            direction='horizontal',
            titleAnchor='middle'))
    ).properties(
        height=500
    )

    bars.configure_axis(labelLimit=400)
    
    asav.save(bars, f'{path}.pdf')

def time_function(function, x):
    start = time()
    output = function(**x)
    end = time()
    elapsed_time = end - start

    return output, elapsed_time

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

def triplets_brute_force(query, query_ids, embeddings, metric, k):
    
    neighborhoods = calc_distances(query, embeddings, metric, k)

    rng = np.random.default_rng()
    res = list()
    triplets = list()
    
    i = 0
    for min_k_ids, min_k_dists, max_k_ids, max_k_dists in neighborhoods:
        min_id = rng.integers(k)
        max_id = rng.integers(k)

        res.append((int(min_k_ids[min_id]), min_k_dists[min_id],
                    int(max_k_ids[max_id]), max_k_dists[max_id]))
        
        triplets.append([query_ids[i], int(min_k_ids[min_id]), int(max_k_ids[max_id])])
        
        i += 1
        
    return neighborhoods, triplets

def nn_fn_clustering(query, data, df, nr_clusters=5, nr_farthest=3, n_random_sample=3000):

    n = len(query)

    rng = np.random.default_rng()

    start = time()

    # clustering
    cluster_algo = KMeans(n_clusters=nr_clusters)
    cluster_algo = cluster_algo.fit(data)
    labels = cluster_algo.labels_

    df["label"] = labels
    df = df.astype({"label": "str"})

    end_clustering = time()
    clustering_time = round(end_clustering - start, 2)

    # nearest_neighbors - sklearn
    n_neighbors = 5 # rather small n_neighbors because of randomness & approximations
    nn = neighbors.NearestNeighbors(n_neighbors=n_neighbors+1, metric="cosine")
    nn.fit(data)
    nn_dists, nn_indices = nn.kneighbors(query)
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
    for i in range(n):

        distance_magic_clustering = calc_distances(query[i].reshape(1,-1), data[random_subsample[i]], "cosine", n_neighbors)

        max_k_ids_subsample = distance_magic_clustering[0][2].astype("int")
        max_k_dists = distance_magic_clustering[0][3]
        max_k_ids_df = random_subsample[i][max_k_ids_subsample]

        results.append((nn_indices[i][1:], nn_dists[i][1:], max_k_ids_df, max_k_dists))

    end = time()

    return results, round(end - start, 2)

def triplets_clustering(results, query_ids):
    results = np.array(results)
    rng = np.random.default_rng()
    sim = rng.permuted(results[:,0], axis=1)[:,0].astype(int)
    disim = results[:,2][:,-1].astype(int)
    return list(zip(query_ids, sim, disim))

def display_one_triplet(triplets, dataframe):
    
    length = len(triplets)
    
    rng = np.random.default_rng()
    idx = rng.integers(length)
    sample = triplets[idx]
    sample_id, similar_id, dissimilar_id = sample
    print("Triplet IDs:", sample)
    
    #sample_data = dataframe.loc[sample_id]
    #similar_data = dataframe.loc[similar_id]
    #dissimilar_data = dataframe.loc[dissimilar_id]
        
    #sample_text = sample_data["full_text"]
    #similar_text = similar_data["full_text"]
    #disimilar_text = dissimilar_data["full_text"]
    
    triplet_df = dataframe.loc[list(sample)]
    triplet_df = triplet_df.assign(label = ["sample", "similar", "disimilar"])
    
    #print("Sample:\n", sample_text)
    #print()
    #print("Similar:\n", similar_text)
    #print()
    #print("Dissimilar:\n", disimilar_text)
    #print()
    
    fig = px.scatter_3d(triplet_df, x='x', y='y', z='z', 
                    color='label', hover_name="title", hover_data=["id", "classifications", "url"],
                    width=1100, height=800,# adjust height and width
                    title="One triplet")

    fig.update_layout(legend = dict(
                            font = dict(size = 10)
                            ), 

                      hoverlabel=dict(
                            font_size=9,
                            )
                    )

    fig.update_traces(marker_size = 3)

    fig.show()

def meta_data_triplets(triplets, dataframe):
    for sample in triplets[:]:
        sample_id, similar_id, dissimilar_id = sample
        print("Triplet IDs:", sample)
        
        sample_data = dataframe.loc[sample_id]
        similar_data = dataframe.loc[similar_id]
        dissimilar_data = dataframe.loc[dissimilar_id]

        sample_title = sample_data["title"]
        similar_title  = similar_data["title"]
        disimilar_title  = dissimilar_data["title"]

        sample_url = sample_data["url"]
        similar_url = similar_data["url"]
        disimilar_url  = dissimilar_data["url"]
        
        #triplet_df = dataframe.loc[list(sample)]
        #triplet_df = triplet_df.assign(label = ["sample", "similar", "disimilar"])
        
        print(f"Sample: {sample_title} \n {sample_url}")
        print(f"Similar: {similar_title} \n {similar_url}")
        print(f"Dissimilar: {disimilar_title} \n {disimilar_url}\n")

        with open("example_triplets.txt", "a") as file:
            file.write(f"Triplet IDs:{sample}\n" +
                       f"Sample: {sample_title}\n{sample_url}\n" +
                       f"Similar: {similar_title}\n{similar_url}\n" +
                       f"Dissimilar: {disimilar_title}\n{disimilar_url}\n\n")

def display_all_triplets(triplets, dataframe):
    
    triplets_df = pd.DataFrame()
    i = 0
    for triplet in triplets:
        
        triplet_df = dataframe.loc[triplet]
        triplet_df = triplet_df.assign(label = ["sample", "similar", "disimilar"])
        triplet_df = triplet_df.assign(sample_id = str(i))
        
        triplets_df = pd.concat([triplets_df, triplet_df])
        
        i += 1
            
    triplets_df.reset_index(drop=True, inplace=True)
    
    
    fig = px.scatter_3d(triplets_df, x='x', y='y', z='z', 
                    color='sample_id', hover_name="title", hover_data=["id", "classifications", "label"],
                    width=1100, height=800,# adjust height and width
                    title="All triplets")

    fig.update_layout(legend = dict(
                            font = dict(size = 10)
                            ), 

                      hoverlabel=dict(
                            font_size=9,
                            )
                    )

    fig.update_traces(marker_size = 3)

    fig.show()

def display_all_dis_similar(query_ids, neighborhoods, dataframe):
    

    idx = query_ids[0]
    neighborhood = neighborhoods[0]
    
    sim_ids = neighborhood[0]
    dissim_ids = neighborhood[2]
    
    sample_df = dataframe.loc[idx].to_frame().transpose()
    sample_df = sample_df.assign(label = ["sample"])
    
    sim_df = dataframe.loc[sim_ids]
    sim_df = sim_df.assign(label = ["similar"]*len(sim_ids))
    
    dissim_df = dataframe.loc[dissim_ids]
    dissim_df = dissim_df.assign(label = ["disimilar"]*len(dissim_ids))
    
    df = pd.concat([sample_df, sim_df, dissim_df])
    df.reset_index(drop=True, inplace=True)
    print(df)
    
    fig = px.scatter_3d(df, x='x', y='y', z='z', 
                    color='label', hover_name="title", hover_data=["id", "classifications"],
                    width=1100, height=800,# adjust height and width
                    title="All similar and dissimlar samples for one sample")

    fig.update_layout(legend = dict(
                            font = dict(size = 10)
                            ), 

                      hoverlabel=dict(
                            font_size=9,
                            )
                    )

    fig.update_traces(marker_size = 3)

    fig.show()













