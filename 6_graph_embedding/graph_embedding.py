import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
from node2vec import Node2Vec
from sklearn.decomposition import PCA
import numpy as np
import plotly.express as px
import gc

### Wien Museum####
museum = "wm"
df_data = pd.read_csv("data/wm/wien_museum.csv")
embedding_column_names = ["id", "title", "subjects"]

df_data = df_data[embedding_column_names]

# filter all samples with no subjects
df_data = df_data[~df_data["subjects"].isnull()].reset_index(drop=True)

# take subsample 
data_length = len(df_data)
n_total = 30000
rng = np.random.default_rng()
id_list = list(range(data_length))
sample_ids = rng.choice(id_list, size=n_total, replace=False)

df_data = df_data.copy().loc[sample_ids]

# create graph 
G = nx.Graph()

for i in range(len(df_data)):

    id,title,subjects = df_data.iloc[i]

    # add sample node
    G.add_node(id)

    # add subject node
    subjects = subjects.split("|")
    G.add_nodes_from(subjects)

    # generate edges as tuples
    edges = [(subject, id) for subject in subjects]
    G.add_edges_from(edges)

model = Node2Vec(G, workers=6)
node2vec = model.fit(window=10, min_count=1 ,workers=6)
del model
del G

keyed_vectors = node2vec.wv
del node2vec
gc.collect()

identifier = list(df_data["id"].astype(str))
graph_embedding = keyed_vectors[identifier]
format = ['%s']
format += ['%.18e']*(graph_embedding.shape[1])
np.savetxt(f"wm_graph_embedding_{n_total}rs.csv",np.concatenate([np.array(identifier).reshape(-1,1), graph_embedding], axis=1, dtype=object), delimiter=',', fmt=format)



# pca to 3d
# standardize data
graph_embedding = (graph_embedding - np.mean(graph_embedding, axis=0)) / np.std(graph_embedding, axis=0)
# project to d dimensions
dimensions = 3
pca = PCA(n_components=dimensions)
embedding_matrix_3d = pca.fit_transform(graph_embedding)

# take random subsample of dataset
n=5000
rng = np.random.default_rng()
id_list = list(range(n_total))
sampled_id_list = rng.choice(id_list, size=n, replace=False)
sample_ids = np.array(df_data.index.values)[sampled_id_list]
embedding_matrix_3d = embedding_matrix_3d[sampled_id_list]
df = df_data.copy().loc[sample_ids]
del df_data

# Create 3d plot of random subsample
id_column = "id"
color_column = "subjects"
title_column = "title"
df = df.astype({id_column: "str"})

# for better visualization crop title
length = 75
df[title_column] = df[title_column].apply(lambda x: str(x)[:length] if len(str(x))>length else str(x))
df[color_column] = df[color_column].apply(lambda x: str(x)[:100] if len(str(x))>100 else str(x))
df["x"] = embedding_matrix_3d[:,0]
df["y"] = embedding_matrix_3d[:,1]
df["z"] = embedding_matrix_3d[:,2]
df.fillna('NaN', inplace=True)

fig = px.scatter_3d(df, 
                x='x', y='y', z='z', 
                color=color_column, 
                hover_name=title_column, # what to show when hovered over
                width=2000, height=850, # adjust height and width
                title=f"Node2Vec - 3D Plot {n}rs")

# make set size for legend and hover label
fig.update_layout(showlegend=True,
                    legend = dict(
                        font = dict(size = 10)
                        ), 
                hoverlabel=dict(
                        font_size=10,
                        ),
                title_x=0.5
                )

# set marker size
fig.update_traces(marker_size = 3)
fig.show()


