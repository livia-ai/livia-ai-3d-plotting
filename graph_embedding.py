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
n = 10000
rng = np.random.default_rng()
id_list = list(range(data_length))
sample_ids = rng.choice(id_list, size=n, replace=False)

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

model = Node2Vec(G, num_walks=32, workers=4)
node2vec = model.fit(window=10, min_count=1 ,workers=4)
sample_ids = list(df_data["id"].astype(str))
del model
del G

keyed_vectors = node2vec.wv

del node2vec
gc.collect()

print(keyed_vectors)
graph_embedding = keyed_vectors[sample_ids]
print(graph_embedding)

# pca to 3d
# standardize data
graph_embedding = (graph_embedding - np.mean(graph_embedding, axis=0)) / np.std(graph_embedding, axis=0)
# project to d dimensions
dimensions = 3
pca = PCA(n_components=dimensions)
embedding_matrix_3d = pca.fit_transform(graph_embedding)

# Create 3d plot of random subsample
id_column = "id"
color_column = "subjects"
title_column = "title"

df_data = df_data.astype({id_column: "str"})
#df = df.copy()
# take subsample 

# for better visualization crop title
length = 75
df_data[title_column] = df_data[title_column].apply(lambda x: str(x)[:length] if len(str(x))>length else str(x))
df_data[color_column] = df_data[color_column].apply(lambda x: str(x)[:100] if len(str(x))>100 else str(x))
df_data["x"] = embedding_matrix_3d[:,0]
df_data["y"] = embedding_matrix_3d[:,1]
df_data["z"] = embedding_matrix_3d[:,2]
df_data.fillna('NaN', inplace=True)

fig = px.scatter_3d(df_data, 
                x='x', y='y', z='z', 
                color=color_column, 
                hover_name=title_column, # what to show when hovered over
                width=2000, height=850, # adjust height and width
                title="Node2Vec - 3D Plot")

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


