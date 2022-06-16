import livia
from livia import embedding
from livia import triplet
import pandas as pd
from time import time

#### Compute Embedding ####
## wm
#museum = "wm"
#data_folder = f"data/{museum}/"
#df_data = pd.read_csv(data_folder + "wien_museum.csv")
#embedding_column_names = ["title","districts", "subjects"]
#id_column_name = "id"
#columns_str="_".join(embedding_column_names)

## bel
#museum = "bel"
#data_folder = f"data/{museum}/"
#df_data = pd.read_csv(data_folder +  "belvedere.csv")
#embedding_column_names = ["Title", "Description"]
#id_column_name = "Identifier"
#columns_str="_".join(embedding_column_names)

## mak
#museum = "mak"
#data_folder = f"data/{museum}/"
#mak_1 = pd.read_csv(data_folder + "mak_1.csv", low_memory=False)
#mak_2 = pd.read_csv(data_folder + "mak_2.csv", low_memory=False)
#mak_3 = pd.read_csv(data_folder + "mak_3.csv", low_memory=False)

#df_data = pd.concat([mak_1, mak_2, mak_3]).reset_index(drop=True)
#embedding_column_names = ["title", "description"]
#id_column_name = "priref"
#columns_str="_".join(embedding_column_names)

## compute embedding
#museum_embedding = embedding.compute_embedding(df_data, embedding_column_names, id_column_name)
#embedding.save_to_csv(museum_embedding, f"{museum}_sbert_{columns_str}_{museum_embedding.shape[1]}d")
#print(museum_embedding.shape)

## pca reduce
#d = 256
#museum_embedding_dp = embedding.pca_reduce(museum_embedding, d)
#embedding.save_to_csv(museum_embedding_dp, f"{museum}_sbert_{columns_str}_{museum_embedding.shape[1]}d")
#print(museum_embedding_dp.shape)



##########################################

## 3d plot
#embedding_to_plot = embedding.load_csv(data_folder + f"{museum}_sbert_{columns_str}_512d.csv")

df_data = pd.read_csv("data/combined/df_combined.csv", low_memory=False)
embedding_to_plot = embedding.load_csv("data/combined/combined_sbert_title_info_1_256d.csv")


title_column_name = "title"
color_column_name = "museum"
info_column_names = []

n = 10000
embedding.plot_3d(embedding_to_plot, df_data, n, "id", title_column_name, color_column_name, info_column_names, title_plot=f"3D Plot of Combined Dataset", window_shape=(1000,1000))
