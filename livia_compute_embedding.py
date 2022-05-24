import livia
from livia import embedding
from livia import triplet
import pandas as pd
from time import time
#### Compute Embedding ####
## wm
#museum = "wm"
#df_data = pd.read_csv("data/wm/wien_museum.csv")
#embedding_column_names = ["title", "subjects"]
#id_column_name = "id"
#columns_str="_".join(embedding_column_names)

## bel
#museum = "bel"
#df_data = pd.read_csv("data/bel/belvedere.csv")
#embedding_column_names = ["Title", "Description"]
#id_column_name = "Identifier"
#columns_str="_".join(embedding_column_names)

# mak
museum = "mak"
mak_1 = pd.read_csv("data/mak/mak_1.csv", low_memory=False)
mak_2 = pd.read_csv("data/mak/mak_2.csv", low_memory=False)
mak_3 = pd.read_csv("data/mak/mak_3.csv", low_memory=False)
df_data = pd.concat([mak_1, mak_2, mak_3]).reset_index(drop=True)
embedding_column_names = ["title", "description"]
id_column_name = "priref"
columns_str="_".join(embedding_column_names)

## compute embedding
#museum_embedding = embedding.compute_embedding(df_data, embedding_column_names, id_column_name)
#embedding.save_to_csv(museum_embedding, f"{museum}_sbert_{columns_str}_{museum_embedding.shape[1]}d")
#print(museum_embedding.shape)


embedding_to_plot = embedding.load_csv("mak_sbert_title_description_512d.csv")
print(embedding_to_plot.shape)
# pca reduce
d = 256
museum_embedding_dp = embedding.pca_reduce(embedding_to_plot, d)
print("pca done")
embedding.save_to_csv(museum_embedding_dp, f"{museum}_sbert_{columns_str}_{museum_embedding_dp.shape[1]}d")
print(museum_embedding_dp.shape)










#from livia.embedding import compute_embedding, pca_reduce

#embedding = compute_embedding(df_data, ['title', 'districts', 'subjects'], 'id')
#start = time()
#embedding_3d = pca_reduce(embedding, 3)
#end = time()
#print(f"PCA:{round(end - start, 2)}s")
#print(embedding_3d.shape)