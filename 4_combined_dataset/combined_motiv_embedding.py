from livia import embedding
import pandas as pd


#############################################################
## Compute embedding

## load dataframe
#wm = pd.read_csv("data/wm/wien_museum.csv")
#bel = pd.read_csv("data/bel/belvedere.csv")
#mak_1 = pd.read_csv("data/mak/mak_1.csv", low_memory=False)
#mak_2 = pd.read_csv("data/mak/mak_2.csv", low_memory=False)
#mak_3 = pd.read_csv("data/mak/mak_3.csv", low_memory=False)
#mak = pd.concat([mak_1, mak_2, mak_3])

## bring the individual dataframes into the right format
#wm_filtered = wm[["id", "title", "subjects"]]
#wm_filtered = wm_filtered.assign(museum = "wm")
#wm_filtered = wm_filtered.rename({"id":"id", "title":"title", "subjects":"info_1"}, axis=1)

#bel_filtered = bel[["Identifier", "Title", "Description"]]
#bel_filtered = bel_filtered.assign(museum = "bel")
#bel_filtered = bel_filtered.rename({"Identifier":"id", "Title":"title", "Description":"info_1"}, axis=1)

#mak_filtered = mak[["priref", "title", "description"]]
#mak_filtered = mak_filtered.assign(museum = "mak")
#mak_filtered = mak_filtered.rename({"priref":"id", "title":"title", "description":"info_1"}, axis=1)

## combine dataframes
#df_combined = pd.concat([wm_filtered, bel_filtered, mak_filtered]).reset_index(drop=True)
#df_combined.to_csv("df_combined.csv")

## compute embeddings
#embedding_column_names = ["title", "info_1"]
#id_column_name = "id"
#columns_str="_".join(embedding_column_names)

#museum_embedding = embedding.compute_embedding(df_combined, embedding_column_names, id_column_name)
#embedding.save_to_csv(museum_embedding, f"combined_sbert_{columns_str}_{museum_embedding.shape[1]}d")
#print(museum_embedding.shape)

#museum_embedding_dp = embedding.pca_reduce(museum_embedding, 256)
#embedding.save_to_csv(museum_embedding_dp, f"combined_sbert_{columns_str}_{museum_embedding_dp.shape[1]}d")
#print(museum_embedding_dp.shape)
###########################################################


###########################################################
# Load and Plot
df_combined = pd.read_csv("df_combined.csv", low_memory=False)
combined_emb = embedding.load_csv("combined_sbert_title_info_1_256d.csv")

n = 5000
embedding.plot_3d(combined_emb, df_combined, n,
            "id", "title", "museum", [], 
            f"Combined before - Motiv Embeddings - {n}rs", standardize=False)


############################################################