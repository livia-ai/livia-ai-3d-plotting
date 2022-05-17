import livia
from livia import embedding
from livia import triplet
import pandas as pd

df_data = pd.read_csv("data/wien_museum.csv")
column_names_embedding = ["title", "districts", "subjects"]
column_name_id = "id"

# compute embedding
wm_embedding = embedding.compute_embedding(df_data, column_names_embedding, column_name_id)
embedding.save_to_csv(wm_embedding, "wm_embedding_motiv")

## pca reduce
d = 3
wm_embedding_3d = embedding.pca_reduce(wm_embedding, d)
embedding.save_to_csv(wm_embedding_3d, "wm_embedding_motiv_3d")

#embedding_3d = embedding.load_csv("bel_embedding_3d.csv")


#id_column_name = "Identifier"
#title_column_name = "Title"
#color_column_name = "Collection"
#info_column_names = []
#embedding.plot_3d(embedding_3d, df_data, id_column_name, title_column_name, color_column_name, info_column_names)

