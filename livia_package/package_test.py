import pandas as pd
import livia.embedding as embedding
import livia.triplet as triplet


#######################################
# embedding
#mak_1 = pd.read_csv("data/mak/mak_1.csv", low_memory=False)
#mak_2 = pd.read_csv("data/mak/mak_2.csv", low_memory=False)
#mak_3 = pd.read_csv("data/mak/mak_3.csv", low_memory=False)
#mak = pd.concat([mak_1, mak_2, mak_3])
#print(len(mak))

#column_names_embedding = ["title", "object_name", "description"]
#column_name_id = "priref"

#mak_embedding = embedding.compute_embedding(mak, column_names_embedding, column_name_id)
#embedding.save_to_csv(mak_embedding, "mak_embedding_motiv")

## pca reduce
#d = 3
#mak_embedding_3d = embedding.pca_reduce(mak_embedding, d)
#embedding.save_to_csv(mak_embedding_3d, "mak_embedding_motiv_3d")

## pca reduce
#d = 100
#mak_embedding_100d = embedding.pca_reduce(mak_embedding, d)
#embedding.save_to_csv(mak_embedding_100d, "mak_embedding_motiv_100d")

########################################
## triplets

## generate triplets
#n = 10
#bel_triplets = triplet.generate_triplets(bel_embedding, method="clustering", n=n)

### compare precision: brute-force vs clustering algo
#triplet.precision_comparison_histogram(bel_embedding, 100)

## compare performance: brute-force vs clustering algo
#triplet.performance_comparison_plot(bel_embedding, n_list=[1,10,100,300])

########################################


##########################################
## 3d plot
embedding_to_plot = embedding.load_csv("data/wm/wm_embedding_motiv.csv")
# load and prepare wien museum
wm_original = pd.read_csv("data/wm/wien_museum.csv") 

id_column_name = "id"
title_column_name = "title"
color_column_name = "subjects"
info_column_names = ["subjects"]
n = 5000

embedding.plot_3d(embedding_to_plot, wm_original, n, id_column_name, title_column_name, color_column_name, info_column_names)
