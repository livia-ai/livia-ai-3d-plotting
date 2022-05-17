import pandas as pd
import livia.embedding as embedding
import livia.triplet as triplet


#######################################
## embedding

bel_data = pd.read_csv("data/belvedere.csv")
#column_names_embedding = ["Title", "Description"]
#column_name_id = "Identifier"

#bel_embedding = embedding.compute_embedding(bel_data, column_names_embedding, column_name_id)
#embedding.save_to_csv(bel_embedding, "bel_embedding")

# load embedding
bel_embedding = embedding.load_csv("bel_embedding.csv")

## pca reduce
d = 3
bel_embedding_3d = embedding.pca_reduce(bel_embedding, d)
#embedding.save_to_csv(bel_3d_embedding, "bel_embedding_3d")

#######################################
## triplets

## generate triplets
#n = 10
#bel_triplets = triplet.generate_triplets(bel_embedding, method="clustering", n=n)

### compare precision: brute-force vs clustering algo
#triplet.precision_comparison_histogram(bel_embedding, 100)

## compare performance: brute-force vs clustering algo
#triplet.performance_comparison_plot(bel_embedding, n_list=[1,10,100,300])

########################################


########################################
# 3d plot
print(bel_data.columns)

id_column_name = "Identifier"
title_column_name = "Title"
color_column_name = "ClassificationName"
info_column_names = ["Creator", "ClassificationName"]
embedding.plot_3d(bel_embedding_3d, bel_data, id_column_name, title_column_name, color_column_name, info_column_names)

