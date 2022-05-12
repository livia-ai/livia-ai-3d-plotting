import livia
import livia.embedding as embedding
import livia.triplet as triplet


#######################################
## embedding
#import pandas as pd

#bel_data = pd.read_csv("data/belvedere.csv")
#column_names_embedding = ["Title", "Description"]
#column_name_id = "Identifier"

#bel_embedding = embedding.compute_embedding(bel_data, column_names_embedding, column_name_id)
#embedding.save_to_csv(bel_embedding, "bel_embedding")

#d = 3
#bel_3d_embedding = embedding.pca_reduce(bel_embedding, d)
#embedding.save_to_csv(bel_3d_embedding, "bel_embedding_3d")

# load embedding
#bel_embedding = embedding.load_csv("bel_embedding.csv")


#######################################
## triplets

n = 10


## generate triplets
#bel_triplets = triplet.generate_triplets(bel_embedding, method="brute-force", n=n)

## compare precision: brute-force vs clustering algo
#triplet.precision_comparison_histogram(bel_embedding, 100)


# compare performance: brute-force vs clustering algo
triplet.performance_comparison_plot(bel_embedding, n_list=[1,10,100,300])

########################################
