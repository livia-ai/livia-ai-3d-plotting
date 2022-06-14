
import livia.embedding as embedding
import livia.triplet as triplet
import pickle

n = 5000
embedding_loaded = embedding.load_csv("data/wm/wm_sbert_title_districts_subjects_512d.csv")

for i in range(1):
    triplets = triplet.generate_triplets(embedding_loaded, "clustering", n)
    with open(f'5000_triplets', 'wb') as fp:
        pickle.dump(triplets, fp)
    del triplets

#with open('triplets', 'rb') as fp:
#    triplets = pickle.load(fp)
