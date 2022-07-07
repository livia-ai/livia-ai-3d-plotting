
from livia import embedding 
from livia import triplet 
import pickle
from dataset import TripletDataset


############################
###generate triplets from sentence embeddings

## generate triplets
#n = 10000
#embedding_loaded = embedding.load_csv("data/wm/wm_sbert_title_districts_subjects_512d.csv")

#for i in range(15):
#    print(i)
#    triplets = triplet.generate_triplets(embedding_loaded, "clustering", n, n_neighbors=200)
#    with open(f'data/wm_triplets_{i}', 'wb') as fp:
#        pickle.dump(triplets, fp)
#    del triplets
###########################

##########################
# load triplets, combine into long list and create image_paths
root_dir = "data/images/wm_cropped"

# load triplets
triplets = []
path = "data/"
for i in range(15):
    file_name = f"wm_triplets_{i}"
    with open(path + file_name, 'rb') as fp:
        triplets_loaded = pickle.load(fp)
        triplets.extend(triplets_loaded)

useable_triplets = []
for ori, sim, dis in triplets:

    split_ori = ori.split("/")
    split_sim = sim.split("/")
    split_dis = dis.split("/")

    useable_triplets.append(("__@@__".join(split_ori), "__@@__".join(split_sim), "__@@__".join(split_dis)))

unique_triplets, removed = triplet.uniqueness_triplets(useable_triplets)

## create dataset
unique_dataset = TripletDataset(triplets = unique_triplets,
                                root_dir = root_dir)

unique_image_path_triplets = unique_dataset.samples
file_name = "wm_image_paths"
with open(file_name, 'wb') as fp:
    pickle.dump(unique_image_path_triplets, fp)