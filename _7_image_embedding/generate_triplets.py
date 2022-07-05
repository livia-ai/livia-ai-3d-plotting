
from livia import embedding 
from livia import triplet 
import pickle
from dataset import TripletDataset


###########################
##generate triplets from sentence embeddings

## generate triplets
#n = 15000
#embedding_loaded = embedding.load_csv("data/bel/bel_sbert_Title_Description_ExpertTags_512d.csv")

#for i in range(4):
#    print(i)
#    triplets = triplet.generate_triplets(embedding_loaded,"clustering", n)
#    with open(f'data/bel_triplets_{i}', 'wb') as fp:
#        pickle.dump(triplets, fp)
#    del triplets
##########################

#########################
# load triplets, combine into long list and create image_paths
root_dir = "data/images/bel_cropped"

# load triplets
triplets = []
path = "data/triplets/bel/"
for i in range(4):
    file_name = f"bel_triplets_{i}"
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

print(len(unique_triplets), removed)

# create dataset
unique_dataset = TripletDataset(triplets = unique_triplets,
                         root_dir = root_dir)

unique_image_path_triplets = unique_dataset.samples
file_name = "bel_image_paths"
with open(file_name, 'wb') as fp:
    pickle.dump(unique_image_path_triplets, fp)