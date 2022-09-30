
from livia import embedding 
from livia import triplet 
import pickle
from dataset import TripletDataset
import tqdm
import os

fnk = 500
nnk = 25
folder = f'data/wm_triplets_nnk={nnk}_fnk={fnk}'

###########################
###generate triplets from sentence embeddings

## generate triplets
#n = 10000
#embedding_loaded = embedding.load_csv("data/wm/wm_sbert_title_districts_subjects_512d.csv")

#os.mkdir(folder)
#for i in tqdm.tqdm(range(25)):
#    triplets = triplet.generate_triplets(embedding_loaded, "clustering", n, nn_k=nnk, fn_k=fnk)#, nr_clusters=8, nr_random_sample=10000)
#    with open(f'{folder}/{i}', 'wb') as fp:
#        pickle.dump(triplets, fp)
#    del triplets
##########################

##########################
# load triplets, combine into long list and create image_paths
root_dir = "data/images/wm_cropped"

# load triplets
triplets = []
for i in range(25):
    file_name = f"/{i}"
    with open(folder + file_name, 'rb') as fp:
        triplets_loaded = pickle.load(fp)
        triplets.extend(triplets_loaded)

print(len(triplets))
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
file_name = "/image_paths"
with open(folder + file_name, 'wb') as fp:
    pickle.dump(unique_image_path_triplets, fp)