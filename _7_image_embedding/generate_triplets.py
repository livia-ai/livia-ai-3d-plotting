
from livia import embedding 
from livia import triplet 
import pickle

# generate triplets
n = 15000
embedding_loaded = embedding.load_csv("data/bel/bel_sbert_Title_Description_ExpertTags_512d.csv")

for i in range(2,5):
    print(i)
    triplets = triplet.generate_triplets(embedding_loaded,"clustering", n)
    with open(f'data/bel_triplets_{i}', 'wb') as fp:
        pickle.dump(triplets, fp)
    del triplets
    print()



## load triplets, combine into long list and create image_paths
#root_dir = "data/test_images/wm_cropped_train"

## load triplets
#triplets = []
#path = "data/triplets/15000_nneighbors=15/"
#for i in range(6):
#    file_name = f"more_div_15000_triplets_{i}"
#    with open(path + file_name, 'rb') as fp:
#        triplets_loaded = pickle.load(fp)
#        triplets.extend(triplets_loaded)

## create dataset
#from dataset import TripletDataset
#dataset = TripletDataset(triplets = triplets,
#                         root_dir = root_dir)

#image_path_triplets = dataset.samples
#print(len(image_path_triplets))
#file_name = "image_paths_for_dataset_nneighbors=15"
#with open(file_name, 'wb') as fp:
#    pickle.dump(image_path_triplets, fp)
