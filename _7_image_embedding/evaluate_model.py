import torch
torch.cuda.empty_cache() 
from torch import nn
import torch.nn.functional as F
from torchvision import transforms

import livia.embedding as embedding
import livia.triplet as triplet
from dataset import TripletDataset, ImageDataset
from torch.utils.tensorboard import SummaryWriter

from tqdm import tqdm
import pandas as pd
import numpy as np
import pickle

#################################################################
# specify run_name
run_name = f'pretrained_triplets=59898_size=224_bs=6_margin=1_epochs=15'

#load museum data
museum_data = pd.read_csv("data/bel/belvedere.csv")
id_column = "Identifier"
title_column = "Title"
color_column = "ExpertTags"
museum_data = museum_data[[id_column, title_column, color_column]]
museum_data = museum_data.astype({id_column: "str"})
#################################################################

#################################################################
## Create dataset
## specify root directory that contains the images
#root_dir = 'data/images/bel_cropped'
#size=224
#transform = transforms.Compose([transforms.CenterCrop(size), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
#eval_dataset = ImageDataset(root_dir=root_dir, transform=transform)
#eval_set_size = len(eval_dataset)
#eval_dataloader = torch.utils.data.DataLoader(eval_dataset, batch_size=1, shuffle=False)
####################################################################



####################################################################
## evaluate model
#log_dir = "experiments/runs/"
#triplet_model = torch.load(log_dir + run_name + "/triplet_net.pt", map_location='cpu')
##triplet_model = triplet_model.to(device="cuda")

## create image embedding
## load sentece embedding that should be used for computing triplets
## only works if batch size is 1

#with torch.no_grad():
#    image_embedding_list = []
#    id_list = []
#    unique = set()
#    for img_id, img in tqdm(eval_dataloader):

#        if img_id not in unique:
#            #img = img.to(device="cuda")
#            encoded_img = triplet_model.encode(img)
#            image_embedding_list.append(encoded_img.numpy()[0])

#            unique.add(img_id)
#            id_list.append(img_id)

#image_embedding = embedding.Embedding(np.array(image_embedding_list), np.array(id_list, dtype=object))
#embedding.save_to_csv(image_embedding, f"data/bel/bel_imagemb:{run_name}")
####################################################################

image_embedding = embedding.load_csv(f"data/bel/bel_imagemb:{run_name}.csv")

##################################################################
# make 3d plots
#n = 5000
## plot sentence embedding
#embedding_loaded = embedding.load_csv("data/bel/bel_sbert_Title_Description_ExpertTags_512d.csv")
#embedding.plot_3d(embedding_loaded, museum_data, n, 
#                  id_column, title_column, color_column, [], 
#                  "Sentence Embedding")
## take only samples from df where ids are in id_list
#meta_data = museum_data.loc[museum_data[id_column].isin(image_embedding.identifier)]
## plot image embedding
#embedding.plot_3d(image_embedding, meta_data, n, 
#                id_column, title_column, color_column, [], 
#                f"Image Embedding stand: \n {run_name}", True, window_shape=(2000,750))
## plot image embedding
#embedding.plot_3d(image_embedding, meta_data, n, 
#                id_column, title_column, color_column, [], 
#                f"Image Embedding: \n {run_name}", False, window_shape=(2000,750))
###################################################################



####################################################################
# create triplets from existing image embedding and plots some
# generate triplets
n = 1000
triplets = triplet.generate_triplets(image_embedding, "clustering", n)
with open(f'data/bel/bel_image_triplets_not_frozen', 'wb') as fp:
    pickle.dump(triplets, fp)
####################################################################



##################################################################
# load already created image_triplets
root_dir = "data/images/bel_cropped"
with open(f'data/bel/bel_image_triplets_not_frozen', 'rb') as fp:
    image_triplets_loaded = pickle.load(fp)

useable_triplets = []
for ori, sim, dis in image_triplets_loaded:

    split_ori = ori.split("/")
    split_sim = sim.split("/")
    split_dis = dis.split("/")

    useable_triplets.append(("__@@__".join(split_ori), "__@@__".join(split_sim), "__@@__".join(split_dis)))

from dataset import TripletDataset
dataset = TripletDataset(triplets = useable_triplets,
                         root_dir = root_dir,)

import matplotlib.pyplot as plt
for triplet in dataset: 

    print(triplet[3])

    fig,ax = plt.subplots(1,3)
    ax[0].imshow(triplet[0])
    ax[1].imshow(triplet[1])
    ax[2].imshow(triplet[2])

    ax[0].set_title("Sample")
    ax[1].set_title("Similar")
    ax[2].set_title("Dissimilar")

    plt.show()
##################################################################