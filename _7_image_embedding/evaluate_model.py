import torch
torch.cuda.empty_cache() 
from torch import nn
import torch.nn.functional as F
from torchvision import transforms

import livia.embedding as embedding
import livia.triplet as triplet
from dataset import TripletDataset, ImageDataset, MixedImageDataset
from torch.utils.tensorboard import SummaryWriter

import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
import numpy as np
import pickle
import os

#################################################################
# specify run_name + log_dir + eval_dir + root directory
root_dir = 'data/images/bel_cropped_test'
log_dir = "experiments/bel/"
run_name = f'pretrained_triplets=26676_bs=6_margin=1_lr=0.0001_wd=0'
evaluation_dir = f"evaluation_results/{run_name}"

# create evaluation dir
if not os.path.isdir("evaluation_results"):
    os.mkdir("evaluation_results")
if not os.path.isdir(evaluation_dir):
    os.mkdir(evaluation_dir)
#################################################################


#################################################################
# load museums data for plotting
# bel df
museum_data = pd.read_csv("data/bel/belvedere.csv")
id_column = "Identifier"
title_column = "Title"
color_column = "ExpertTags"
museum_data = museum_data[[id_column, title_column, color_column]]
museum_data = museum_data.astype({id_column: "str"})

# # wm df
# museum_data = pd.read_csv("data/wm/wien_museum.csv")
# id_column = "id"
# title_column = "title"
# color_column = "subjects"
# museum_data = museum_data[[id_column, title_column, color_column]]
# museum_data = museum_data.assign(museum = "wm")
# museum_data = museum_data.rename({id_column:"id", title_column:"title", color_column:"info_1"}, axis=1)
#################################################################


#################################################################
# Create dataset
# specify root directory that contains the images
size=224
transform = transforms.Compose([transforms.CenterCrop(size), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
eval_dataset = ImageDataset(root_dir=root_dir, transform=transform)
eval_set_size = len(eval_dataset)
eval_dataloader = torch.utils.data.DataLoader(eval_dataset, batch_size=1, shuffle=False)
####################################################################



####################################################################
# evaluate model
triplet_model = torch.load(log_dir + run_name + "/triplet_net.pt", map_location='cpu')
#triplet_model = triplet_model.to(device="cuda")

# create image embedding
with torch.no_grad():
    image_embedding_list = []
    id_list = []
    museum_list = []
    unique = set()
    for img_id, img in tqdm(eval_dataloader):
        if img_id[0] not in unique:
            #img = img.to(device="cuda")
            encoded_img = triplet_model.encode(img)
            image_embedding_list.append(encoded_img.numpy()[0])
            unique.add(img_id[0])
            id_list.append(img_id[0])

image_embedding = embedding.Embedding(np.array(image_embedding_list), np.array(id_list, dtype=object))
embedding.save_to_csv(image_embedding, evaluation_dir + "/image_embedding")
####################################################################

image_embedding = embedding.load_csv(evaluation_dir + "/image_embedding.csv")

##################################################################
#make 3d plots
n = 3000

# plot sentence embedding
embedding_loaded = embedding.load_csv("data/bel/bel_sbert_Title_Description_ExpertTags_512d.csv")
figure_1 = embedding.plot_3d(embedding_loaded, museum_data, n, 
                 id_column, title_column, color_column, [], 
                "Sentence Embedding")

figure_1.write_html(evaluation_dir + "/Sentence_Embedding.html")

# take only samples from df where ids are in id_list
meta_data = museum_data.loc[museum_data[id_column].isin(image_embedding.identifier)]
# plot image embedding
figure_2 = embedding.plot_3d(image_embedding, meta_data, n, 
               id_column, title_column, color_column, [], 
               f"Image Embedding stand: \n {run_name}", True, window_shape=(2000,750))

figure_2.write_html(evaluation_dir + "/Image_Embedding_stand.html")

# plot image embedding
figure_3 = embedding.plot_3d(image_embedding, meta_data, n, 
               id_column, title_column, color_column, [], 
               f"Image Embedding: \n {run_name}", False, window_shape=(2000,750))

figure_3.write_html(evaluation_dir + "/Image_Embedding.html")
###################################################################



####################################################################
# create triplets from existing image embedding and plots some
# generate triplets
n = 2000
triplets = triplet.generate_triplets(image_embedding, "clustering", n)
with open(evaluation_dir + '/triplets', 'wb') as fp:
    pickle.dump(triplets, fp)
####################################################################



##################################################################
# load already created image_triplets
with open(evaluation_dir + '/triplets', 'rb') as fp:
    image_triplets_loaded = pickle.load(fp)

useable_triplets = []
for ori, sim, dis in image_triplets_loaded:

    split_ori = ori.split("/")
    split_sim = sim.split("/")
    split_dis = dis.split("/")

    useable_triplets.append(("__@@__".join(split_ori), "__@@__".join(split_sim), "__@@__".join(split_dis)))

dataset = TripletDataset(triplets = useable_triplets,
                         root_dir = root_dir)

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