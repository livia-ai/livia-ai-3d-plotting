import torch
torch.cuda.empty_cache() 
from torchvision import transforms

import livia.embedding as embedding
import livia.triplet as triplet
from dataset import ImageDataset

import pandas as pd
import pickle
import os

import utility_functions  as uf

if torch.cuda.is_available(): 
    device = "cuda" 
else: 
    device = "cpu"   


#################################################################
# specify run_name + log_dir + eval_dir + root directory

# root_dir -> where images are stored
root_dir = 'data/images/wm_cropped'

# log_dir + run_name
log_dir = "experiments/runs/"
run_name = 'wm_pretrained_triplets=149996_size=224_bs=6_margin=1_epochs=5_not_normalized'

# directory where the results are stored
evaluation_dir = f"evaluation_results/{run_name}"
# create evaluation dir
if not os.path.isdir("evaluation_results"):
    os.mkdir("evaluation_results")
if not os.path.isdir(evaluation_dir):
    os.mkdir(evaluation_dir)

#################################################################
# load sentence embedding
# wm
# embedding_loaded = embedding.load_csv("data/wm/wm_sbert_title_districts_subjects_512d.csv")
# bel
# embedding_loaded = embedding.load_csv("data/bel/bel_sbert_Title_Description_ExpertTags_256d.csv")
#################################################################
# load museums data for plotting

# bel dataframe
#museum_data = pd.read_csv("data/bel/belvedere.csv")
#id_column = "Identifer"
#title_column = "Title"
#color_column = "ExpertTags"
#museum_data = museum_data[[id_column, title_column, color_column]]
#museum_data = museum_data.astype({id_column: "str"})

 # wm dataframe
museum_data = pd.read_csv("data/bel/belvedere.csv")
id_column = "Identifier"
title_column = "Title"
color_column = "ExpertTags"
museum_data = museum_data[[id_column, title_column, color_column]]
museum_data = museum_data.astype({id_column: "str"})
#################################################################


##################################################################
## Create dataset

## specify root directory that contains the images
#size=224
#transform = transforms.Compose([transforms.CenterCrop(size), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
#eval_dataset = ImageDataset(root_dir=root_dir, transform=transform)
#eval_dataloader = torch.utils.data.DataLoader(eval_dataset, batch_size=1, shuffle=False)
#####################################################################


#####################################################################
## Evaluate Model

## load model
#triplet_model = torch.load(log_dir + run_name + "/triplet_net.pt", map_location=device)
## create image embedding
#image_embedding = uf.compute_image_embedding(triplet_model, device, eval_dataloader)
## save image embedding as csv file
#embedding.save_to_csv(image_embedding, evaluation_dir + "/image_embedding")

#####################################################################

### load image embedding if necessary
##image_embedding = embedding.load_csv(evaluation_dir + "/image_embedding.csv")

###################################################################
##make 3d plots
#n = 3000
## plot sentence embedding
#figure_1 = embedding.plot_3d(embedding_loaded, museum_data, n, 
#                 id_column, title_column, color_column, [], 
#                "Sentence Embedding")
#figure_1.write_html(evaluation_dir + "/Sentence_Embedding.html")

## take only samples from df where ids are in id_list
#meta_data = museum_data.loc[museum_data[id_column].isin(image_embedding.identifier)]

## plot image embedding standardized
#figure_2 = embedding.plot_3d(image_embedding, meta_data, n, 
#               id_column, title_column, color_column, [], 
#               f"Image Embedding stand: \n {run_name}", True, window_shape=(2000,750))
#figure_2.write_html(evaluation_dir + "/Image_Embedding_stand.html")

## plot image embedding normal
#figure_3 = embedding.plot_3d(image_embedding, meta_data, n, 
#               id_column, title_column, color_column, [], 
#               f"Image Embedding: \n {run_name}", False, window_shape=(2000,750))
#figure_3.write_html(evaluation_dir + "/Image_Embedding.html")
####################################################################


#####################################################################
## create triplets from existing image embedding and plots some
## generate triplets
#n = 5000
#triplets = triplet.generate_triplets(image_embedding, "clustering", n)
#with open(evaluation_dir + f'/{n}_triplets', 'wb') as fp:
#    pickle.dump(triplets, fp)
#####################################################################



##################################################################
# load already created image_triplets
n = 5000
with open(evaluation_dir + f'/{n}_triplets', 'rb') as fp:
    image_triplets_loaded = pickle.load(fp)

# plot some triplets
uf.display_triplets(image_triplets_loaded, root_dir, evaluation_dir)
##################################################################