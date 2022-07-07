import torch
torch.cuda.empty_cache() 
from torchvision import transforms

import livia.embedding as embedding
import livia.triplet as triplet
from dataset import MixedTripletDataset, MixedImageDataset

from tqdm import tqdm
import pandas as pd
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
import plotly.express as px

import utility_functions as uf

if torch.cuda.is_available(): 
    device = "cuda" 
else: 
    device = "cpu"

#################################################################
# specify run_name + log_dir + eval_dir
log_dir = "experiments/runs/"
run_name = f'wm_pretrained_triplets=149996_size=224_bs=6_margin=1_epochs=5_not_normalized'
evaluation_dir = f"evaluation_results_mixed/{run_name}"
root_dir = 'data/images/mixed_cropped'

# create evaluation dir
if not os.path.isdir("evaluation_results_mixed"):
    os.mkdir("evaluation_results_mixed")
if not os.path.isdir(evaluation_dir):
    os.mkdir(evaluation_dir)
#################################################################


#################################################################
# randomly sample images from two datasets and copy to folder
k = 5000
# bel images
src = "data/images/bel_cropped"
folder_path = os.path.join(root_dir, "bel")
uf.sample_images(src, folder_path, k)
# wm images
src = "data/images/wm_cropped"
folder_path = os.path.join(root_dir, "wm")
uf.sample_images(src, folder_path, k)
#################################################################


#################################################################
# Create dataset
size=224
transform = transforms.Compose([transforms.CenterCrop(size), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
eval_dataset = MixedImageDataset(root_dir=root_dir, transform=transform)
eval_set_size = len(eval_dataset)
eval_dataloader = torch.utils.data.DataLoader(eval_dataset, batch_size=1, shuffle=False)
####################################################################


####################################################################
# evaluate model
triplet_model = torch.load(log_dir + run_name + "/triplet_net.pt", map_location=device)
triplet_model = triplet_model.to(device=device)

# create image embedding
with torch.no_grad():
   image_embedding_list = []
   id_list = []
   museum_list = []
   unique = set()
   for museum_id, img_id, img in tqdm(eval_dataloader):

       if img_id[0] not in unique:
           img = img.to(device=device)
           encoded_img = triplet_model.encode(img)
           image_embedding_list.append(encoded_img.cpu().numpy()[0])
           unique.add(img_id[0])
           id_list.append(img_id[0] + "@/@" + museum_id[0])

image_embedding = embedding.Embedding(np.array(image_embedding_list), np.array(id_list, dtype=object))
embedding.save_to_csv(image_embedding, evaluation_dir + "/image_embedding")
####################################################################

image_embedding = embedding.load_csv(evaluation_dir + "/image_embedding.csv")

#################################################################
# load museums data for plotting

# load wm df
bel_df = pd.read_csv("data/bel/belvedere.csv")
id_column = "Identifier"
title_column = "Title"
color_column = "ExpertTags"
bel_df = bel_df[[id_column, title_column, color_column]]
bel_df = bel_df.assign(museum = "bel")
bel_df = bel_df.astype({id_column: "str"})
bel_df = bel_df.rename({id_column:"id", title_column:"title", color_column:"info_1"}, axis=1)

# load bel df
wm_df = pd.read_csv("data/wm/wien_museum.csv")
id_column = "id"
title_column = "title"
color_column = "subjects"
wm_df = wm_df[[id_column, title_column, color_column]]
wm_df = wm_df.assign(museum = "wm")
wm_df = wm_df.rename({id_column:"id", title_column:"title", color_column:"info_1"}, axis=1)

# load mak df
#mak_filtered = mak[["priref", "title", "description"]]
#mak_filtered = mak_filtered.assign(museum = "mak")
#mak_filtered = mak_filtered.rename({"priref":"id", "title":"title", "description":"info_1"}, axis=1)

## combine dataframes
df_combined = pd.concat([wm_df, bel_df]).reset_index(drop=True)
#################################################################


##################################################################
# mixed dataset plots
# n
n=3000

if image_embedding.shape[1] > 3:
    embedding_to_plot = embedding.pca_reduce(image_embedding,3, standardize=True)

# dataframe that contains meta data
df_combined = df_combined.astype({id_column: "str"})

#df that contains the 3d embedding
df_emb_emb = pd.DataFrame(embedding_to_plot.embedding, columns = ["x", "y", "z"])
df_emb_id = pd.DataFrame([tuple(idx.split("@/@")) for idx in embedding_to_plot.identifier], columns = ["id", "museum"])
df_emb = df_emb_id.join(df_emb_emb)
embedding_length = len(df_emb)

# if specified take random subsample of dataset
if n != None:
    if n < embedding_length:
        rng = np.random.default_rng()
        id_list = list(range(embedding_length))
        sample_ids = rng.choice(id_list, size=n, replace=False)
        df_emb = df_emb.copy().iloc[sample_ids]

# merge both dataframes based on id_column in case there is different ordering
df = pd.merge(df_emb, df_combined, on=["id", "museum"])


# for better visualization crop title
length = 75
df["title"] = df["title"].apply(lambda x: str(x)[:length] if len(str(x))>length else str(x))
df["info_1"] = df["info_1"].apply(lambda x: str(x)[:100] if len(str(x))>100 else str(x))
df.fillna('NaN', inplace=True)

title_plot = f"WM and Bel combined \n Model:{run_name}"
fig = px.scatter_3d(df, 
                x='x', y='y', z='z', 
                color="museum", 
                hover_name="title", # what to show when hovered over
                hover_data=["id"] ,
                width=2000, height=800, # adjust height and width
                title=title_plot)

# make set size for legend and hover label
fig.update_layout(showlegend=True,
                    legend = dict(
                        font = dict(size = 10)
                        ), 
                hoverlabel=dict(
                        font_size=10,
                        ),
                title_x=0.5
                )
# set marker size
fig.update_traces(marker_size = 3)
fig.show()

fig.write_html(evaluation_dir + "/Image_Embedding.html")
####################################################################


####################################################################
# create triplets from existing image embedding and plots some
# generate triplets
n = 5000
triplets = triplet.generate_triplets(image_embedding, "clustering", n)
with open(evaluation_dir + '/{n}_triplets', 'wb') as fp:
    pickle.dump(triplets, fp)
####################################################################


##################################################################
# load already created image_triplets
with open(evaluation_dir + '/{n}_triplets', 'rb') as fp:
    image_triplets_loaded = pickle.load(fp)

uf.display_triplets_mixed(triplets, root_dir, evaluation_dir)

##################################################################