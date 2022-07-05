import torch
torch.cuda.empty_cache() 
from torch import nn
import torch.nn.functional as F
from torchvision import transforms

import livia.embedding as embedding
import livia.triplet as triplet
from dataset import TripletDataset, ImageDataset, MixedImageDataset
from torch.utils.tensorboard import SummaryWriter

from tqdm import tqdm
import pandas as pd
import numpy as np
import pickle
import os

#################################################################
# specify run_name + log_dir + eval_dir
log_dir = "experiments/bel/"
run_name = f'pretrained_triplets=26676_bs=6_margin=1_lr=0.0001_wd=0'
evaluation_dir = f"evaluation_results_mixed/{run_name}"

# create evaluation dir
if not os.path.isdir(evaluation_dir):
    os.mkdir(evaluation_dir)
#################################################################


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


#################################################################
# Create dataset
# specify root directory that contains the images
root_dir = 'data/images/mixed_cropped'
size=224
transform = transforms.Compose([transforms.CenterCrop(size), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
eval_dataset = MixedImageDataset(root_dir=root_dir, transform=transform)
eval_set_size = len(eval_dataset)
eval_dataset[0]
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
   for museum_id, img_id, img in tqdm(eval_dataloader):

       if img_id[0] not in unique:
           #img = img.to(device="cuda")
           encoded_img = triplet_model.encode(img)
           image_embedding_list.append(encoded_img.numpy()[0])
           unique.add(img_id[0])
           id_list.append(img_id[0] + "_" + museum_id[0])

image_embedding = embedding.Embedding(np.array(image_embedding_list), np.array(id_list, dtype=object))
embedding.save_to_csv(image_embedding, evaluation_dir + "/image_embedding")
####################################################################

image_embedding = embedding.load_csv(evaluation_dir + "/image_embedding.csv")



# ##################################################################
# #make 3d plots
# n = 5000

# # plot sentence embedding
# embedding_loaded = embedding.load_csv("data/bel/bel_sbert_Title_Description_ExpertTags_512d.csv")
# figure_1 = embedding.plot_3d(embedding_loaded, museum_data, n, 
#                  id_column, title_column, color_column, [], 
#                 "Sentence Embedding")

# figure_1.write_html(evaluation_dir + "/Sentence_Embedding.html")

# # take only samples from df where ids are in id_list
# meta_data = museum_data.loc[museum_data[id_column].isin(image_embedding.identifier)]
# # plot image embedding
# figure_2 = embedding.plot_3d(image_embedding, meta_data, n, 
#                id_column, title_column, color_column, [], 
#                f"Image Embedding stand: \n {run_name}", True, window_shape=(2000,750))

# figure_2.write_html(evaluation_dir + "/Image_Embedding_stand.html")

# # plot image embedding
# figure_3 = embedding.plot_3d(image_embedding, meta_data, n, 
#                id_column, title_column, color_column, [], 
#                f"Image Embedding: \n {run_name}", False, window_shape=(2000,750))

# figure_3.write_html(evaluation_dir + "path/to/Image_Embedding.html")
# ###################################################################



# ####################################################################
# # create triplets from existing image embedding and plots some
# # generate triplets
# n = 1000
# triplets = triplet.generate_triplets(image_embedding, "clustering", n)
# with open(f'data/bel/bel_image_triplets_not_frozen', 'wb') as fp:
#     pickle.dump(triplets, fp)
# ####################################################################



# ##################################################################
# # load already created image_triplets
# root_dir = "data/images/bel_cropped"
# with open(f'data/bel/bel_image_triplets_not_frozen', 'rb') as fp:
#     image_triplets_loaded = pickle.load(fp)

# useable_triplets = []
# for ori, sim, dis in image_triplets_loaded:

#     split_ori = ori.split("/")
#     split_sim = sim.split("/")
#     split_dis = dis.split("/")

#     useable_triplets.append(("__@@__".join(split_ori), "__@@__".join(split_sim), "__@@__".join(split_dis)))

# from dataset import TripletDataset
# dataset = TripletDataset(triplets = useable_triplets,
#                          root_dir = root_dir,)

# import matplotlib.pyplot as plt
# for triplet in dataset: 

#     print(triplet[3])

#     fig,ax = plt.subplots(1,3)
#     ax[0].imshow(triplet[0])
#     ax[1].imshow(triplet[1])
#     ax[2].imshow(triplet[2])

#     ax[0].set_title("Sample")
#     ax[1].set_title("Similar")
#     ax[2].set_title("Dissimilar")

#     plt.show()
# ##################################################################



# ##################################################################
# mixed dataset plots
mixed_embeddin
if mixed_embedding.shape[1] > 3:
    embedding_to_plot = pca_reduce(embedding_to_plot,3, standardize=standardize)

# make column list is unique
column_list = [id_column, title_column, color_column] + info_columns
columns_unique = list(dict.fromkeys(column_list))

# dataframe that contains meta data
df_meta = meta_data[columns_unique]
df_meta = df_meta.astype({id_column: "str"})

#df that contains the 3d embedding
df_emb_emb = pd.DataFrame(embedding_to_plot.embedding, columns = ["x", "y", "z"])
df_emb_id = pd.DataFrame(embedding_to_plot.identifier, columns = [id_column])
df_emb = df_emb_id.join(df_emb_emb)

embedding_length = len(df_emb)

# if specified take random subsample of dataset
if n != None:
    if n < embedding_length:
        rng = np.random.default_rng()
        id_list = list(range(embedding_length))
        sample_ids = rng.choice(id_list, size=n, replace=False)
        df_emb = df_emb.copy().iloc[sample_ids]
        #df_meta = df_meta.copy().iloc[sample_ids]

# merge both dataframes based on id_column in case there is different ordering
df = pd.merge(df_emb, df_meta, on=id_column)

# for better visualization crop title
length = 75
df[title_column] = df[title_column].apply(lambda x: str(x)[:length] if len(str(x))>length else str(x))
df[color_column] = df[color_column].apply(lambda x: str(x)[:100] if len(str(x))>100 else str(x))
df.fillna('NaN', inplace=True)

fig = px.scatter_3d(df, 
                x='x', y='y', z='z', 
                color=color_column, 
                hover_name=title_column, # what to show when hovered over
                hover_data=[id_column] + info_columns,
                width=window_shape[0], height=window_shape[1], # adjust height and width
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
