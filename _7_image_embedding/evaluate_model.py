import torch
torch.cuda.empty_cache() 
from torchvision import transforms

import livia.embedding as embedding
import livia.triplet as triplet
from dataset import ImageDataset

import pandas as pd
import pickle
import os 
from sklearn.decomposition import PCA
import utility_functions  as uf

if torch.cuda.is_available(): 
    device = "cuda" 
else: 
    device = "cpu"

#################################################################
# specify museum
museum = "bel"
# specify run_name + log_dir + eval_dir + root directory
# root_dir -> where images are stored
root_dir = f'data_local/images/{museum}_cropped'
# log_dir + run_name
log_dir = "experiments/runs/from_server/"
run_name = 'grayscale_wm_pretrained_unfrozen_triplets=249996_size=224_bs=32_margin=1_epochs=10_lr=0.0001_noise=0.05'

# directory where the results are stored
evaluation_dir = f"evaluation_results/{run_name}/{museum}"
# create evaluation dir
if not os.path.isdir("evaluation_results"):
    os.mkdir("evaluation_results")
if not os.path.isdir(f"evaluation_results/{run_name}"):
    os.mkdir(f"evaluation_results/{run_name}")
if not os.path.isdir(evaluation_dir):
    os.mkdir(evaluation_dir)

#################################################################
# load museums data
if museum == "bel":
    embedding_loaded = embedding.load_csv("data_local/bel/bel_sbert_Title_Description_ExpertTags_256d.csv")
    museum_data = pd.read_csv("data_local/bel/belvedere.csv")
    id_column = "Identifier"
    title_column = "Title"
    color_column = "ExpertTags"
    museum_data = museum_data[[id_column, title_column, color_column]]
    museum_data = museum_data.astype({id_column: "str"})

if museum == "wm":
    embedding_loaded = embedding.load_csv("data_local/wm/wm_sbert_title_districts_subjects_512d.csv")
    museum_data = pd.read_csv("data_local/wm/wien_museum.csv")
    id_column = "id"
    title_column = "title"
    color_column = "subjects"
    museum_data = museum_data[[id_column, title_column, color_column]]
    museum_data = museum_data.astype({id_column: "str"})
#################################################################


###################################################################
# Create dataset
# specify root directory that contains the images
size=224
# colored
# transform = transforms.Compose([transforms.CenterCrop(size), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
# grayscale
transform = transforms.Compose([transforms.Grayscale(num_output_channels=1), transforms.ToTensor()])
eval_dataset = ImageDataset(root_dir=root_dir, transform=transform)
eval_dataloader = torch.utils.data.DataLoader(eval_dataset, batch_size=128, shuffle=False)
#####################################################################


######################################################################
# Evaluate Model
# load model
triplet_model = torch.load(log_dir + run_name + "/triplet_net.pt", map_location=device)
# create image embedding
image_embedding = uf.compute_image_embedding(triplet_model, device, eval_dataloader)
# save image embedding as csv file
embedding.save_to_csv(image_embedding, evaluation_dir + "/image_embedding")
#####################################################################

# load image embedding if necessary
#image_embedding = embedding.load_csv(evaluation_dir + "/image_embedding.csv")


###################################################################
#PCA 
pca = PCA(n_components=128)
pca.fit(image_embedding.embedding)
print(pca.explained_variance_ratio_)
print(pca.explained_variance_)
###################################################################


###################################################################
#make 3d plots
n = 3000
# plot sentence embedding
figure_1 = embedding.plot_3d(embedding_loaded, museum_data, n, 
                 id_column, title_column, color_column, [], 
                "Sentence Embedding")
figure_1.write_html(evaluation_dir + "/Sentence_Embedding.html")

# take only samples from df where ids are in id_list
meta_data = museum_data.loc[museum_data[id_column].isin(image_embedding.identifier)]

# plot image embedding standardized
figure_2 = embedding.plot_3d(image_embedding, meta_data, n, 
               id_column, title_column, color_column, [], 
               f"Image Embedding stand: \n {run_name}", True, window_shape=(2000,750))
figure_2.write_html(evaluation_dir + "/Image_Embedding_stand.html")

# plot image embedding normal
figure_3 = embedding.plot_3d(image_embedding, meta_data, n, 
               id_column, title_column, color_column, [], 
               f"Image Embedding: \n {run_name}", False, window_shape=(2000,750))
figure_3.write_html(evaluation_dir + "/Image_Embedding.html")
###################################################################


#####################################################################
# create triplets from existing image embedding and plot some
# generate triplets
n = 5000
triplets = triplet.generate_triplets(image_embedding, "clustering", n, nn_k=1, fn_k=5)#, nr_clusters=8, nr_random_sample=10000)
with open(evaluation_dir + f'/{n}_triplets', 'wb') as fp:
    pickle.dump(triplets, fp)
#####################################################################



##################################################################
# load already created image_triplets
n = 5000
with open(evaluation_dir + f'/{n}_triplets', 'rb') as fp:
    image_triplets_loaded = pickle.load(fp)

# plot some triplets
uf.display_triplets(image_triplets_loaded, root_dir, evaluation_dir)
##################################################################