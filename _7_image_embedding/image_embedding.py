# imports
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
import livia.embedding as embedding

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

from dataset import TripletDataset
from model import EmbeddingNet, TripletNet, train

torch.cuda.empty_cache() 

#################################################################
# create dataset
size = 224
batch_size = 6
# specify root directory that contains the images
root_dir = 'data/images/bel_cropped'
# load triplets of image paths
with open(f'bel_image_paths', 'rb') as fp:
    image_path_triplets = pickle.load(fp)
# specify transforms
transform = transforms.Compose([transforms.CenterCrop(size), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
# generate train and test dataset
train_dataset = TripletDataset(samples = image_path_triplets, root_dir = root_dir, transform = transform)
# create dataloader
# sampler = torch.utils.data.RandomSampler(train_dataset, num_samples=25000)
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=False)
###################################################################



###################################################################
# Instantiate model
# use pretrained model and just change head
model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=True)
num_ftrs = model.fc.in_features
model.fc = nn.Sequential(nn.Linear(num_ftrs, num_ftrs), nn.ReLU(),nn.Linear(num_ftrs, 128))
## freeze all layers except the fully connected layers
#or param in model.parameters():
#    param.requires_grad = False
#for param in list(model.parameters())[-4:]:
#    param.requires_grad = True
#model = EmbeddingNet()
triplet_net = TripletNet(model).to(device="cuda")
#for anchor, pos, neg, ori_path in test_dataloader:
#    print("in:", anchor.shape)
#    anchor = anchor.to(device="cuda")
#    pos = pos.to(device="cuda")
#    neg = neg.to(device="cuda")
#    anch_hidden, pos_hidden, neg_hidden = triplet_net(anchor, pos, neg)
#    print("h:", anch_hidden.shape)
#    break
#################################################################



#################################################################
# train model
# hyperparameters
lr = 1e-4
n_epochs = 15
margin = 1
wd = 0
# progress bar
progress_bar = tqdm(range(n_epochs))
# optimizer
optimizer = torch.optim.Adam(triplet_net.parameters(), lr=lr, weight_decay=wd)
# triplet loss
triplet_loss = nn.TripletMarginWithDistanceLoss(distance_function=lambda x, y: 1-F.cosine_similarity(x, y), margin=margin) #torch.nn.TripletMarginLoss(margin=margin)
# tensorboard writer
log_dir = "experiments/runs/"
run_name = f'pretrained_triplets={len(image_path_triplets)}_size={size}_bs={batch_size}_margin={margin}_epochs={n_epochs}_not_normalized'
writer_log_path = log_dir + run_name
writer = SummaryWriter(writer_log_path)
# train model
trained_triplet_net, epoch_losses = train(model = triplet_net,
                                    dataloader = train_dataloader,
                                    progress_bar = progress_bar,
                                    loss_fn = triplet_loss,
                                    optimizer = optimizer,
                                    writer = writer)
writer.close()
torch.save(trained_triplet_net, writer_log_path + "/triplet_net.pt")
#################################################################



#################################################################
# create image embedding
museum_data = pd.read_csv("data/bel/belvedere.csv")
id_column = "Identifier"
title_column = "Title"
color_column = "ExpertTags"
museum_data = museum_data[[id_column, title_column, color_column]]
museum_data = museum_data.astype({id_column: "str"})
# load sentece embedding that should be used for computing triplets
embedding_loaded = embedding.load_csv("data/bel/bel_sbert_Title_Description_ExpertTags_512d.csv")
#print("Iterating over images...")
with torch.no_grad():
    image_embedding_list = []
    id_list = []
    unique = set()
    for imgs in tqdm(test_dataloader):
        img, _,_, paths = imgs
        sample_id = paths[0][0].split(".")[0]

        if sample_id not in unique:
            img = img.to(device="cuda")
            encoded_img = trained_triplet_net.encode(img)
            image_embedding_list.append(encoded_img.cpu().numpy()[0])

            unique.add(sample_id)
            id_list.append(sample_id)

#print(len(image_embedding_list))
image_embedding = embedding.Embedding(np.array(image_embedding_list), np.array(id_list, dtype=object))
embedding.save_to_csv(image_embedding, f"bel_image_embedding_{len(image_path_triplets)}")
#################################################################



#################################################################
image_embedding = embedding.load_csv("data/bel/bel_imagemb_pretrained_not_frozen.csv")
# make 3d plots
n = 5000
# plot sentence embedding
embedding.plot_3d(embedding_loaded, museum_data, n, 
                  id_column, title_column, color_column, [], 
                  "Sentence Embedding")
# take only samples from df where ids are in id_list
meta_data = museum_data.loc[museum_data[id_column].isin(id_list)]
# plot image embedding
embedding.plot_3d(image_embedding, meta_data, n, 
                id_column, title_column, color_column, [], 
                f"Image Embedding stand: \n {run_name}", True, window_shape=(2000,750))
# plot image embedding
embedding.plot_3d(image_embedding, meta_data, n, 
                id_column, title_column, color_column, [], 
                f"Image Embedding: \n {run_name}", False, window_shape=(2000,750))
#################################################################