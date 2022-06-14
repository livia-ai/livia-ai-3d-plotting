import torch
torch.cuda.empty_cache() 
from torch import nn
import torch.nn.functional as F
from torchvision import transforms

import livia.embedding as embedding
import livia.triplet as triplet
from dataset import TripletDataset, ImageDataset
from model import EmbeddingNet,TripletNet, train
from torch.utils.tensorboard import SummaryWriter

from tqdm import tqdm
import pandas as pd
import numpy as np
import pickle

#################################################################
# Create datasetembedding_loaded = embedding.load_csv("data/wm/wm_sbert_title_districts_subjects_256d.csv")
size = 50
batch_size = 16

# specify root directory that contains the images
root_dir = 'data/test_images/wm_cropped_train'

# specify transforms
transform = transforms.Compose([transforms.ToTensor(), transforms.CenterCrop(size)])

## load triplets
#triplets = []
#for i in range(6):
#    with open(f'data/triplets/15000/triplets_{i}', 'rb') as fp:
#        triplets_loaded = pickle.load(fp)
#        triplets.extend(triplets_loaded)
with open(f'image_paths_for_dataset', 'rb') as fp:
    image_path_triplets = pickle.load(fp)[:2000]

trainset_size = len(image_path_triplets)

## generate train and test dataset
train_dataset = TripletDataset(samples = image_path_triplets,
                               root_dir = root_dir,
                               transform=transform)

##sampler = torch.utils.data.RandomSampler(train_dataset, num_samples=25000)
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
##################################################################


##################################################################
# train model

# hyperparameters
lr = 1e-5
n_epochs = 150
margin = 1.5

# instantiate model
emb_net = EmbeddingNet()
triplet_net = TripletNet(emb_net).to(device="cuda")

#for anchor, pos, neg in train_dataloader:

#    print("in:", anchor.shape)

#    anchor = anchor.to(device="cuda")
#    pos = pos.to(device="cuda")
#    neg = neg.to(device="cuda")

#    anch_hidden, pos_hidden, neg_hidden = triplet_net(anchor, pos, neg)

#    print("h:", anch_hidden.shape)
#    break


# progress bar
progress_bar = tqdm(range(n_epochs))

# optimizer
wd = 0
optimizer = torch.optim.Adam(triplet_net.parameters(), lr=lr, weight_decay=wd)

# triplet loss
#triplet_loss = torch.nn.TripletMarginLoss(margin=margin)
triplet_loss = nn.TripletMarginWithDistanceLoss(distance_function=lambda x, y: 1-F.cosine_similarity(x, y), margin=margin)

# tensorboard writer
log_dir = "experiments/runs/"
writer_log_path = log_dir + f'centercropped={size}_triplets={trainset_size}_bs={batch_size}_margin={margin}_lr={lr}_wd={wd}'
#writer_log_path = log_dir + "test_script"
writer = SummaryWriter(writer_log_path)

trained_triplet_net, epoch_losses = train(model = triplet_net,
                                    dataloader = train_dataloader,
                                    progress_bar = progress_bar,
                                    loss_fn = triplet_loss,
                                    optimizer = optimizer,
                                    writer = writer)

torch.save(trained_triplet_net, writer_log_path + "/triplet_net.pt")
#################################################################


#################################################################
# plot results

# create image embedding
wm_data = pd.read_csv("data/wm/wien_museum.csv")
wm_data = wm_data[["id", "title", "subjects"]]
wm_data = wm_data.astype({"id": "str"})

# load sentece embedding that should be used for computing triplets
embedding_loaded = embedding.load_csv("data/wm/wm_sbert_title_districts_subjects_512d.csv")

img_dataset = ImageDataset(wm_data["id"], root_dir, transform)
img_dataloader = torch.utils.data.DataLoader(img_dataset, batch_size=1, shuffle=False)

triple_net = trained_triplet_net.to(device="cpu")

print("iterating over images...")
with torch.no_grad():
    image_embedding_list = []
    for img in img_dataloader:
        img = img
        encoded_img = trained_triplet_net.encode(img)
        image_embedding_list.append(encoded_img.numpy()[0])

image_embedding = embedding.Embedding(np.array(image_embedding_list), np.array(wm_data["id"]))
embedding.save_to_csv(image_embedding, "image_embedding")
# plot sentence embedding
embedding.plot_3d(embedding_loaded, wm_data, 3000, 
                  "id", "title", "subjects", [], 
                  "Sentence Embedding")

# plot image embedding
embedding.plot_3d(image_embedding, wm_data, 3000, 
                  "id", "title", "subjects", [], 
                  "Image Embedding", True)

# plot image embedding
embedding.plot_3d(image_embedding, wm_data, 3000, 
                  "id", "title", "subjects", [], 
                  "Image Embedding", False)

