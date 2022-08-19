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
root_dir = 'data/images/wm_cropped'
# load triplets of image paths
with open(f'data/wm_triplets_nnk=25_fnk=500/image_paths', 'rb') as fp:
    image_path_triplets = pickle.load(fp)
# specify transforms
transform = transforms.Compose([transforms.CenterCrop(size), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
# generate train and test dataset
train_dataset = TripletDataset(samples = image_path_triplets, root_dir = root_dir, transform = transform)
# create dataloader
# sampler = torch.utils.data.RandomSampler(train_dataset, num_samples=25000)
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
###################################################################


###################################################################
# Instantiate model
# use pretrained model and just change head
model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=True)
num_ftrs = model.fc.in_features
model.fc = nn.Sequential(nn.Linear(num_ftrs, num_ftrs), nn.ReLU(),nn.Linear(num_ftrs, 1024), nn.ReLU(), nn.Linear(1024, 128) )
# freeze all layers except the fully connected layers

for param in model.parameters():
    param.requires_grad = True
#for param in list(model.parameters())[-54:]:
#    param.requires_grad = True

triplet_net = TripletNet(model).to(device="cuda")

##test model
#for anchor, pos, neg, ori_path in train_dataloader:
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
n_epochs = 10
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
run_name = f'wm_pretrained_unfrozen_triplets={len(image_path_triplets)}_size={size}_bs={batch_size}_margin={margin}_epochs={n_epochs}'
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