# imports
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

import utility_functions as uf
from model import train


device = "cuda:1"

########################################
# read out parameters
model_dir = "models_to_retrain/"
model_name = 'grayscale_wm_pretrained_unfrozen_triplets=249996_size=224_bs=32_margin=1_epochs=10_lr=0.0001_noise=[0.005, 0.0, 0.0]'
for info in model_name.split("_"):

    split_by_equals = info.split("=")

    if len(split_by_equals) == 2:
        
        hyper_parameter = split_by_equals[0]
        value = split_by_equals[1]

        if hyper_parameter == "triplets":
            triplets = int(value)

        elif hyper_parameter == "size":
            size = int(value)

        elif hyper_parameter ==  "bs": 
            batch_size = int(value)
        
        elif hyper_parameter == "margin":
            margin = float(value)
        
        elif hyper_parameter == "epochs":
            start_epoch = int(value)

        elif hyper_parameter == "lr":
            lr = float(value)

        elif hyper_parameter == "noise":
            noise = [float(x) for x in value.strip("[]").split(",")]

        else:
            continue


########################################
# prepare dataset

# parameters
image_dir = 'data/wm_cropped'
triplets_image_paths_dir = 'data/wm_triplets_nnk=25_fnk=500/image_paths'

# create dataset
train_dataloader, len_dataset = uf.prepare_grayscale_data(image_dir, triplets_image_paths_dir, batch_size)


########################################
# load model 
model = torch.load(model_dir + model_name + "/triplet_net.pt", map_location=device)

# hyperparameters
n_epochs = 10

# progress bar
progress_bar = tqdm(range(start_epoch, start_epoch + n_epochs))
# optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0)
# triplet loss
triplet_loss = nn.TripletMarginWithDistanceLoss(distance_function=lambda x, y: 1-F.cosine_similarity(x, y), margin=margin) #torch.nn.TripletMarginLoss(margin=margin)

# tensorboard writer
writer_log_path = model_dir + model_name
writer = SummaryWriter(writer_log_path)

model_name = f"triplet_net_retrained_n_epochs={n_epochs}.pt"

print(model_name)
retrained_model, epoch_losses = train(model = model,
                                    dataloader = train_dataloader,
                                    progress_bar = progress_bar,
                                    loss_fn = triplet_loss,
                                    optimizer = optimizer,
                                    writer = writer,
                                    noise = noise,
                                    device = device,
                                    model_name = model_name)

writer.close()
torch.save(retrained_model, writer_log_path + "/" + model_name)
