import torch
torch.cuda.empty_cache() 
from torch import nn
import torch.nn.functional as F
from torchvision import transforms

import livia.embedding as embedding
from dataset import TripletDataset
from model import EmbeddingNet,TripletNet, train
from torch.utils.tensorboard import SummaryWriter

from tqdm import tqdm


#################################################################
# Create dataset

# hyperparameters
size = 224
batch_size = 32
trainset_size = 2000
testset_size = 10


# specify root directory that contains the images
root_dir = 'data/test_images/wm_cropped_train'
# load sentece embedding that should be used for computing triplets
embedding_loaded = embedding.load_csv("data/wm/wm_sbert_title_districts_subjects_256d.csv")

# specify transforms
transform = transforms.Compose([transforms.ToTensor(), transforms.CenterCrop(size)])

# generate train and test dataset
train_dataset = TripletDataset(sentence_embedding=embedding_loaded, 
                         root_dir=root_dir,
                         n = trainset_size,
                         transform=transform)

test_dataset = TripletDataset(sentence_embedding=embedding_loaded, 
                         root_dir=root_dir,
                         n = testset_size,
                         transform=transform)

train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
#################################################################

#################################################################
# train model

# hyperparameters
lr = 1e-4
n_epochs = 100
margin=1

# instantiate model
emb_net = EmbeddingNet()
triplet_net = TripletNet(emb_net).to(device="cuda")

# progress bar
progress_bar = tqdm(range(n_epochs))

# optimizer
optimizer = torch.optim.Adam(triplet_net.parameters(), lr=lr)

# triplet loss
#triplet_loss = torch.nn.TripletMarginLoss(margin=margin)
triplet_loss = nn.TripletMarginWithDistanceLoss(distance_function=lambda x, y: 1-F.cosine_similarity(x, y), margin=margin)

# tensorboard writer
log_dir = "experiments/runs/"
writer_log_path = log_dir + f'triplet_network_cosinedist_margin={margin}_lr={lr}' #_centercropped={size}
#writer_log_path = log_dir + "test_script"
writer = SummaryWriter(writer_log_path)

triplet_net, epoch_losses = train(model = triplet_net,
                                    dataloader = train_dataloader,
                                    progress_bar = progress_bar,
                                    loss_fn = triplet_loss,
                                    optimizer = optimizer,
                                    writer = writer)

torch.save(triplet_net, writer_log_path + "/triplet_net.pt")




#for anchor, pos, neg in train_dataloader:

#    print("in:", anchor.shape)

#    anchor = anchor.to(device="cuda")
#    pos = pos.to(device="cuda")
#    neg = neg.to(device="cuda")
#    anch_hidden, pos_hidden, neg_hidden = model(anchor, pos, neg)

#    print("h:", anch_hidden.shape)
#    break





##print(train_loss)
#with torch.no_grad():
#    model = model.to("cpu")
#    for image, museum in train_dataloader:
#        output, h = model(image)

#        combined = torch.stack([image[0],output[0]])
#        writer.add_images('Target vs Reconstruction',combined)

#        #print(f"Max: image:{torch.max(image)}, output:{torch.max(output)}")
#        #print(f"Min: image:{torch.min(image)}, output:{torch.min(output)}")

#        #fig, ax = plt.subplots(1,2, figsize=(6,20))
#        #ax[0].imshow(image[0].permute(1,2,0))
#        #ax[1].imshow(output[0].permute(1,2,0))
#        ##plt.show()

#        #writer.add_figure('Target vs Reconstruction figure',fig)
#        break
        
#writer.close()