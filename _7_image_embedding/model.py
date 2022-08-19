import torch
from torch import nn
from tqdm import tqdm
import numpy as np
import torch.nn.functional as F


class EmbeddingNet(torch.nn.Module):
    def __init__(self):
        super(EmbeddingNet, self).__init__() 

        self.pool = nn.MaxPool2d(2,2)
        self.relu = nn.ReLU(True)
        self.sigmoid = nn.Sigmoid()
        self.flatten = nn.Flatten()

        # encoder layers
        self.conv1 = nn.Conv2d(3, 9, 3)
        self.conv2 = nn.Conv2d(9, 9, 3)
        self.conv3 = nn.Conv2d(9, 5, 3)

        #self.conv4 = nn.Conv2d(16, 16, 7)
        
        #self.conv5 = nn.Conv2d(16, 16, 5)
        #self.conv6 = nn.Conv2d(16, 4, 5)
        
        self.lin1 = nn.Linear(2420, 2048)
        self.lin2 = nn.Linear(2048,128)

    def forward(self, x):

        # encoder
        h = self.relu(self.conv1(x))
        h = self.relu(self.conv2(h))
        h = self.relu(self.conv3(h))
        h = self.pool(h)

        #h = self.relu(self.conv3(h))
        #h = self.relu(self.conv4(h))
        #h = self.pool(h)        
        
        #h = self.relu(self.conv5(h))
       # h = self.relu(self.conv6(h))
        #h = self.pool(h)
            
        h = self.relu(self.lin1(self.flatten(h)))
        h = self.lin2(h)

        # L2 norm at the end
       # h = F.normalize(h, p=2, dim=1)
        #print(h.shape)
            
        return h   
    
class TripletNet(nn.Module):
    def __init__(self, embedding_model):
        super(TripletNet, self).__init__()
        self.embedding_model = embedding_model

    def forward(self, anchor, pos, neg):
        embedded_anchor = self.embedding_model(anchor)
        embedded_pos = self.embedding_model(pos)
        embedded_neg = self.embedding_model(neg)
        
        return embedded_anchor, embedded_pos, embedded_neg

    def encode(self, image):
        return self.embedding_model(image)

def train(model, dataloader, progress_bar, loss_fn, optimizer, writer):

    train_loss = list()
    # store number of updates
    j = 0
    for i in progress_bar:

        epoch_loss = list()
        for anchor, positive, negative, _ in dataloader:
            
            anchor = anchor.to(device="cuda")
            pos = positive.to(device="cuda")
            neg = negative.to(device="cuda")

            anch_hidden, pos_hidden, neg_hidden = model(anchor, pos, neg)
            loss = loss_fn(anch_hidden, pos_hidden, neg_hidden)     
            
            loss.backward()
            optimizer.step()
            
            j += 1
            
            optimizer.zero_grad()
            
            loss_info = loss.cpu().detach().numpy()
            
            epoch_loss.append(loss_info)
            
            if j%10==0:
                writer.add_scalar('training/loss',
                        loss_info,
                        j)

        torch.save(model, writer.log_dir + "/triplet_net.pt")

        # compute epoch loss and write into progress bar
        loss_str = str(np.around(np.mean(epoch_loss),5))

        progress_bar.set_postfix_str(loss_str)
        
        # tensorboard
        writer.add_scalar('training/mean epoch loss',
                        np.around(np.mean(epoch_loss),5),
                        i)
        
        # save epoch loss
        train_loss.append(np.mean(epoch_loss))

    return model, train_loss 
