from turtle import xcor
import torch
from torch import nn


# Creating a DeepAutoencoder class
class ConvAutoencoder(torch.nn.Module):
    def __init__(self):
        super().__init__()  

        self.pool = nn.MaxPool2d(2,2)
        self.relu = nn.ReLU(True)
        self.sigmoid = nn.Sigmoid()
        self.flatten = nn.Flatten()

        # encoder layers
        self.en_conv1 = nn.Conv2d(3, 16, 7)
        self.en_conv2 = nn.Conv2d(16, 4, 5)
        self.en_lin1 = nn.Linear(4*20*20, 200)
        self.en_lin2 = nn.Linear(200, 25)

        # decoder layers
        self.de_lin1 = nn.Linear(25, 200)
        self.de_lin2 = nn.Linear(200, 4*20*20)
        self.de_conv1 = nn.ConvTranspose2d(4, 16, 5)
        self.de_conv2 = nn.ConvTranspose2d(16, 3, 7)

    def forward(self, x):

        # encoder
        h = self.relu(self.en_conv1(x))
        h = self.relu(self.en_conv2(h))
        h = self.relu(self.en_lin1(self.flatten(h)))
        h = self.en_lin2(h)

        # decoder
        y = self.relu(self.de_lin1(h))
        y = self.relu(self.de_lin2(y)).view([-1,4,20,20])
        y = self.relu(self.de_conv1(y))
        y = self.sigmoid(self.de_conv2(y))

        return y, h



        ## encoder layers
        #self.en_conv1 = nn.Conv2d(1, 32, 9)
        #self.en_conv2 = nn.Conv2d(32, 16, 7)
        #self.en_conv3 = nn.Conv2d(16, 16, 5)
        #self.en_lin1 = nn.Linear(23*23*16, 512)
        #self.en_lin2 = nn.Linear(512, 128)

        ## decoder layers
        #self.de_lin2 = nn.Linear(128, 512)
        #self.de_lin3 = nn.Linear(512, 23*23*16)
        #self.de_conv1 = nn.ConvTranspose2d(16, 16, 7, stride=2)
        #self.de_conv2 = nn.ConvTranspose2d(16, 32, 8, stride=2)
        #self.de_conv3 = nn.ConvTranspose2d(32, 1, 10, stride=2)

    #def forward(self, x):
        
    #    # encoder
    #    h = self.relu(self.en_conv1(x))
    #    h = self.pool(h)

    #    h = self.relu(self.en_conv2(h))
    #    h = self.pool(h)

    #    h = self.relu(self.en_conv3(h))
    #    h = self.pool(h)

    #    h = self.relu(self.en_lin1(self.flatten(h)))
    #    h = self.en_lin2(h)
    #    #h = self.en_lin3(h)

    #    # decoder
    #    #y = self.relu(self.de_lin1(h))
    #    y = self.relu(self.de_lin2(h))
    #    y = self.relu(self.de_lin3(y)).view([-1,16,23,23])

    #    y = self.relu(self.de_conv1(y))

    #    y = self.relu(self.de_conv2(y))

    #    y = self.sigmoid(self.de_conv3(y))

    #    return y, h