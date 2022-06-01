# imports
import numpy as np
from torchvision import datasets, transforms
from PIL import Image
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

data_dir = 'data/test_images'

transform = transforms.Compose([transforms.ToTensor()])

train_dataset = datasets.ImageFolder(data_dir + "/train", transform=transform)
test_dataset = datasets.ImageFolder(data_dir+ "/test", transform=transform)


train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=4, shuffle=True)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)

# Creating a DeepAutoencoder class
class ConvAutoencoder(torch.nn.Module):
    def __init__(self):
        super().__init__()        
        self.encoder = torch.nn.Sequential(
            nn.Conv2d(3, 16, 32),  #193x193
            nn.ReLU(),
            nn.Conv2d(16, 4, 16),  #178x178
            nn.ReLU(),
            nn.MaxPool2d(2, 2), #89x89

            nn.Conv2d(4, 2, 8),  #82x82 
            nn.ReLU(),
            nn.Conv2d(2, 1, 5),  #78x78
            nn.ReLU(),
            nn.MaxPool2d(2, 2), #39x39

            nn.Flatten(start_dim=1),
            nn.Linear(39*39, 300),
            nn.ReLU(),
            nn.Linear(300, 10)
        )
          
        self.linear_decoder = torch.nn.Sequential(
            nn.Linear(10, 300),
            nn.ReLU(),
            nn.Linear(300, 39*39),
            nn.ReLU(),
        )
        self.conv_decoder = torch.nn.Sequential(
            nn.ConvTranspose2d(1, 1, 2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(1, 2, 5),
            nn.ReLU(),
            nn.ConvTranspose2d(2, 4, 8),
            nn.ReLU(),

            nn.ConvTranspose2d(4, 4, 2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(4, 16, 16),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 3, 32),
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded_1 = self.linear_decoder(encoded).view(-1,1,39,39)
        decoded_2 = self.conv_decoder(decoded_1)
        return decoded_2
  
# Instantiating the model and hyperparameters
model = ConvAutoencoder().to(device="cuda")
criterion = torch.nn.MSELoss()
num_epochs = 40
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

train_loss = list()
for i in range(num_epochs):
    if i%10 == 0:
        print(f"Epoch:{i}")
    epoch_loss = list()
    for image, museum in train_dataloader:
        image = image.to(device="cuda")

        output = model(image)
        loss = criterion(output, image)
        loss.backward()
        optimizer.step()

        optimizer.zero_grad()
        epoch_loss.append(loss.cpu().detach().numpy())

    if i%10 == 0:
        print(np.mean(epoch_loss))
    train_loss.append(np.mean(epoch_loss))

#print(train_loss)
with torch.no_grad():
    model = model.to("cpu")
    for image, museum in train_dataloader:
        output = model(image)
        fig, ax = plt.subplots(1,2, figsize=(6,20))
        ax[0].imshow(image[0].permute(1,2,0))
        ax[1].imshow(output[0].permute(1,2,0))
        plt.show()
        break