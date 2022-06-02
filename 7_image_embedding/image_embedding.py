# imports
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm


from torchvision import datasets, transforms
import torch
import torch.nn as nn

from model import ConvAutoencoder
from torch.utils.tensorboard import SummaryWriter

torch.cuda.empty_cache() 

data_dir = 'data/test_images'

size=30
transform = transforms.Compose([transforms.ToTensor(), transforms.Grayscale(3), transforms.CenterCrop(size)])

train_dataset = datasets.ImageFolder(data_dir + "/train", transform=transform)
test_dataset = datasets.ImageFolder(data_dir+ "/test", transform=transform)

n_images = 30
rng = np.random.default_rng()
permuted_ids = rng.permutation(len(train_dataset))[:n_images]
train_dataset_subset = torch.utils.data.Subset(train_dataset, permuted_ids)

batch_size = 8
train_dataloader = torch.utils.data.DataLoader(train_dataset_subset, batch_size=batch_size, shuffle=True)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


# Instantiating the model and hyperparameters
model = ConvAutoencoder().to(device="cuda")

#for image, museum in train_dataloader:
#    print("in:", image.shape)
#    image = image.to(device="cuda")
#    output, h = model(image)
#    print("h:", h.shape)
#    print("out:", output.shape)
#error

#criterion = torch.nn.L1Loss()
criterion = torch.nn.MSELoss()
num_epochs = 1000
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

train_loss = list()
print_at = 2000

writer = SummaryWriter(f'experiments/{n_images}_images_grayscale_{size}_centercropped')

progress_bar = tqdm(range(num_epochs))
for i in progress_bar:

    #if i%print_at == 0:
    #    print(f"Epoch:{i}")

    epoch_loss = list()
    for image, museum in train_dataloader:
        image = image.to(device="cuda")

        output,h = model(image)
        loss = criterion(output, image)
        loss.backward()
        optimizer.step()

        optimizer.zero_grad()
        epoch_loss.append(loss.cpu().detach().numpy())




    loss_str = str(np.around(np.mean(epoch_loss),5))
    progress_bar.set_postfix_str(loss_str)

    # tensorboard
    writer.add_scalar('training/mean epoch loss',
                    np.around(np.mean(epoch_loss),5),
                    i)
    #if i%print_at == 0:
    #    print(np.mean(epoch_loss))

    train_loss.append(np.mean(epoch_loss))

#print(train_loss)
with torch.no_grad():
    model = model.to("cpu")
    for image, museum in train_dataloader:
        output, h = model(image)

        combined = torch.stack([image[0],output[0]])
        writer.add_images('Target vs Reconstruction',combined)

        #print(f"Max: image:{torch.max(image)}, output:{torch.max(output)}")
        #print(f"Min: image:{torch.min(image)}, output:{torch.min(output)}")

        #fig, ax = plt.subplots(1,2, figsize=(6,20))
        #ax[0].imshow(image[0].permute(1,2,0))
        #ax[1].imshow(output[0].permute(1,2,0))
        ##plt.show()

        #writer.add_figure('Target vs Reconstruction figure',fig)
        break
        
writer.close()