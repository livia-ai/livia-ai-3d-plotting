# Imports
import torch
from torchvision import transforms
import livia.embedding as embedding

from dataset import ImageDataset
import utility_functions  as uf

# check if GPU is available
if torch.cuda.is_available(): 
    device = "cuda" 
else: 
    device = "cpu"

# specify path to images
path_images = "inference/test_images"
# specify path to model
path_model = "inference/triplet_net.pt"
# specify path of output csv file containing the image embeddings
path_image_embeddings = "inference/test_embeddings" # without .csv at the end 

##############################################################
# Code to load images, the model and compute the embeddings
##############################################################
# create dataset
size=224
transform = transforms.Compose([transforms.CenterCrop(size), transforms.Grayscale(num_output_channels=1), transforms.ToTensor()])
dataset = ImageDataset(root_dir=path_images, transform=transform)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
# load model
model = torch.load(path_model, map_location=device)
# compute embedding
image_embedding = uf.compute_image_embedding(model, device, dataloader)
# save image embedding as csv file
embedding.save_to_csv(image_embedding, path_image_embeddings)