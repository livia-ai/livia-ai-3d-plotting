import os
import random
import shutil
import torch
from tqdm import tqdm
import numpy as np
from livia import embedding
from dataset import TripletDataset, MixedTripletDataset
import matplotlib.pyplot as plt

def compute_image_embedding(model, device, dataloader):
    
    model = model.to(device=device)

    # create image embedding
    with torch.no_grad():
        image_embedding_list = []
        id_list = []
        #unique = set()
        for img_id, img in tqdm(dataloader):

            #if img_id[0] not in unique:
            img = img.to(device=device)
            encoded_img = model.encode(img)

            image_embedding_list.extend(encoded_img.cpu().numpy())

            #unique = unique.union(img_id)
            id_list.extend(img_id)
        print(len(image_embedding_list))
        print(len(id_list))
    image_embedding = embedding.Embedding(np.array(image_embedding_list), np.array(id_list, dtype=object))

    return image_embedding

def display_triplets(triplets, root_dir, evaluation_dir):

    useable_triplets = []
    for ori, sim, dis in triplets:
        split_ori = ori.split("/")
        split_sim = sim.split("/")
        split_dis = dis.split("/")
        useable_triplets.append(("__@@__".join(split_ori), "__@@__".join(split_sim), "__@@__".join(split_dis)))

    dataset = TripletDataset(triplets = useable_triplets,
                            root_dir = root_dir)

    i = 0
    for triplet in dataset: 

        print(triplet[3])

        fig,ax = plt.subplots(1,3)
        ax[0].imshow(triplet[0])
        ax[1].imshow(triplet[1])
        ax[2].imshow(triplet[2])

        ax[0].set_title("Sample")
        ax[1].set_title("Similar")
        ax[2].set_title("Dissimilar")

        plt.savefig(evaluation_dir + f"/triplet_{i}.png")

        plt.show()

        i+= 1

def display_triplets_mixed(triplets, root_dir, evaluation_dir):

    useable_triplets = []
    useable_mus_ids = []
    for ori, sim, dis in triplets:

        ori_id, ori_musuem_id = ori.split("@/@")
        sim_id, sim_musuem_id = sim.split("@/@")
        dis_id, dis_musuem_id = dis.split("@/@")

        useable_triplets.append((ori_id, sim_id, dis_id))
        useable_mus_ids.append((ori_musuem_id, sim_musuem_id, dis_musuem_id))


    dataset = MixedTripletDataset(triplets = useable_triplets,
                                root_dir = root_dir,
                                museum_ids = useable_mus_ids)

    i = 0
    for triplet_i in dataset: 
        print(triplet_i[3])

        fig,ax = plt.subplots(1,3)
        ax[0].imshow(triplet_i[0])
        ax[1].imshow(triplet_i[1])
        ax[2].imshow(triplet_i[2])

        ax[0].set_title(f"Sample \n ImgPath: {triplet_i[3][0]} \n Museum:{triplet_i[4][0]}")
        ax[1].set_title(f"Similar\n ImgPath: {triplet_i[3][1]} \n Museum:{triplet_i[4][1]}")
        ax[2].set_title(f"Dissimilar\n ImgPath: {triplet_i[3][2]} \n Museum:{triplet_i[4][2]}")
        
        plt.savefig(evaluation_dir + f"/triplet_{i}.png")

        plt.show()

        i += 1

def sample_images(src, folder_path, k):
    sample = random.sample(os.listdir(src), k)

    if os.path.isdir(folder_path):
        shutil.rmtree(folder_path)
        os.mkdir(folder_path)
    else:
        os.mkdir(folder_path)
    for file in sample:
        file_path = os.path.join(src, file)
        shutil.copy(file_path, folder_path)