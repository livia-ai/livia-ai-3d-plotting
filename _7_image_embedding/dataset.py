
from torch.utils.data import Dataset
import livia.embedding as embedding
import livia.triplet as triplet
import os
from PIL import Image


class TripletDataset(Dataset):

    def __init__(self,
                 root_dir:str,
                 triplets = None,
                 samples = None, 
                 transform = None):
        
        self.root_dir = root_dir
        self.transform = transform

        if triplets is None and samples is None:
            raise ValueError("Either triplets or samples must be not None.")

        else:

            if samples is None:
                self.triplets = triplets
                self.samples = self.triplets_to_image_paths()

            else:
                self.samples = samples
        
    def triplets_to_image_paths(self):
        instances = []
        root, dirs, files = sorted(os.walk(self.root_dir, followlinks=True))[0]

        for ori, sim, dis in self.triplets:
            
            ori_path = ori + ".224.jpg"
            sim_path = sim + ".224.jpg"
            dis_path = dis + ".224.jpg"
            
            if ori_path in files:
                if sim_path in files:
                    if dis_path in files:
                        instances.append((ori_path, sim_path, dis_path))

        return instances

    def __len__(self) -> int:
        return len(self.samples)

    def loader(self, path):
        with open(path, "rb") as f:
            img = Image.open(f)
            return img.convert("RGB")

    def __getitem__(self, idx):
        ori_path, sim_path, dis_path = self.samples[idx]

        ori_img = self.loader(os.path.join(self.root_dir, ori_path))
        sim_img  = self.loader(os.path.join(self.root_dir, sim_path))
        dis_img  = self.loader(os.path.join(self.root_dir, dis_path))

        if self.transform is not None:
            ori_img = self.transform(ori_img)
            sim_img  = self.transform(sim_img)
            dis_img  = self.transform(dis_img)

        return ori_img, sim_img, dis_img, (ori_path, sim_path, dis_path)

class MixedTripletDataset(Dataset):

    def __init__(self,
                 root_dir:str,
                 museum_ids:list,
                 triplets = None,
                 samples = None,
                 transform = None):
        
        self.root_dir = root_dir
        self.transform = transform
        self.museum_ids = museum_ids

        if triplets is None and samples is None:
            raise ValueError("Either triplets or samples must be not None.")

        else:

            if samples is None:
                self.triplets = triplets
                self.samples = self.triplets_to_image_paths()

            else:
                self.samples = samples

    def triplets_to_image_paths(self):
        instances = []
        root, dataset_1, dataset_2 = os.walk(self.root_dir, followlinks=True)

        files = dataset_1[2] + dataset_2[2] + root[2]
        for i in range(len(self.triplets)):

            ori, sim, dis = self.triplets[i]
            ori_path = ori + ".224.jpg"
            sim_path = sim + ".224.jpg"
            dis_path = dis + ".224.jpg"

            if ori_path in files:
                if sim_path in files:
                    if dis_path in files:
                        instances.append((ori_path, sim_path, dis_path))

        return instances

    def __len__(self) -> int:
        return len(self.samples)

    def loader(self, path):
        with open(path, "rb") as f:
            img = Image.open(f)
            return img.convert("RGB")

    def __getitem__(self, idx):
        ori_path, sim_path, dis_path = self.samples[idx]
        ori_mus_id, sim_mus_id, dis_mus_id = self.museum_ids[idx]

        ori_img = self.loader(os.path.join(self.root_dir, ori_mus_id, ori_path))
        sim_img  = self.loader(os.path.join(self.root_dir, sim_mus_id, sim_path))
        dis_img  = self.loader(os.path.join(self.root_dir, dis_mus_id, dis_path))

        if self.transform is not None:
            ori_img = self.transform(ori_img)
            sim_img  = self.transform(sim_img)
            dis_img  = self.transform(dis_img)

        return ori_img, sim_img, dis_img, (ori_path, sim_path, dis_path), (ori_mus_id, sim_mus_id, dis_mus_id)

class ImageDataset(Dataset):
    
    def __init__(self, root_dir:str, transform=None):
        self.root_dir = root_dir
        self.transform = transform

        self.samples = self.get_samples()

    def get_samples(self):
        
        root, dirs, files = sorted(os.walk(self.root_dir, followlinks=True))[0]

        return files     
    
    def __len__(self):
        return len(self.samples)
    
    def loader(self, path):
        with open(path, "rb") as f:
            img = Image.open(f)
            return img.convert("RGB")
        
    def __getitem__(self, idx):
        
        img_path = self.samples[idx]

        img_id = img_path.split(".")[0]

        img = self.loader(os.path.join(self.root_dir, img_path))

        if self.transform is not None:
            img = self.transform(img)

        return img_id, img

class MixedImageDataset(Dataset):
    
    def __init__(self, root_dir:str, transform=None):
        self.root_dir = root_dir
        self.transform = transform

        self.samples = self.get_samples()

    def get_samples(self):
        
        root, bel, wm = os.walk(self.root_dir, topdown=True)

        bel_id = root[1][0]
        wm_id = root[1][1]

        bel_files = list(zip([bel_id]*len(bel[2]), bel[2]))
        wm_files = list(zip([wm_id]*len(wm[2]), wm[2]))

        return bel_files + wm_files     
    
    def __len__(self):
        return len(self.samples)
    
    def loader(self, path):
        with open(path, "rb") as f:
            img = Image.open(f)
            return img.convert("RGB")
        
    def __getitem__(self, idx):
        
        museum_id, img_path = self.samples[idx]

        img_id = img_path.split(".")[0]

        img = self.loader(os.path.join(self.root_dir, museum_id, img_path))

        if self.transform is not None:
            img = self.transform(img)

        return (museum_id, img_id, img)