
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

        return ori_img, sim_img, dis_img, ori_path


class ImageDataset(Dataset):
    
    def __init__(self, sample_ids, root_dir:str, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = self.samples_to_image_paths(sample_ids)
        
    def samples_to_image_paths(self, sample_ids):
        
        root, dirs, files = sorted(os.walk(self.root_dir, followlinks=True))[0]
        
        paths = []
        for idx in sample_ids:
            
            image_path = idx + ".224.jpg" 
            
            if image_path in files:
                paths.append((idx,image_path))
            else:
                paths.append((idx, None))
                
        return paths     
    
    def __len__(self):
        return len(self.image_paths)
    
    def loader(self, path):
        with open(path, "rb") as f:
            img = Image.open(f)
            return img.convert("RGB")
        
    def __getitem__(self, idx):
        
        sample_id, img_path = self.image_paths[idx]
        img = self.loader(os.path.join(self.root_dir, img_path))

        if self.transform is not None:
            img = self.transform(img)

        return img