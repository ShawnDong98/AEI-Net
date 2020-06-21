import sys
import os
path = os.path.abspath(os.path.dirname(__file__))
sys.path.append(path)

import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data.distributed import DistributedSampler

from PIL import Image
import glob
import random
from utils.util import *


class DataSet():
    def __init__(self, root, trans_Xs, same_prob=0.2):
        self.imgs = glob.glob(root+"/*")
        self.trans_Xs = trans_Xs
        self.same_prob = same_prob
        

    def __getitem__(self, index):
        path = self.imgs[index]
        Xs = Image.open(path)
        if random.random() > self.same_prob:
            Xt = Image.open(random.choice(self.imgs))
            same_person = 0  
        else:
            Xt = Xs
            same_person = 1
        return self.trans_Xs(Xs), self.trans_Xs(Xt),  same_person

    
    def __len__(self):
        return len(self.imgs)


class Data_Loader():
    def __init__(self, img_size, img_path, batch_size, same_prob=0.2):
        self.img_size = img_size
        self.img_path = img_path
        self.batch_size = batch_size
        self.trans_Xs = transforms.Compose(
            [
                transforms.Resize((self.img_size, self.img_size)),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ]
        )

        self.dataset = DataSet(img_path, self.trans_Xs)

    def loader(self):
        #sampler = DistributedSampler(self.dataset)
        loader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            drop_last=True,
            shuffle=True
            #sampler=sampler
        )
        
        return loader







if __name__ == "__main__":
    
    path = "./data/train/"
    img_size = 256
    loader = iter(Data_Loader(img_size, path, 8).loader())

    Xs, Xt, same_person = next(loader)

    print(Xs.shape)
    m = Xs
    imshow(m[0])




