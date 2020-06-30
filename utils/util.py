import torch
from torchvision import transforms
from torchvision.utils import save_image, make_grid

import pandas as pd
import numpy as np 
import seaborn as sns
import os
import matplotlib.pyplot as plt

# loader使用torchvision中自带的transforms函数
loader = transforms.Compose([
    transforms.ToTensor()])

unloader = transforms.ToPILImage()


def imshow(tensor, title=None):
    image = denorm(tensor)
    image = image.cpu().clone()  # we clone the tensor to not do changes on it
    image = image.squeeze(0)  # remove the fake batch dimension
    image = unloader(image)
    image.save("./output/test.jpg")



def make_folder(path):
    if not os.path.exists(os.path.join(path)):
        os.makedirs(os.path.join(path))


def denorm(x):
    out = (x + 1)/ 2
    return out.clamp_(0, 1)

def make_image(Xs, Xt, Y):
    Xs = make_grid(denorm(Xs).cpu(), nrow=Xs.shape[0])
    Xt = make_grid(denorm(Xt).cpu(), nrow=Xt.shape[0])
    Y = make_grid(denorm(Y).cpu(), nrow=Y.shape[0])
    return unloader(torch.cat((Xs, Xt, Y), dim=1))





if __name__ == "__main__":
    pass