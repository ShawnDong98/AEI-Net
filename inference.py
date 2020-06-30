import sys
import os
path = os.path.abspath(os.path.dirname(__file__))
sys.path.append(path)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

import argparse
import PIL.Image as Image
import numpy as np
import cv2

from face_modules.model import Backbone

from core.networks import *
from utils.util import *


class inference():
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transform = transforms.Compose(
            [
                transforms.Resize((self.config.img_size, self.config.img_size)),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ]
        )

        self.net_init()

    def net_init(self):
        self.E = U_Net().cuda()
        self.E.load_state_dict(torch.load("./saved_models/latest_E_DDP.pth", map_location=torch.device('cpu')))
        self.G = AADGenerator(c_id=512).cuda()
        self.G.load_state_dict(torch.load("./saved_models/latest_G_DDP.pth", map_location=torch.device('cpu')))

        print("model loaded...")
            
        self.arcface = Backbone(50, 0.6, 'ir_se')
        self.arcface.eval()
        self.arcface.load_state_dict(torch.load("./face_modules/model_ir_se50.pth", map_location=torch.device('cpu')))
        self.arcface = self.arcface.cuda()
        print("Arcface loaded...")

    def infer(self):
        self.E.eval()
        self.G.eval()

        img1 = Image.open(self.config.img1_path)
        img2 = Image.open(self.config.img2_path)

        img1 = self.transform(img1).unsqueeze(0).cuda()
        img2 = self.transform(img2).unsqueeze(0).cuda()

        with torch.no_grad():
            z_id_1, _ = self.arcface(F.interpolate(img1[:, :, 19:237, 19:237], [112, 112], mode='bilinear', align_corners=True))
            z_id_2, _ = self.arcface(F.interpolate(img2[:, :, 19:237, 19:237], [112, 112], mode='bilinear', align_corners=True))

            z_att_1 = self.E(img1)
            z_att_2 = self.E(img2)

            Y_hat_1 = self.G(z_id_1, z_att_2)
            Y_hat_2 = self.G(z_id_2, z_att_1)

        Y_hat_1 = denorm(Y_hat_1).cpu().numpy().squeeze(0).transpose(1, 2, 0)
        Y_hat_2 = denorm(Y_hat_2).cpu().numpy().squeeze(0).transpose(1, 2, 0)

        img1 = denorm(img1).cpu().numpy().squeeze(0).transpose(1, 2, 0)
        img2 = denorm(img2).cpu().numpy().squeeze(0).transpose(1, 2, 0)

        img = np.concatenate((img1, img2), axis=1)
        Y = np.concatenate((Y_hat_2, Y_hat_1), axis=1)

        output = np.concatenate((img, Y), axis=0)
        output = output * 255
        output = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
        cv2.imwrite("./output/test.jpg", output)

        
        



def get_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--img1_path', type=str, default="./data/test/test7s.png", help="The first image's path")
    parser.add_argument('--img2_path', type=str, default="./data/test/test7t.png", help="The second image's path")

    parser.add_argument('--img_size', type=int, default=256)

    parser.add_argument('--local_rank', default=-1, type=int,
                    help='node rank for distributed training')

    return parser.parse_args()


if __name__ == "__main__":
    config = get_config()
    infer = inference(config)
    infer.infer()