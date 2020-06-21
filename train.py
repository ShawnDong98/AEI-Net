import sys
import os
path = os.path.abspath(os.path.dirname(__file__))
sys.path.append(path)
os.environ['CUDA_VISIBLE_DEVICES'] = '0, 2, 3, 4'

import torch
import torch.nn as nn
import torch.nn.functional as F

import argparse
import time
import cv2

from face_modules.model import Backbone

from core.networks import *
from core.losses import *
from core.data_load import Data_Loader

from utils.plot import *
from utils.util import *


class trainer():
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.dataloader = Data_Loader(config.img_size, config.image_path, config.batch_size, same_prob=0.2).loader()

        self.net_init()

        self.optimizer_E = torch.optim.Adam(self.E.parameters(), 0.0004, [0, 0.999])
        self.optimizer_G = torch.optim.Adam(self.G.parameters(), 0.0004, [0, 0.999])
        self.optimizer_D = torch.optim.Adam(self.D.parameters(), 0.0004, [0, 0.999])

    def net_init(self):
        try:
            self.E = nn.DataParallel(U_Net().cuda())
            self.E.load_state_dict(torch.load("./saved_models/latest_E.pth"))
            self.G = nn.DataParallel(AADGenerator(c_id=512).cuda())
            self.G.load_state_dict(torch.load("./saved_models/latest_G.pth"))
            self.D = nn.DataParallel(MultiscaleDiscriminator(input_nc=3, n_layers=6).cuda())
            self.D.load_state_dict(torch.load("./saved_models/latest_D.pth"))
            self.state = torch.load("./saved_models/state.pth")

            print("model loaded...")
        except:
            print("No pre-trained model...")
            self.E = nn.DataParallel(U_Net().cuda())
            self.G = nn.DataParallel(AADGenerator(c_id=512).cuda())
            self.D = nn.DataParallel(MultiscaleDiscriminator(input_nc=3, n_layers=6).cuda())
            self.state = {
                "iter": 0,
                "id_loss": [],
                "att_loss": [],
                "rec_loss": [],
                "g_loss": [],
                "d_loss": []
            }

        self.arcface = Backbone(50, 0.6, 'ir_se').cuda()
        self.arcface.eval()
        self.arcface.load_state_dict(torch.load("./face_modules/model_ir_se50.pth"))
        self.arcface = nn.DataParallel(self.arcface)
        print("Arcface loaded...")

    def train(self):
        self.E.train()
        self.G.train()
        self.D.train()

        loader = iter(self.dataloader)

        for step in range(self.state["iter"], self.config.total_iter):
            try:
                Xs, Xt, same_person = next(loader)
                Xs = Xs.cuda()
                Xt = Xt.cuda()
                same_person = same_person.cuda()
            except:
                loader = iter(self.dataloader)
                Xs, Xt, same_person = next(loader)
                Xs = Xs.cuda()
                Xt = Xt.cuda()
                same_person = same_person.cuda()
            
            start_time = time.time()

            with torch.no_grad():
                z_id, _ = self.arcface(F.interpolate(Xs[:, :, 19:237, 19:237], [112, 112], mode='bilinear', align_corners=True))

            z_att = self.E(Xt)

            Y_hat = self.G(z_id, z_att)

            # rec_loss
            rec_loss = Rec_Loss(Y_hat, Xs, same_person)
            self.state["rec_loss"].append(float(rec_loss.cpu()))

            # id_loss
            id_loss = ID_Loss(Y_hat, z_id, self.arcface)
            self.state["id_loss"].append(float(id_loss.cpu()))

            # att_loss
            att_loss = Attr_Loss(Y_hat.detach(), z_att, self.E)
            self.state["att_loss"].append(float(att_loss.cpu()))

            # G_Loss
            g_loss = G_Loss(Y_hat, self.D)
            self.state["g_loss"].append(float(g_loss.cpu()))

            lossG = g_loss + 10*att_loss + 10*id_loss + 10*rec_loss

            self.optimizer_E.zero_grad()
            self.optimizer_G.zero_grad()
            lossG.backward()
            self.optimizer_E.step()
            self.optimizer_G.step()

            if step%1==0:
                # D_loss
                lossD, d_real, d_fake = D_Loss(Y_hat, Xs, self.D)
                self.state["d_loss"].append(float(lossD.cpu()))

                self.optimizer_D.zero_grad()
                lossD.backward()
                self.optimizer_D.step()

            if (step + 1) % 1 == 0:
                batch_time = time.time() - start_time
                print("time: {:.4f}s,  G_loss: {:.4f},  D_loss: {:.4f}, D_real: {:.4f}"
                      .format(batch_time, g_loss, lossD, d_real))
                print("step [{}/{}], rec_loss: {:.4f},  id_loss: {:.4f}, att_loss: {:.4f},"
                      .format(step + 1, self.config.total_iter, rec_loss,  id_loss, att_loss))
            
            if (step + 1) % 10 == 0:
                image = make_image(Xs, Xt, Y_hat)
                image.save("./output/latest.jpg")

            if (step+1) % 100 == 0:
                torch.save(self.E.state_dict(), './saved_models/latest_E.pth')
                torch.save(self.G.state_dict(), './saved_models/latest_G.pth')
                torch.save(self.D.state_dict(), './saved_models/latest_D.pth')
                self.state['iter'] = step + 1
                torch.save(self.state, './saved_models/state.pth')



    def inference(self):
        pass


def get_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--img_size', type=int, default=256)
    parser.add_argument('--total_iter', type=int, default=5000000)

    #parser.add_argument('--image_path', type=str, default='/content/datasets/shawndong98/aligncelebahqpix256')
    parser.add_argument('--image_path', type=str, default='./data/train/')
    parser.add_argument('--local_rank', default=-1, type=int,
                    help='node rank for distributed training')

    return parser.parse_args()

if __name__ == "__main__":
    #torch.distributed.init_process_group(backend='nccl')
    config = get_config()
    #torch.cuda.set_device(config.local_rank)
    trainer = trainer(config)
    trainer.train()