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
from core.data_load import *
from core.losses import *

from utils.plot import *
from utils.util import *


class trainer_DDP():
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.dataloader = Data_Loader_DDP(config.img_size, config.image_path, config.batch_size, same_prob=0.2).loader()
        self.transform = transforms.Compose(
            [
                transforms.Resize((self.config.img_size, self.config.img_size)),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ]
        )

        self.net_init()

        self.optimizer_E = torch.optim.Adam(self.E.parameters(), 0.0004, [0, 0.999])
        self.optimizer_G = torch.optim.Adam(self.G.parameters(), 0.0004, [0, 0.999])
        self.optimizer_D = torch.optim.Adam(self.D.parameters(), 0.0004, [0, 0.999])

    def net_init(self):
        try:
            self.E = nn.parallel.DistributedDataParallel(U_Net().cuda(self.config.local_rank), device_ids=[self.config.local_rank], broadcast_buffers=False)
            self.E.module.load_state_dict(torch.load("./saved_models/latest_E_DDP.pth", map_location=torch.device('cpu')))
            self.G = nn.parallel.DistributedDataParallel(AADGenerator(c_id=512).cuda(self.config.local_rank), device_ids=[self.config.local_rank])
            self.G.module.load_state_dict(torch.load("./saved_models/latest_G_DDP.pth", map_location=torch.device('cpu')))
            self.D = nn.parallel.DistributedDataParallel(MultiscaleDiscriminator(input_nc=3, n_layers=6).cuda(self.config.local_rank), device_ids=[self.config.local_rank])
            self.D.module.load_state_dict(torch.load("./saved_models/latest_D_DDP.pth", map_location=torch.device('cpu')))
            self.state = torch.load("./saved_models/state_DDP.pth",  map_location=torch.device('cpu'))

            print("model loaded...")
            
        except:
            print("No pre-trained model...")
            self.E = nn.parallel.DistributedDataParallel(U_Net().cuda(self.config.local_rank), device_ids=[self.config.local_rank], broadcast_buffers=False)
            self.G = nn.parallel.DistributedDataParallel(AADGenerator(c_id=512).cuda(self.config.local_rank), device_ids=[self.config.local_rank])
            self.D = nn.parallel.DistributedDataParallel(MultiscaleDiscriminator(input_nc=3, n_layers=6).cuda(self.config.local_rank), device_ids=[self.config.local_rank])
            self.state = {
                "iter": 0,
                "id_loss": [],
                "att_loss": [],
                "rec_loss": [],
                "g_loss": [],
                "d_loss": []
            }

        self.arcface = Backbone(50, 0.6, 'ir_se')
        self.arcface.eval()
        self.arcface.load_state_dict(torch.load("./face_modules/model_ir_se50.pth", map_location=torch.device('cpu')))
        self.arcface = nn.parallel.DistributedDataParallel(self.arcface.cuda(self.config.local_rank), device_ids=[self.config.local_rank])
        print("Arcface loaded...")

    def train(self):
        self.E.train()
        self.G.train()
        self.D.train()

        loader = iter(self.dataloader)

        for step in range(self.state["iter"], self.config.total_iter):
            try:
                Xs, Xt, same_person = next(loader)
                Xs = Xs.cuda(self.config.local_rank, non_blocking=True)
                Xt = Xt.cuda(self.config.local_rank, non_blocking=True)
                same_person = same_person.cuda(self.config.local_rank, non_blocking=True)
            except:
                loader = iter(self.dataloader)
                Xs, Xt, same_person = next(loader)
                Xs = Xs.cuda(self.config.local_rank, non_blocking=True)
                Xt = Xt.cuda(self.config.local_rank, non_blocking=True)
                same_person = same_person.cuda(self.config.local_rank, non_blocking=True)
            
            start_time = time.time()

            with torch.no_grad():
                z_id, _ = self.arcface(F.interpolate(Xs[:, :, 19:237, 19:237], [112, 112], mode='bilinear', align_corners=True))

            z_att = self.E(Xt)

            Y_hat = self.G(z_id, z_att)

            # rec_loss
            rec_loss = torch.sum(torch.mean(((Y_hat - Xs) ** 2).view(Y_hat.size(0), -1), dim=1) * same_person) / (same_person.sum() + 1e-6)
            self.state["rec_loss"].append(float(rec_loss.cpu()))

            # id_loss
            Y_id,_ = self.arcface(F.interpolate(Y_hat[:, :, 19:237, 19:237], [112, 112], mode='bilinear', align_corners=True))
            id_loss = (1 - torch.cosine_similarity(z_id, Y_id, dim=1)).mean()
            self.state["id_loss"].append(float(id_loss.cpu()))

            # att_loss
            Y_attr = self.E(Y_hat.detach())
            att_loss = 0
            for i in range(len(z_att)):
                att_loss = att_loss + torch.mean(((Y_attr[i] - z_att[i]) ** 2).view(Y_attr[i].size(0), -1), dim=1).mean()
            self.state["att_loss"].append(float(att_loss.cpu()))

            # G_Loss
            Di = self.D(Y_hat)
            g_loss = 0
            for di in Di:
                g_loss += hinge_loss(di[0], True)
            self.state["g_loss"].append(float(g_loss.cpu()))

            lossG = g_loss + 10*att_loss + 5*id_loss + 10*rec_loss
            # lossG = 10*rec_loss + 10*att_loss + 5*id_loss

            self.optimizer_E.zero_grad()
            self.optimizer_G.zero_grad()
            lossG.backward()
            self.optimizer_E.step()
            self.optimizer_G.step()

            # D_loss
            fake_D = self.D(Y_hat.detach())
            loss_fake = 0
            for di in fake_D:
                loss_fake = loss_fake + hinge_loss(di[0], False)

            true_D = self.D(Xs)
            loss_true = 0
            for di in true_D:
                loss_true = loss_true + hinge_loss(di[0], True)

            lossD = loss_true.mean() + loss_fake.mean()

            self.state["d_loss"].append(float(lossD.cpu()))

            self.optimizer_D.zero_grad()
            lossD.backward()
            self.optimizer_D.step()

            if (step + 1) % 1 == 0 :
                batch_time = time.time() - start_time
                print("time: {:.4f}s,  G_loss: {:.4f},  D_loss: {:.4f}, D_real: {:.4f}"
                      .format(batch_time, g_loss, lossD, loss_true))
                print("step [{}/{}], rec_loss: {:.4f},  id_loss: {:.4f}, att_loss: {:.4f},"
                      .format(step + 1, self.config.total_iter, rec_loss,  id_loss, att_loss))
            
            if (step + 1) % 10 == 0 and self.config.local_rank==0:
                image = make_image(Xs, Xt, Y_hat)
                image.save("./output/latest_DDP.jpg")

            if (step+1) % 100 == 0 and self.config.local_rank==0:
                torch.save(self.E.module.state_dict(), './saved_models/latest_E_DDP.pth')
                torch.save(self.G.module.state_dict(), './saved_models/latest_G_DDP.pth')
                torch.save(self.D.module.state_dict(), './saved_models/latest_D_DDP.pth')
                self.state['iter'] = step + 1
                torch.save(self.state, './saved_models/state_DDP.pth')
                draw_losses_DDP()







def get_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=5)
    parser.add_argument('--img_size', type=int, default=256)
    parser.add_argument('--total_iter', type=int, default=5000000)

    parser.add_argument('--image_path', type=str, default='./data/train/')
    parser.add_argument('--DDP', action='store_true', )
    parser.add_argument('--local_rank', default=-1, type=int,
                    help='node rank for distributed training')

    return parser.parse_args()

if __name__ == "__main__":
    # DDP
    torch.distributed.init_process_group(backend='nccl')
    config = get_config()
    torch.cuda.set_device(config.local_rank)
    torch.autograd.set_detect_anomaly(True)
    trainer = trainer_DDP(config)
    trainer.inference()
