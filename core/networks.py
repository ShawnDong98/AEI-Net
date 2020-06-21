import sys
import os
path = os.path.abspath(os.path.dirname(__file__))
sys.path.append(path)

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.spectral_norm as SN

import numpy as np

from modules import *


class U_Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = Conv4x4(3, 32) # 256->128
        self.conv2 = Conv4x4(32, 64) # 128->64
        self.conv3 = Conv4x4(64, 128) # 64->32
        self.conv4 = Conv4x4(128, 256) # 32->16
        self.conv5 = Conv4x4(256, 512) # 16->8
        self.conv6 = Conv4x4(512, 1024) # 8->4
        self.conv7 = Conv4x4(1024, 1024) # 4->2

        self.deconv1 = DeConv4x4(1024, 1024) # 2->4
        self.deconv2 = DeConv4x4(2048, 512) # 4->8
        self.deconv3 = DeConv4x4(1024, 256) # 8->16
        self.deconv4 = DeConv4x4(512, 128) # 16->32
        self.deconv5 = DeConv4x4(256, 64) # 32->64
        self.deconv6 = DeConv4x4(128, 32) # 64->128
    
    def forward(self, Xt):
        feat1 = self.conv1(Xt) # 32 x 128 x 128
        feat2 = self.conv2(feat1) # 64 x 64 x 64
        feat3 = self.conv3(feat2) # 128 x 32 x 32
        feat4 = self.conv4(feat3) # 256 x 16 x 16
        feat5 = self.conv5(feat4) # 512 x 8 x 8
        feat6 = self.conv6(feat5) # 1024 x 4 x 4
        z_att1 = self.conv7(feat6) # 1024 x 2 x 2

        z_att2 = self.deconv1(z_att1, feat6)
        z_att3 = self.deconv2(z_att2, feat5)
        z_att4 = self.deconv3(z_att3, feat4)
        z_att5 = self.deconv4(z_att4, feat3)
        z_att6 = self.deconv5(z_att5, feat2)
        z_att7 = self.deconv6(z_att6, feat1)

        z_att8 = F.interpolate(z_att7, scale_factor=2, mode="bilinear", align_corners=True)
    
        return z_att1, z_att2, z_att3, z_att4, z_att5, z_att6, z_att7, z_att8



class AADGenerator(nn.Module):
    def __init__(self, c_id):
        super().__init__()
        self.upsample = nn.ConvTranspose2d(c_id, 1024, kernel_size=2, stride=1, padding=0)
        self.AADBlk1 = AADResBlk(1024, 1024, c_id, 1024)
        self.AADBlk2 = AADResBlk(1024, 1024, c_id, 2048)
        self.AADBlk3 = AADResBlk(1024, 1024, c_id, 1024)
        self.AADBlk4 = AADResBlk(1024, 512, c_id, 512)
        self.AADBlk5 = AADResBlk(512, 256, c_id, 256)
        self.AADBlk6 = AADResBlk(256, 128, c_id, 128)
        self.AADBlk7 = AADResBlk(128, 64, c_id, 64)
        self.AADBlk8 = AADResBlk(64, 3, c_id, 64)

    def forward(self, z_id, z_att):
        out = self.upsample(z_id.view(z_id.size(0), -1, 1, 1))
        out = F.interpolate(self.AADBlk1(out, z_id, z_att[0]), scale_factor=2, mode='bilinear', align_corners=True)
        out = F.interpolate(self.AADBlk2(out, z_id, z_att[1]), scale_factor=2, mode='bilinear', align_corners=True)    
        out = F.interpolate(self.AADBlk3(out, z_id, z_att[2]), scale_factor=2, mode='bilinear', align_corners=True)
        out = F.interpolate(self.AADBlk4(out, z_id, z_att[3]), scale_factor=2, mode='bilinear', align_corners=True)
        out = F.interpolate(self.AADBlk5(out, z_id, z_att[4]), scale_factor=2, mode='bilinear', align_corners=True)
        out = F.interpolate(self.AADBlk6(out, z_id, z_att[5]), scale_factor=2, mode='bilinear', align_corners=True)
        out = F.interpolate(self.AADBlk7(out, z_id, z_att[6]), scale_factor=2, mode='bilinear', align_corners=True)

        y = self.AADBlk8(out, z_id, z_att[7])

        return torch.tan(y)




class NLayerDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, use_sigmoid=False, getIntermFeat=False):
        super(NLayerDiscriminator, self).__init__()
        self.getIntermFeat = getIntermFeat
        self.n_layers = n_layers

        kw = 4
        padw = int(np.ceil((kw-1.0)/2))
        sequence = [[nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]]

        nf = ndf
        for n in range(1, n_layers):
            nf_prev = nf
            nf = min(nf * 2, 512)
            sequence += [[
                SN(nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=2, padding=padw)),
                nn.LeakyReLU(0.2, True)
            ]]

        nf_prev = nf
        nf = min(nf * 2, 512)
        sequence += [[
            SN(nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=1, padding=padw)),
            nn.LeakyReLU(0.2, True)
        ]]

        sequence += [[nn.Conv2d(nf, 1, kernel_size=kw, stride=1, padding=padw)]]

        if use_sigmoid:
            sequence += [[nn.Sigmoid()]]

        if getIntermFeat:
            for n in range(len(sequence)):
                setattr(self, 'model'+str(n), nn.Sequential(*sequence[n]))
        else:
            sequence_stream = []
            for n in range(len(sequence)):
                sequence_stream += sequence[n]
            self.model = nn.Sequential(*sequence_stream)

    def forward(self, input):
        if self.getIntermFeat:
            res = [input]
            for n in range(self.n_layers+2):
                model = getattr(self, 'model'+str(n))
                res.append(model(res[-1]))
            return res[1:]
        else:
            return self.model(input)


class MultiscaleDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, use_sigmoid=False, num_D=3, getIntermFeat=False):
        super(MultiscaleDiscriminator, self).__init__()
        self.num_D = num_D
        self.n_layers = n_layers
        self.getIntermFeat = getIntermFeat

        for i in range(num_D):
            netD = NLayerDiscriminator(input_nc, ndf, n_layers, use_sigmoid, getIntermFeat)
            if getIntermFeat:
                for j in range(n_layers + 2):
                    setattr(self, 'scale' + str(i) + '_layer' + str(j), getattr(netD, 'model' + str(j)))
            else:
                setattr(self, 'layer' + str(i), netD.model)

        self.downsample = nn.AvgPool2d(3, stride=2, padding=[1, 1], count_include_pad=False)

    def singleD_forward(self, model, input):
        if self.getIntermFeat:
            result = [input]
            for i in range(len(model)):
                result.append(model[i](result[-1]))
            return result[1:]
        else:
            return [model(input)]

    def forward(self, input):
        num_D = self.num_D
        result = []
        input_downsampled = input
        for i in range(num_D):
            if self.getIntermFeat:
                model = [getattr(self, 'scale' + str(num_D - 1 - i) + '_layer' + str(j)) for j in
                         range(self.n_layers + 2)]
            else:
                model = getattr(self, 'layer' + str(num_D - 1 - i))
            result.append(self.singleD_forward(model, input_downsampled))
            if i != (num_D - 1):
                input_downsampled = self.downsample(input_downsampled)
        return result