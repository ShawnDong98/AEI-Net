import torch
import torch.nn as nn
import torch.nn.functional as F 


def hinge_loss(input, positive):
    if positive:
        output = nn.ReLU()(1.0 + input).mean()
    else:
        output = nn.ReLU()(1.0 - input).mean()

    return output

def Rec_Loss(Y_hat, Xs, same_person):

    rec_loss = torch.sum(torch.mean(((Y_hat - Xs) ** 2).view(Y_hat.size(0), -1), dim=1) * same_person) / (same_person.sum() + 1e-6)
    
    return rec_loss

def ID_Loss(Y_hat, Xs_id, arcface):

    Y_aligned = Y_hat[:, :, 19:237, 19:237]
    Y_id, _ = arcface(F.interpolate(Y_aligned, [112, 112], mode='bilinear', align_corners=True))
    id_loss = (1 - torch.cosine_similarity(Xs_id, Y_id, dim=1)).mean()
    
    return id_loss

def Attr_Loss(Y_hat, Xt_attr, Encoder):
    
    Y_attr = Encoder(Y_hat)
    attr_loss = 0
    for i in range(len(Xt_attr)):
        attr_loss += torch.mean(((Y_attr[i] - Xt_attr[i]) ** 2).view(Y_attr[i].size(0), -1), dim=1).mean()

    return attr_loss

def G_Loss(Y_hat, D):
    Di = D(Y_hat)
    g_loss = 0
    for di in Di:
        g_loss += -di[0].mean()

    return g_loss

def D_Loss(Y_hat, Xs, D):
    fake_D = D(Y_hat.detach())
    loss_fake = 0
    for di in fake_D:
        loss_fake += hinge_loss(di[0], False)

    true_D = D(Xs)
    loss_true = 0
    for di in true_D:
        loss_true += hinge_loss(di[0], True)

    lossD = loss_true.mean() + loss_fake.mean()

    return lossD, loss_true.mean(), loss_fake.mean()