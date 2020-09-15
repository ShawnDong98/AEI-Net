import torch
import torch.nn as nn
import torch.nn.functional as F


def hinge_loss(input, positive):
    if positive:
        output = nn.ReLU()(1.0 + input).mean()
    else:
        output = nn.ReLU()(1.0 - input).mean()

    return output