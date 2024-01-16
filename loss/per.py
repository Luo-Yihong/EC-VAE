from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import torch
import torch.nn as nn
import torchvision.models as models


class netVGGFeatures(nn.Module):
    def __init__(self,device):
        super(netVGGFeatures, self).__init__()
        self.vggnet = models.vgg16(pretrained=True).to(device)
        self.layer_ids = [2, 7, 12, 21, 30]

    def main(self, z, levels):
        layer_ids = self.layer_ids[:levels]
        id_max = layer_ids[-1] + 1
        output = []
        for i in range(id_max):
            z = self.vggnet.features[i](z)
            if i in layer_ids:
                output.append(z)
        return output

    def forward(self, z, levels):
        output = self.main(z, levels)
        return output


class VGGDistance(nn.Module):
    def __init__(self, levels, device):
        super(VGGDistance, self).__init__()
        self.vgg = netVGGFeatures(device)
        self.levels = levels
        self.factors = [0] * (self.levels + 1)
        self.pool = nn.AvgPool2d(8, 8)
        
    def forward(self, I1, I2, use_factors=False):
        eps = 1e-8
        sum_factors = sum(self.factors)
        f1 = self.vgg(I1, self.levels)
        f2 = self.vgg(I2, self.levels)
        loss = torch.abs(I1 - I2).sum()
        for i in range(self.levels):
            layer_loss = torch.abs(f1[i] - f2[i]).sum()
            self.factors[i] += layer_loss.item()
            if use_factors:
                layer_loss = sum_factors / (self.factors[i] + eps) * layer_loss
            
            loss = loss + layer_loss
            
        return loss#*8e-3
    
def distance_metric(args, force_l2=False):
    return VGGDistance(4, args.device)
    if args.dataset in ['celeba','bedroom','church','cifar']:
        return VGGDistance(4, args.device)
    return VGGDistance(3, args.device)