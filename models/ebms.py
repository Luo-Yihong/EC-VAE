"""largely adopted from https://github.com/hankook/CLEL/blob/main/architectures.py"""
from typing import Any, Mapping, Optional, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torch.nn.utils.spectral_norm as spectral_norm


# from models.res import BasicBlockEnc

class BasicBlockEnc(nn.Module):

    def __init__(self, in_planes, stride=1):
        super().__init__()

        planes = in_planes * stride

        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.f = nn.LeakyReLU(0.1)

        if stride == 1:
            self.shortcut = nn.Sequential()
        else:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes)
            )

    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.f(out)
        return out


class ResNet18Enc(nn.Module):

    def __init__(self, num_Blocks=[2, 2, 2, 2], z_dim=10, nc=3):
        super().__init__()
        self.in_planes = 16
        self.z_dim = z_dim
        self.dim_h = 16
        self.conv1 = nn.Conv2d(nc, self.dim_h, kernel_size=3, stride=2, padding=1, bias=False)
        # self.bn1 = nn.BatchNorm2d(self.dim_h)
        self.layer1 = self._make_layer(BasicBlockEnc, self.dim_h, num_Blocks[0], stride=1)
        self.layer2 = self._make_layer(BasicBlockEnc, self.dim_h * 2, num_Blocks[1], stride=2)
        self.layer3 = self._make_layer(BasicBlockEnc, self.dim_h * 4, num_Blocks[2], stride=2)
        self.layer4 = self._make_layer(BasicBlockEnc, self.dim_h * 8, num_Blocks[3], stride=2)
        self.linear = nn.Linear(self.dim_h * 8, z_dim)
        self.pre_head = nn.Linear(self.dim_h * 8, 10)

    def _make_layer(self, BasicBlockEnc, planes, num_Blocks, stride):
        strides = [stride] + [1] * (num_Blocks - 1)
        layers = []
        for stride in strides:
            layers += [BasicBlockEnc(self.in_planes, stride)]
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x, pre=False):
        x = torch.relu(self.conv1(x))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = F.adaptive_avg_pool2d(x, 1)
        x = x.view(x.size(0), -1)
        if pre:
            y = self.pre_head(x)
            x = self.linear(x)
            return x, y
        x = self.linear(x)
        return x


class _netD(nn.Module):
    def __init__(self, args, decoder):
        super().__init__()

        self.args = args
        self.args.ndf = args.num_latent * 2
        self.args.nez = 1  # shapr of the output of ebm
        self.decoder = decoder
        self.encoder = ResNet18Enc(num_Blocks=[2, 2, 1, 1], z_dim=args.num_latent)

    def forward(self, z, pre=False):
        x = self.decoder(z)
        return self.encoder(x, pre)


class _netE(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.args = args
        self.args.ndf = args.num_latent * 2
        self.args.nez = 1  # shape of the output of ebm
        apply_sn = lambda x: x

        f = nn.LeakyReLU(0.1)
        # f = get_activation(args.e_activation, args)

        #         self.ebm = nn.Sequential(
        #             apply_sn(nn.Linear(args.num_latent, args.ndf)),
        #             f,

        #             apply_sn(nn.Linear(args.ndf, args.ndf)),
        #             f,

        #             apply_sn(nn.Linear(args.ndf, args.nez))
        #         )

        self.feature = nn.Sequential(
            apply_sn(nn.Linear(args.num_latent, args.ndf)),
            f,

            apply_sn(nn.Linear(args.ndf, args.ndf)),
            f,
        )

        self.ebm_head = nn.Sequential(
            apply_sn(nn.Linear(args.ndf, 1))
        )
        self.pre_head = nn.Sequential(
            apply_sn(nn.Linear(args.ndf, args.ndf)),
            f,

            apply_sn(nn.Linear(args.ndf, 10))
        )

    def forward(self, z, pre=False):
        output = self.feature(z.squeeze())  # .view(-1, 1)
        en = self.ebm_head(output).view(-1, 1)
        if pre:
            y = self.pre_head(output)  # .view(-1, 1)
            return en, y
        return en


def Lip_swish(x):
    return (x * torch.sigmoid(x)) / 1.1


class Normalize(nn.Module):
    def __init__(self, mean, std):
        super().__init__()
        self.register_buffer('mean', torch.tensor(mean))
        self.register_buffer('std', torch.tensor(std))

    def forward(self, x):
        x = (x - self.mean.view(1, -1, 1, 1)) / self.std.view(1, -1, 1, 1)
        return x

    def extra_repr(self):
        return f'mean={self.mean}, std={self.std}'


class GlobalAveragePooling(nn.Module):
    def forward(self, x):
        return F.adaptive_avg_pool2d(x, (1, 1)).view(x.shape[0], -1)


def get_mlp(dims, act_fn=nn.ReLU, bn=False, bias=True):
    layers = []
    for i in range(len(dims) - 1):
        if i < len(dims) - 2:
            layers.append(nn.Linear(dims[i], dims[i + 1], bias=not bn and bias))
            if bn:
                layers.append(nn.BatchNorm1d(dims[i + 1]))
            layers.append(act_fn())
        else:
            layers.append(nn.Linear(dims[i], dims[i + 1], bias=bias))
    mlp = nn.Sequential(*layers)
    mlp.out_dim = dims[-1]
    return mlp


class ResBlock(nn.Module):
    '''Pre-activation version of the BasicBlock.'''

    def __init__(self, in_channels, out_channels, act_fn='lrelu'):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.act_fn = act_fn
        if self.act_fn == 'lrelu':
            self.act = nn.LeakyReLU(0.2)
        else:
            self.act = nn.SiLU()

        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        shortcut = self.shortcut(x)
        x = self.conv1(self.act(x))
        x = self.conv2(self.act(x))
        x = x + shortcut
        return x


class ResNet(nn.Module):
    def __init__(self, layers, act_fn='lrelu', tau = 1e-5):
        super().__init__()
        self.act_fn = act_fn
        self.tau = tau
        if self.act_fn == 'lrelu':
            self.act = nn.LeakyReLU(0.2)
        else:
            self.act = nn.SiLU()

        in_channels = layers[0][0]
        self.stem = nn.Sequential(
            # Normalize(mean=mean, std=std),
            nn.Conv2d(3, in_channels, kernel_size=3, stride=1, padding=1))

        stages = []
        for i, (out_channels, num_blocks) in enumerate(layers):
            stage = []
            if i > 0:
                stage.append(nn.AvgPool2d(2))
            for _ in range(num_blocks):
                stage.append(ResBlock(in_channels, out_channels, self.act_fn))
                in_channels = out_channels
            stages.append(nn.Sequential(*stage))
        self.stages = nn.Sequential(*stages)

        self.pool = GlobalAveragePooling()
        if self.act_fn == 'lrelu':
            self.head = get_mlp([in_channels, 2048, 1],
                                act_fn=lambda: nn.LeakyReLU(0.2), bn=False, bias=True)
        else:
            self.head = get_mlp([in_channels, 2048, 1],
                                act_fn=lambda: nn.SiLU(), bn=False, bias=True)

        for name, m in self.named_modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0., .01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='linear')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        out = self.stem(x)
        out = self.stages(out)
        out = self.pool(self.act(out))
        f = self.head(out)
        return f / self.tau


class MSResNet(nn.Module):
    def __init__(self, layers, agg=False, tau = 1e-5):
        super().__init__()
        self.act = nn.LeakyReLU(0.2)
        self.agg = agg
        self.tau = tau
        in_channels = layers[0][0]
        self.stem1 = nn.Sequential(
            # Normalize(mean=mean, std=std),
            nn.Conv2d(3, in_channels, kernel_size=3, stride=1, padding=1))
        self.stem2 = nn.Sequential(
            # Normalize(mean=mean, std=std),
            nn.Conv2d(3, in_channels, kernel_size=3, stride=1, padding=1))
        self.stem3 = nn.Sequential(
            # Normalize(mean=mean, std=std),
            nn.Conv2d(3, in_channels, kernel_size=3, stride=1, padding=1))

        stages = []
        in_channels = layers[0][0]
        for i, (out_channels, num_blocks) in enumerate(layers):
            stage = []
            if i > 0:
                stage.append(nn.AvgPool2d(2))
            for _ in range(num_blocks):
                stage.append(ResBlock(in_channels, out_channels))
                in_channels = out_channels
            stages.append(nn.Sequential(*stage))
        self.stages1 = nn.Sequential(*stages)

        stages = []
        in_channels = layers[0][0]
        for i, (out_channels, num_blocks) in enumerate(layers):
            stage = []
            if i > 0:
                stage.append(nn.AvgPool2d(2))
            for _ in range(num_blocks):
                stage.append(ResBlock(in_channels, out_channels))
                in_channels = out_channels
            stages.append(nn.Sequential(*stage))
        self.stages2 = nn.Sequential(*stages)

        stages = []
        in_channels = layers[0][0]
        for i, (out_channels, num_blocks) in enumerate(layers):
            stage = []
            if i > 0:
                stage.append(nn.AvgPool2d(2))
            for _ in range(num_blocks):
                stage.append(ResBlock(in_channels, out_channels))
                in_channels = out_channels
            stages.append(nn.Sequential(*stage))
        self.stages3 = nn.Sequential(*stages)

        self.pool = GlobalAveragePooling()
        self.head = get_mlp([in_channels * 3, 2048, 1],
                            act_fn=lambda: nn.LeakyReLU(0.2), bn=False, bias=True)
        # self.projection = get_mlp(projection_layers,
        #                           act_fn=lambda: nn.LeakyReLU(0.2), bn=False, bias=False)
        # self.projection_layers = projection_layers

        for name, m in self.named_modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0., .01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='linear')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        out1 = self.stages1(self.stem1(x))
        out2 = self.stages2(self.stem2(F.avg_pool2d(x, 2)))
        out3 = self.stages3(self.stem3(F.avg_pool2d(x, 4)))
        out = torch.cat([self.pool(self.act(out1)), self.pool(self.act(out2)), self.pool(self.act(out3))], dim=1)
        f = self.head(out)
        return f / self.tau

# def get_ebm(name, input_shape, projection_layers,
#             mean: list[float], std: list[float]):
#     if name == 'resnet':
#         layers = [(128, 2), (128, 2), (256, 2), (256, 2)]
#         encoder = ResNet(layers, projection_layers=projection_layers, mean=mean, std=std)

#     elif name == 'resnet_small':
#         layers = [(64, 1), (64, 1), (128, 1), (128, 1)]
#         encoder = ResNet(layers, projection_layers=projection_layers, mean=mean, std=std)

#     elif name == 'resnet64':
#         layers = [(64, 2), (128, 2), (128, 2), (256, 2), (256, 2)]
#         encoder = ResNet(layers, projection_layers=projection_layers, mean=mean, std=std)

#     else:
#         raise Exception(f'Unknown Encoder: {name}')

#     for m in encoder.modules():
#         if isinstance(m, nn.Conv2d):
#             nn.utils.parametrizations.spectral_norm(m)

#     return encoder
