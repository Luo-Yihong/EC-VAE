import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm
import torch.nn.functional as F
import random
import FrEIA.framework as Ff
import FrEIA.modules as Fm


class View(nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.shape = shape,  # extra comma

    def forward(self, x):
        return x.view(*self.shape)


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        try:
            m.weight.data.normal_(0.0, 0.02)
        except:
            pass
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def conv2d(*args, **kwargs):
    return spectral_norm(nn.Conv2d(*args, **kwargs))


def convTranspose2d(*args, **kwargs):
    return spectral_norm(nn.ConvTranspose2d(*args, **kwargs))


def batchNorm2d(*args, **kwargs):
    return nn.BatchNorm2d(*args, **kwargs)


def linear(*args, **kwargs):
    return spectral_norm(nn.Linear(*args, **kwargs))


class PixelNorm(nn.Module):
    def forward(self, input):
        return input * torch.rsqrt(torch.mean(input ** 2, dim=1, keepdim=True) + 1e-8)


class Reshape(nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.target_shape = shape

    def forward(self, feat):
        batch = feat.shape[0]
        return feat.view(batch, *self.target_shape)


class GLU(nn.Module):
    def forward(self, x):
        nc = x.size(1)
        assert nc % 2 == 0, 'channels dont divide 2!'
        nc = int(nc / 2)
        return x[:, :nc] * torch.sigmoid(x[:, nc:])


class NoiseInjection(nn.Module):
    def __init__(self):
        super().__init__()

        # self.weight = nn.Parameter(torch.zeros(1), requires_grad=True)
        self.weight = 0.1

    def forward(self, feat, noise=None):
        return feat


class Swish(nn.Module):
    def forward(self, feat):
        return feat * torch.sigmoid(feat)


class SEBlock(nn.Module):
    def __init__(self, ch_in, ch_out):
        super().__init__()

        self.main = nn.Sequential(nn.AdaptiveAvgPool2d(4),
                                  conv2d(ch_in, ch_out, 4, 1, 0, bias=False), Swish(),
                                  conv2d(ch_out, ch_out, 1, 1, 0, bias=False), nn.Sigmoid())

    def forward(self, feat_small, feat_big):
        return feat_big * self.main(feat_small)


class InitLayer(nn.Module):
    def __init__(self, nz, channel):
        super().__init__()

        self.init = nn.Sequential(
            convTranspose2d(nz, channel * 2, 4, 1, 0, bias=False),
            batchNorm2d(channel * 2), GLU())

    def forward(self, noise):
        noise = noise.view(noise.shape[0], -1, 1, 1)
        return self.init(noise)


def UpBlock(in_planes, out_planes):
    block = nn.Sequential(
        nn.Upsample(scale_factor=2, mode='nearest'),
        conv2d(in_planes, out_planes * 2, 3, 1, 1, bias=False),
        # convTranspose2d(in_planes, out_planes*2, 4, 2, 1, bias=False),
        batchNorm2d(out_planes * 2), GLU())
    return block


def UpBlockComp(in_planes, out_planes):
    block = nn.Sequential(
        nn.Upsample(scale_factor=2, mode='nearest'),
        conv2d(in_planes, out_planes * 2, 3, 1, 1, bias=False),
        # convTranspose2d(in_planes, out_planes*2, 4, 2, 1, bias=False),
        batchNorm2d(out_planes * 2), GLU(),
        conv2d(out_planes, out_planes * 2, 3, 1, 1, bias=False),
        batchNorm2d(out_planes * 2), GLU()
    )
    return block


def build_flow(hidden_units, num_block, num_latent, name='nvp'):
    def subnet_constructor(c_in, c_out):
        return nn.Sequential(nn.Linear(c_in, hidden_units), nn.BatchNorm1d(hidden_units), nn.ReLU(),
                             nn.Linear(hidden_units, hidden_units), nn.BatchNorm1d(hidden_units), nn.ReLU(),
                             nn.Linear(hidden_units,
                                       c_out))  # newest version with activation at the last layer -> cancel the act at the last layer

    nodes = [Ff.InputNode(num_latent, name='input')]
    for k in range(num_block):
        nodes.append(Ff.Node(nodes[-1],
                             Fm.RNVPCouplingBlock,
                             {'subnet_constructor': subnet_constructor, 'clamp': 2.0}, ))
        nodes.append(Ff.Node(nodes[-1],
                             Fm.PermuteRandom,
                             {'seed': k}, ))

    nodes.append(Ff.OutputNode(nodes[-1], name='output'))
    flow = Ff.ReversibleGraphNet(nodes, verbose=False)
    return flow


class Generator(nn.Module):
    def __init__(self, ngf=64, nz=128, nc=3, im_size=256):
        super(Generator, self).__init__()

        nfc_multi = {4: 16, 8: 8, 16: 4, 32: 2, 64: 2, 128: 1, 256: 0.5, 512: 0.25, 1024: 0.125}
        nfc = {}
        for k, v in nfc_multi.items():
            nfc[k] = int(v * ngf)

        self.im_size = im_size

        self.init = InitLayer(nz, channel=nfc[4])

        self.feat_8 = UpBlockComp(nfc[4], nfc[8])
        self.feat_16 = UpBlock(nfc[8], nfc[16])
        self.feat_32 = UpBlockComp(nfc[16], nfc[32])

        # self.to_32 = nn.Sequential(conv2d(nfc[32], nc, 1, 1, 0, bias=False),
        #                                 nn.Sigmoid())

        self.feat_64 = UpBlock(nfc[32], nfc[64])
        self.feat_128 = UpBlockComp(nfc[64], nfc[128])
        self.feat_256 = UpBlock(nfc[128], nfc[256])

        self.se_64 = SEBlock(nfc[4], nfc[64])
        self.se_128 = SEBlock(nfc[8], nfc[128])
        self.se_256 = SEBlock(nfc[16], nfc[256])

        if self.im_size == 128:
            self.to_64 = nn.Sequential(conv2d(nfc[64], nc, 1, 1, 0, bias=False),
                                       nn.Tanh())

        self.to_128 = nn.Sequential(conv2d(nfc[128], nc, 1, 1, 0, bias=False),
                                    nn.Tanh())
        if self.im_size > 128:
            self.to_big = nn.Sequential(conv2d(nfc[im_size], nc, 3, 1, 1, bias=False),
                                        nn.Tanh())

        if im_size > 256:
            self.feat_512 = UpBlockComp(nfc[256], nfc[512])
            self.se_512 = SEBlock(nfc[32], nfc[512])
        if im_size > 512:
            self.feat_1024 = UpBlock(nfc[512], nfc[1024])

    def forward(self, input, mode='single'):

        feat_4 = self.init(input)
        # print(feat_4.shape)
        feat_8 = self.feat_8(feat_4)
        feat_16 = self.feat_16(feat_8)
        # print(feat_16.shape)
        feat_32 = self.feat_32(feat_16)

        feat_64 = self.se_64(feat_4, self.feat_64(feat_32))

        feat_128 = self.se_128(feat_8, self.feat_128(feat_64))

        if self.im_size == 128:
            if mode == 'multi':
                return [self.to_128(feat128), self.to_64(feat_64), feat128]
            return [self.to_128(feat128), self.to_64(feat_64)]

        feat_256 = self.se_256(feat_16, self.feat_256(feat_128))

        if self.im_size == 256:
            if mode == 'multi':
                return [self.to_big(feat_256), self.to_128(feat_128), feat_256]
            else:
                return self.to_big(feat_256)
            # return [self.to_big(feat_256), self.to_128(feat_128)]

        feat_512 = self.se_512(feat_32, self.feat_512(feat_256))
        if self.im_size == 512:
            return [self.to_big(feat_512), self.to_128(feat_128)]

        feat_1024 = self.feat_1024(feat_512)

        im_128 = torch.tanh(self.to_128(feat_128))
        im_1024 = torch.tanh(self.to_big(feat_1024))

        return [im_1024, im_128]


class DownBlock(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(DownBlock, self).__init__()

        self.main = nn.Sequential(
            conv2d(in_planes, out_planes, 4, 2, 1, bias=False),
            batchNorm2d(out_planes), nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, feat):
        return self.main(feat)


class DownBlockComp(nn.Module):
    def __init__(self, in_planes, out_planes, down=True):
        super(DownBlockComp, self).__init__()
        if down:
            self.main = nn.Sequential(
                conv2d(in_planes, out_planes, 4, 2, 1, bias=False),
                batchNorm2d(out_planes), nn.LeakyReLU(0.2, inplace=True),
                conv2d(out_planes, out_planes, 3, 1, 1, bias=False),
                batchNorm2d(out_planes), nn.LeakyReLU(0.2)
            )

            self.direct = nn.Sequential(
                nn.AvgPool2d(2, 2),
                conv2d(in_planes, out_planes, 1, 1, 0, bias=False),
                batchNorm2d(out_planes), nn.LeakyReLU(0.2))
        else:
            self.main = nn.Sequential(
                conv2d(in_planes, out_planes, 3, 1, 1, bias=False),
                batchNorm2d(out_planes), nn.LeakyReLU(0.2, inplace=True),
                conv2d(out_planes, out_planes, 3, 1, 1, bias=False),
                batchNorm2d(out_planes), nn.LeakyReLU(0.2)
            )

            self.direct = nn.Sequential(
                # nn.AvgPool2d(2, 2),
                conv2d(in_planes, out_planes, 1, 1, 0, bias=False),
                batchNorm2d(out_planes), nn.LeakyReLU(0.2))

    def forward(self, feat):
        return (self.main(feat) + self.direct(feat)) / 2


class Discriminator(nn.Module):
    def __init__(self, ndf=64, nc=3, im_size=256, num_latent=128):
        super(Discriminator, self).__init__()
        self.ndf = ndf
        self.im_size = im_size

        nfc_multi = {4: 16, 8: 16, 16: 8, 32: 4, 64: 2, 128: 1, 256: 0.5, 512: 0.25, 1024: 0.125}
        nfc = {}
        for k, v in nfc_multi.items():
            nfc[k] = int(v * ndf)

        if im_size == 1024:
            self.down_from_big = nn.Sequential(
                conv2d(nc, nfc[1024], 4, 2, 1, bias=False),
                nn.LeakyReLU(0.2, inplace=True),
                conv2d(nfc[1024], nfc[512], 4, 2, 1, bias=False),
                batchNorm2d(nfc[512]),
                nn.LeakyReLU(0.2, inplace=True))
        elif im_size == 512:
            self.down_from_big = nn.Sequential(
                conv2d(nc, nfc[512], 4, 2, 1, bias=False),
                nn.LeakyReLU(0.2, inplace=True))
        else:
            self.down_from_big = nn.Sequential(
                conv2d(nc, nfc[512], 3, 1, 1, bias=False),
                nn.LeakyReLU(0.2, inplace=True))

        self.down_4 = DownBlockComp(nfc[512], nfc[256])
        self.down_8 = DownBlockComp(nfc[256], nfc[128])
        self.down_16 = DownBlockComp(nfc[128], nfc[64])
        self.down_32 = DownBlockComp(nfc[64], nfc[32])
        self.down_64 = DownBlockComp(nfc[32], nfc[16], down=True)

        self.rf_big = nn.Sequential(
            conv2d(nfc[16], nfc[8], 1, 1, 0, bias=False),
            batchNorm2d(nfc[8]), nn.LeakyReLU(0.2, inplace=True),
            conv2d(nfc[8], num_latent // 2, 4, 1, 0, bias=False))

        self.linear_head = nn.Sequential(
            nn.Linear(num_latent // 2 * 5 * 5, num_latent),
        )

        self.linear_head2 = nn.Sequential(
            nn.Linear(num_latent // 2 * 5 * 5, num_latent),
        )
        self.se_2_16 = SEBlock(nfc[512], nfc[64])
        self.se_4_32 = SEBlock(nfc[256], nfc[32])
        self.se_8_64 = SEBlock(nfc[128], nfc[16])

    def forward(self, imgs, part=None):
        if type(imgs) is not list:
            imgs = [F.interpolate(imgs, size=self.im_size), F.interpolate(imgs, size=128)]

        feat_2 = self.down_from_big(imgs[0])
        feat_4 = self.down_4(feat_2)
        feat_8 = self.down_8(feat_4)

        feat_16 = self.down_16(feat_8)
        feat_16 = self.se_2_16(feat_2, feat_16)

        feat_32 = self.down_32(feat_16)
        feat_32 = self.se_4_32(feat_4, feat_32)

        feat_last = self.down_64(feat_32)
        feat_last = self.se_8_64(feat_8, feat_last)

        # print(feat_last.shape)

        # print(feat_last.shape) # [bs,512,8,8]

        # rf_0 = torch.cat([self.rf_big_1(feat_last).view(-1),self.rf_big_2(feat_last).view(-1)])
        # rff_big = torch.sigmoid(self.rf_factor_big)
        rf_0 = self.rf_big(feat_last)
        # print(rf_0.shape)
        rf_0 = rf_0.view(rf_0.size(0), -1)
        mu = self.linear_head(rf_0)
        logsig = self.linear_head2(rf_0)
        return mu, logsig


class fast_ae(nn.Module):
    def __init__(self, args):
        super(fast_ae, self).__init__()
        self.decoder = Generator(nz=args.num_latent)
        self.encoder = Discriminator(num_latent=args.num_latent)

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z, mode='single'):
        output = self.decoder(z, mode)
        if mode == 'single':
            return output  # [0]
        else:
            return output

    def forward(self, x):
        z = self.encoder(x)
        output = self.decoder(z)
        return output, z


class Generator_32(nn.Module):
    def __init__(self, ngf=64, nz=128, nc=3, im_size=256):
        super(Generator_32, self).__init__()

        nfc_multi = {4: 16, 8: 8, 16: 8, 32: 8, 64: 2, 128: 1, 256: 0.5, 512: 0.25, 1024: 0.125}
        nfc = {}
        for k, v in nfc_multi.items():
            nfc[k] = int(v * ngf)

        self.im_size = im_size

        self.init = InitLayer(nz, channel=nfc[4])

        self.feat_8 = UpBlockComp(nfc[4], nfc[8])
        self.feat_16 = UpBlock(nfc[8], nfc[16])
        self.feat_32 = UpBlockComp(nfc[16], nfc[32])

        self.to_32 = nn.Sequential(conv2d(nfc[32], nc, 1, 1, 0, bias=False),
                                   nn.Tanh())

    def forward(self, input, mode='single'):
        feat_4 = self.init(input)
        # print(feat_4.shape)
        feat_8 = self.feat_8(feat_4)
        feat_16 = self.feat_16(feat_8)
        # print(feat_16.shape)
        feat_32 = self.feat_32(feat_16)
        return self.to_32(feat_32)


from models.res import ResNet18Enc


class fast_vae_32(nn.Module):
    def __init__(self, args):
        super(fast_vae_32, self).__init__()
        self.decoder = Generator_32(nz=args.num_latent)
        self.encoder = ResNet18Enc(z_dim=args.num_latent, use_vae=True)

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.encoder.reparameterize(mu, logvar)
        output = self.decoder(z)
        return output, z, mu, logvar

    def encode(self, x):
        mu, logvar = self.encoder(x)
        z = self.encoder.reparameterize(mu, logvar)
        return z

    def decode(self, z):
        return self.decoder(z)
