import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.nn.utils import spectral_norm


class NoiseInjection(nn.Module):
    def __init__(self, std=0.1):
        super().__init__()

        # self.weight =  nn.Parameter(torch.zeros(1), requires_grad=True) #
        # LEARNED NOISE IS USELESS.
        self.weight = std

    def forward(self, feat, noise=None):
        if noise is None:
            batch, _, height, width = feat.shape
            noise = torch.randn(batch, 1, height, width).to(feat.device)
        # self.weight = torch.clamp(self.weight, min = 0.1)
        return feat + self.weight * noise


class ResizeConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, scale_factor, mode='nearest'):
        super().__init__()
        self.scale_factor = scale_factor
        self.mode = mode
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode)
        x = self.conv(x)
        return x


class BasicBlockEnc(nn.Module):

    def __init__(self, in_planes, stride=1):
        super().__init__()

        planes = in_planes * stride

        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

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
        out = torch.relu(out)
        return out


class BasicBlockDec(nn.Module):

    def __init__(self, in_planes, stride=1):
        super().__init__()

        planes = int(in_planes / stride)

        self.conv2 = nn.Conv2d(in_planes, in_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(in_planes)
        # self.bn1 could have been placed here, but that messes up the order of the layers when printing the class

        if stride == 1:
            self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(planes)
            self.shortcut = nn.Sequential()
        else:
            self.conv1 = ResizeConv2d(in_planes, planes, kernel_size=3, scale_factor=stride)
            self.bn1 = nn.BatchNorm2d(planes)
            self.shortcut = nn.Sequential(
                ResizeConv2d(in_planes, planes, kernel_size=3, scale_factor=stride),
                nn.BatchNorm2d(planes),
                # NoiseInjection(),
            )

    def forward(self, x):
        out = torch.relu(self.bn2(self.conv2(x)))
        out = self.bn1(self.conv1(out))
        # out = NoiseInjection()(out)
        out += self.shortcut(x)
        out = torch.relu(out)
        return out

class ResNet18Enc(nn.Module):
    def __init__(self, num_Blocks=[2, 2, 2, 2], z_dim=10, nc=3, use_vae=False):
        super().__init__()
        self.z_dim = z_dim
        self.dim_h = 128
        self.in_planes = self.dim_h
        self.use_vae = True
        self.conv1 = nn.Conv2d(nc, self.dim_h, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.dim_h)
        self.layer1 = self._make_layer(BasicBlockEnc, self.dim_h, num_Blocks[0], stride=1)
        self.layer2 = self._make_layer(BasicBlockEnc, self.dim_h * 2, num_Blocks[1], stride=2)
        self.layer3 = self._make_layer(BasicBlockEnc, self.dim_h * 4, num_Blocks[2], stride=2)
        self.layer4 = self._make_layer(BasicBlockEnc, self.dim_h * 8, num_Blocks[3], stride=2)
        self.linear = nn.Linear(self.dim_h * 8, z_dim)
        if use_vae:
            self.logvar_linaer = nn.Linear(self.dim_h * 8, z_dim)
        initialize_weights(self)

    def _make_layer(self, BasicBlockEnc, planes, num_Blocks, stride):
        strides = [stride] + [1] * (num_Blocks - 1)
        layers = []
        for stride in strides:
            layers += [BasicBlockEnc(self.in_planes, stride)]
            self.in_planes = planes
        return nn.Sequential(*layers)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z

    def forward(self, x,):
        x = torch.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = F.adaptive_avg_pool2d(x, 1)
        x = x.view(x.size(0), -1)
        output = self.linear(x)
        output = F.tanh(output)
        if self.use_vae:
            logvar = self.logvar_linaer(x)
            return output,logvar
        return output


class ResNet18Dec(nn.Module):

    def __init__(self, args, num_Blocks=[2, 2, 2, 2], z_dim=10, nc=3):
        super().__init__()
        self.in_planes = 512
        self.dim_h = 64
        self.linear = nn.Linear(z_dim, self.dim_h * 8)
        # final_strid =
        if args.dataset in ['celeba', 'bedroom', 'church']:
            sc_factor, final_stride = 2, 2
        else:
            sc_factor, final_stride = 1, 1
        self.layer4 = self._make_layer(BasicBlockDec, self.dim_h * 4, num_Blocks[3], stride=2)
        self.layer3 = self._make_layer(BasicBlockDec, self.dim_h * 2, num_Blocks[2], stride=2)
        self.layer2 = self._make_layer(BasicBlockDec, self.dim_h * 1, num_Blocks[1], stride=2)
        self.noise_inj = NoiseInjection()

        self.layer1 = self._make_layer(BasicBlockDec, self.dim_h * 1, num_Blocks[0], stride=1)
        self.conv1 = ResizeConv2d(self.dim_h, nc, kernel_size=3, scale_factor=sc_factor)

        if args.dataset == 'celebahq':
            self.conv1 = nn.Sequential(
                ResizeConv2d(self.dim_h, self.dim_h // 2, kernel_size=3, scale_factor=2),
                NoiseInjection(),
                ResizeConv2d(self.dim_h // 2, self.dim_h // 2, kernel_size=3, scale_factor=2),
                ResizeConv2d(self.dim_h // 2, nc, kernel_size=3, scale_factor=2),
            )
        elif '128' in args.dataset:
            self.conv1 = nn.Sequential(
                ResizeConv2d(self.dim_h, self.dim_h // 2, kernel_size=3, scale_factor=2),
                NoiseInjection(),
                ResizeConv2d(self.dim_h // 2, self.dim_h // 2, kernel_size=3, scale_factor=2),
                ResizeConv2d(self.dim_h // 2, nc, kernel_size=3, scale_factor=1),
            )

    def _make_layer(self, BasicBlockDec, planes, num_Blocks, stride):
        strides = [stride] + [1] * (num_Blocks - 1)
        layers = []
        for stride in reversed(strides):
            layers += [BasicBlockDec(self.in_planes, stride)]
        self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, z):
        x = self.linear(z)
        x = x.view(z.size(0), 512, 1, 1)
        x = F.interpolate(x, scale_factor=4)
        x = self.noise_inj(x)

        x = self.layer4(x)
        x = self.noise_inj(x)

        x = self.layer3(x)
        x = self.noise_inj(x)

        x = self.layer2(x)
        x = self.noise_inj(x)

        x = self.layer1(x)
        # print(x.shape)
        x = torch.sigmoid(self.conv1(x))  # [0,1]
        # x = x.view(x.size(0), 3, 32, 32)
        return x


def initialize_weights(net):
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            m.weight.data.normal_(0, 0.02)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.ConvTranspose2d):
            m.weight.data.normal_(0, 0.02)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            m.weight.data.normal_(0, 0.02)
            if m.bias is not None:
                m.bias.data.zero_()


class Res_AE(nn.Module):

    def __init__(self, args):
        super().__init__()
        self.encoder = ResNet18Enc(z_dim=args.num_latent)
        self.decoder = ResNet18Dec(args, z_dim=args.num_latent)
        initialize_weights(self)

    def forward(self, x):
        z = self.encoder(x)
        x = self.decoder(z)
        return x, z

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

    def init(self):
        def weights_init(m):
            init_type = "xavier_uniform"
            classname = m.__class__.__name__
            if classname.find('Conv2d') != -1:
                nn.init.xavier_uniform(m.weight.data, 1.)
            elif classname.find('BatchNorm2d') != -1:
                nn.init.normal_(m.weight.data, 1.0, 0.02)
                nn.init.constant_(m.bias.data, 0.0)

        self.encoder.apply(weights_init)
        self.decoder.apply(weights_init)


class latent_ebm(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.main = nn.Sequential(
            spectral_norm(nn.Linear(args.num_latent, 512)),
            nn.ReLU(),
            spectral_norm(nn.Linear(512, 512)),
            nn.ReLU(),
            spectral_norm(nn.Linear(512, 512)),
            nn.ReLU(),
            spectral_norm(nn.Linear(512, 1)),
        )
        initialize_weights(self)

    def forward(self, x):
        output = self.main(x)
        output = torch.clamp(output, min=-1, max=1)
        return output


class simple_dis(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.main = nn.Sequential(
            spectral_norm(nn.Linear(args.num_latent, 1)),
        )  # Linear Head

    def forward(self, x):
        output = self.main(x)
        output = torch.clamp(output, min=-1, max=1)
        return output
