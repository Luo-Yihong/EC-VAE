import random

import torch
import numpy as np
from tqdm import tqdm
from torchvision import utils
import os
from torch import nn
import asyncio
import time
from FrEIA.framework import InputNode, OutputNode, Node, ReversibleGraphNet
from FrEIA.modules import RNVPCouplingBlock, PermuteRandom
import asyncio
from torchvision import utils
import torch.nn.functional as F


def dual_step(lam, loss, eps, lr=1e-2, max_=100, min_=0):
    lam = lam + lr * (loss.item() - eps)
    lam = min(lam, max_)
    lam = max(lam, min_)
    return lam


def background(f):
    def wrapped(*args, **kwargs):
        return asyncio.get_event_loop().run_in_executor(None, f, *args, **kwargs)

    return wrapped


@background
def save_img_(img, name):
    utils.save_image(img, name, normalize=True)  # efficient save_img


def build_flow(fc_dim, num_block, num_latent, name='nvp', act_fn='relu'):
    if act_fn == 'relu':
        act = nn.ReLU
    else:
        act = SwishFn

    def subnet_fc(c_in, c_out):
        return nn.Sequential(nn.Linear(c_in, fc_dim), nn.BatchNorm1d(fc_dim), act(),
                             nn.Linear(fc_dim, fc_dim), nn.BatchNorm1d(fc_dim), act(),
                             nn.Linear(fc_dim,
                                       c_out))  # newest version with activation at the last layer -> cancel the act at the last layer

    nodes = [InputNode(num_latent, name='input')]
    for k in range(num_block):
        nodes.append(Node(nodes[-1],
                          RNVPCouplingBlock,
                          {'subnet_constructor': subnet_fc, 'clamp': 2.0},
                          name=F'coupling_{k}'))
        nodes.append(Node(nodes[-1],
                          PermuteRandom,
                          {'seed': k},
                          name=F'permute_{k}'))

    nodes.append(OutputNode(nodes[-1], name='output'))
    flow = ReversibleGraphNet(nodes, verbose=False)
    return flow


class SampleBuffer:
    def __init__(self, max_samples=10000):
        self.max_samples = max_samples
        self.buffer = []

    def __len__(self):
        return len(self.buffer)

    def push(self, samples, class_ids=None):
        samples = samples.detach().to('cpu')
        if class_ids is not None:
            class_ids = class_ids.detach().to('cpu')
            for sample, class_id in zip(samples, class_ids):
                self.buffer.append((sample.detach(), class_id))

                if len(self.buffer) > self.max_samples:
                    self.buffer.pop(0)
        else:
            for sample in samples:
                self.buffer.append((sample.detach(), 0))  # the 0 is fake_id
                if len(self.buffer) > self.max_samples:
                    self.buffer.pop(0)

    def get(self, n_samples, device='cuda'):
        items = random.choices(self.buffer, k=n_samples)
        samples, class_ids = zip(*items)
        samples = torch.stack(samples, 0)
        class_ids = torch.tensor(class_ids)
        samples = samples.to(device)
        class_ids = class_ids.to(device)

        return samples, class_ids


def sample_buffer(buffer, batch_size=128, p=0.95, device='cuda'):
    if len(buffer) < 1:
        return (
            torch.rand(batch_size, 3, 32, 32, device=device),
            torch.randint(0, 10, (batch_size,), device=device),
        )

    n_replay = (np.random.rand(batch_size) < p).sum()

    replay_sample, replay_id = buffer.get(n_replay)
    random_sample = torch.rand(batch_size - n_replay, 3, 32, 32, device=device)
    random_id = torch.randint(0, 10, (batch_size - n_replay,), device=device)

    return (
        torch.cat([replay_sample, random_sample], 0),
        torch.cat([replay_id, random_id], 0),
    )


@torch.enable_grad()
def sgld_z(model, img, z, decoder, steps=60, step_size=1e-6, class_id=None, show=True):
    # img.requires_grad = True
    img = img.clone().detach()
    z.requires_grad = True
    img = decoder(z)
    delta = 0
    for k in tqdm(range(steps)):
        noise_z = torch.randn_like(z).cuda() * (step_size ** 0.5)
        energy = model(img + delta)
        g = torch.autograd.grad(energy.sum(), [img], retain_graph=True)[0]
        delta = (-g * step_size / 2).clone().detach()
        g_z = torch.autograd.grad(energy.sum(), [z])[0]
        z = z - g_z * (step_size / 2) + noise_z
        # z = z.clamp(-1.5, 1.5).clone().detach()
        z.requires_grad = True
        img = decoder(z)
    img = img.clone().detach()
    return img, z


@torch.enable_grad()
def sgld(model, img, condition_img=None, steps=30, step_size=1e-6, class_id=None, show=False):
    # img.requires_grad = True
    if condition_img is None:
        img0 = img.clone().detach()
    else:
        img0 = condition_img.clone().detach()
    img = img.clone().detach()
    img.requires_grad = True
    lam_init = 1

    def helper(img, k):
        img.requires_grad = True
        noise = torch.randn_like(img).cuda() * (step_size ** 0.5)
        img = img  # perturb the image
        if class_id is None:
            energy = model(img)
        else:
            energy = model(img, class_id)
        energy_mse = ((img.clamp(-1, 1) - img0.clamp(-1, 1)) ** 2).view(img.shape[0], -1).sum(dim=1)
        energy_mse = lam_init * (steps - k - 1) / steps * (F.relu(energy_mse - 0.07) + 0.07)
        g_con = torch.autograd.grad(energy_mse.sum(), [img])[0]
        g = torch.autograd.grad(energy.sum(), [img], retain_graph=True)[0]  # get the gradient of p(x)
        if model.sampling_strategy == 'mcmc':
            img = img - step_size * 0.5 * (g + g_con) + noise
        else:
            l = len(img.shape) - 1
            g_norm = torch.norm(g.reshape(g.shape[0], -1), dim=1).reshape(-1, *([1] * l))
            img = img - g / (g_norm + 1e-10) * step_size
        return img.clone().detach()

    if show:
        for k in tqdm(range(steps)):
            img = helper(img, k)
    else:
        for k in range(steps):
            img = helper(img, k)
    return img.detach()


@torch.enable_grad()
def sgld_joint(model, img, z, decoder, steps=60, step_size=10, step_size_2=0.1, class_id=None,
               show=True):
    # img.requires_grad = True
    img = img.clone().detach()
    z = z.clone().detach()

    # img0 = img.clone().detach()
    def helper(img, z):
        img.requires_grad = True
        z.requires_grad = True
        noise = torch.randn_like(img).cuda() * (step_size ** 0.5)
        noise_z = torch.randn_like(z).cuda() * (step_size_2 ** 0.5)
        img_ = decoder(z)
        img = (img - img_).detach() + img_
        img = img + noise  # perturb the image
        if class_id is None:
            energy = model(img)
        else:
            energy = model(img, class_id)
        g = torch.autograd.grad(energy.sum(), [img], retain_graph=True)[0]  # get the gradient of p(x)
        g_z = torch.autograd.grad(energy.sum(), [z])[0]
        if model.sampling_strategy == 'mcmc':
            img = img - g * step_size * 0.5  # we have added the noise before, so here noise free
            z = z - g_z * step_size_2 * 0.5 + noise_z
        else:
            l = len(img.shape) - 1
            g_norm = torch.norm(g.reshape(g.shape[0], -1), dim=1).reshape(-1, *([1] * l))
            img = img - g / (g_norm + 1e-10) * step_size
        img = img.clone().detach()  # equal to grad_zero
        return img.detach(), z.detach()

    if show:
        for k in tqdm(range(steps)):
            img, z = helper(img, z)
    else:
        for _ in range(steps):
            img, z = helper(img, z)
    return img.detach(), z.detach()


def apply_sn(m):
    for name, layer in m.named_children():
        m.add_module(name, apply_sn(layer))
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        return nn.utils.spectral_norm(m)
    else:
        return m


def makedirs(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)


def join_dir(path1, path2):
    res = os.path.join(path1, path2)
    return res


def save_img(img, folder='./samples', file_name='sample', nrow=16):
    # Assuming the img range is (0,1)
    makedirs(folder)
    utils.save_image(
        img,
        join_dir(folder, file_name),
        nrow=nrow,
        normalize=True,
        # range=(0, 1),
        # padding=8
    )


def sample_data(loader):
    loader_iter = iter(loader)

    while True:
        try:
            yield next(loader_iter)

        except StopIteration:
            loader_iter = iter(loader)

            yield next(loader_iter)


def requires_grad(parameters, flag=True):
    for p in parameters:
        p.requires_grad = flag


def clip_grad(parameters, optimizer):
    with torch.no_grad():
        for group in optimizer.param_groups:
            for p in group['params']:
                state = optimizer.state[p]

                if 'step' not in state or state['step'] < 1:
                    continue

                step = state['step']
                exp_avg_sq = state['exp_avg_sq']
                _, beta2 = group['betas']

                bound = 3 * torch.sqrt(exp_avg_sq / (1 - beta2 ** step)) + 0.1
                p.grad.data.copy_(torch.max(torch.min(p.grad.data, bound), -bound))


class SwishFn(nn.Module):

    def __init__(self):
        super(SwishFn, self).__init__()
        self.beta = nn.Parameter(torch.tensor([0.5]))

    def forward(self, x):
        return (x * torch.sigmoid_(x * F.softplus(self.beta))).div_(1.1)


def background(f):
    def wrapped(*args, **kwargs):
        return asyncio.get_event_loop().run_in_executor(None, f, *args, **kwargs)

    return wrapped


@background
def save_img_(img, name):
    utils.save_image(img, name, normalize=True)


def generate_(decoder, flow, energy_fn, args, folder="./exp", device='cuda', ebm=True, ebm_z=False):
    makedirs(join_dir(folder, "samples"))
    path = join_dir(folder, "samples")
    makedirs(os.path.join(path, 'ebm'))
    makedirs(os.path.join(path, 'decoder'))
    batch_size = 500
    if args.dataset == 'celebahq':
        batch_size = 50
    with torch.no_grad():
        for i in tqdm(range(10000 // batch_size)):
            noise = torch.randn(batch_size, args.num_latent).to(device)
            zs, _ = flow(noise, rev=True)
            sample = decoder(zs)  # belongs to [0,1]
            nc, h, w = sample.shape[1:]
            for j in range(batch_size):
                sample[j, :, :, :] = (sample[j, :, :, :] - torch.min(sample[j, :, :, :])) / (
                    torch.max(sample[j, :, :, :] - torch.min(sample[j, :, :, :])))
            sample = sample * 255
            for j in range(sample.size(0)):
                # sample = sample * 255
                save_img_(sample.view(batch_size, 3, h, w)[j, :, :, :],
                          (path + '/decoder/{}.png').format(j + i * batch_size))
            sample = sample / 255
            if ebm and not args.dataset == 'celebahq':
                if ebm_z:
                    ebm_sample, z = sgld_z(model=energy_fn, img=sample, z=zs, decoder=decoder, steps=args.sample_step,
                                           step_size=args.step_size, show=False)
                    with torch.no_grad():
                        sample = decoder(z).detach()
                    nc, h, w = sample.shape[1:]
                    for j in range(batch_size):
                        sample[j, :, :, :] = (sample[j, :, :, :] - torch.min(sample[j, :, :, :])) / (
                            torch.max(sample[j, :, :, :] - torch.min(sample[j, :, :, :])))
                    sample = sample * 255
                    for j in range(sample.size(0)):
                        # sample = sample * 255
                        save_img_(sample.view(batch_size, 3, h, w)[j, :, :, :],
                                  (path + '/decoder/{}.png').format(j + i * batch_size))
                else:
                    ebm_sample = sgld(model=energy_fn, img=sample, steps=args.sample_step, step_size=args.step_size,
                                      show=False)
                for j in range(batch_size):
                    ebm_sample[j, :, :, :] = (ebm_sample[j, :, :, :] - torch.min(ebm_sample[j, :, :, :])) / (
                        torch.max(ebm_sample[j, :, :, :] - torch.min(ebm_sample[j, :, :, :])))
                ebm_sample = ebm_sample * 255
                for j in range(sample.size(0)):
                    save_img_(ebm_sample.view(batch_size, 3, h, w)[j, :, :, :],
                              (path + '/ebm/{}.png').format(j + i * batch_size))


import logging
import os


def get_logger(logpath, filepath, package_files=[], displaying=True, saving=True, debug=False):
    logger = logging.getLogger()
    if debug:
        level = logging.DEBUG
    else:
        level = logging.INFO
    logger.setLevel(level)
    if saving:
        info_file_handler = logging.FileHandler(logpath, mode="a")
        info_file_handler.setLevel(level)
        logger.addHandler(info_file_handler)
    if displaying:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        logger.addHandler(console_handler)

    # Write the file.
    logger.info(filepath)
    with open(filepath, "r") as f:
        logger.info(f.read())

    for f in package_files:
        logger.info(f)
        with open(f, "r") as package_f:
            logger.info(package_f.read())

    return logger


def makedirs(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)


def join_dir(path1, path2):
    res = os.path.join(path1, path2)
    return res
