import os


class args:
    num_latent = 128
    dataset = 'stl'
    batch_size = 128
    fc_dim = 256
    device = 'cuda'
    dim_h = 128
    use_tanh = 1
    image_size = 32
    eps = 0.05
    use_sn = True  # optional, not necessary


# os.environ["CUDA_VISIBLE_DEVICES"] = f"{args.gpu_id}"
import copy
import torch
from torch import optim
from torch.nn import MSELoss
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm
import pytorch_fid_wrapper as pfw
import numpy as np

from models.ebms import ResNet, MSResNet
from utils import requires_grad, clip_grad, sample_data, sgld, save_img, \
    get_logger, join_dir, apply_sn
from models.fastae import fast_vae_32
import time
from utils import dual_step


# TODO: 1. Save checkpoint 2. EMA 3. argparse
def train():
    model = fast_vae_32(args).to(args.device)
    root = os.path.join('./data/', 'stl10')
    dataset = datasets.STL10(root=root, split='train+unlabeled', download=False,
                             transform=transforms.Compose([
                                 transforms.Resize(32),
                                 transforms.ToTensor(),
                                 transforms.RandomHorizontalFlip(),
                                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                             ]))
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    loader = tqdm(enumerate(sample_data(loader)))

    # Hyper-parameters
    args.step_size = step_size = 1e-6
    args.tau = tau = 1e-5
    args.alpha = alpha = 1
    args.sample_step = sample_step = 15
    args.model_name = model_name = "Energy-Calibrated_VAE"
    local_time = time.strftime("%Y-%m-%d-%H-%M", time.localtime())
    args.folder_name = folder_name = f"./experiment/{args.dataset}/{model_name}_samples_StepSize{step_size}_Steps{sample_step}_Tau{tau}_Alpha{alpha}_{local_time}"
    os.makedirs(args.folder_name, exist_ok=True)
    logger = get_logger(logpath=os.path.join(args.folder_name, "logs"), filepath=os.path.abspath(__file__))
    # layers = [(256, 1), (256, 1), (512, 1), (512, 1)]
    layers = [(128, 2), (256, 2), (256, 2), (256, 2)]
    # energy_fn = ResNet(layers).cuda()
    energy_fn = MSResNet(layers, tau=args.tau).cuda()
    if args.use_sn:
        energy_fn = apply_sn(energy_fn)
    energy_fn.sampling_strategy = 'mcmc'
    parameters = energy_fn.parameters()
    optimizer_vae = optim.Adam(model.parameters(), lr=2e-4, betas=(0., 0.9))
    optimizer = optim.Adam(parameters, lr=1e-4, betas=(0., 0.9))
    mseloss = MSELoss(reduction='sum')
    logger.info(args)
    logger.info(energy_fn)

    model.train()
    energy_fn.train()
    fid_best = 999
    lam = 30  # initialize the dual variable lambda
    beta = 0.1
    for i, (pos_img, _) in loader:
        pos_img = pos_img.to(args.device)
        # Learning and Calibrating VAE
        optimizer_vae.zero_grad()
        recon_img, z, mu, logvar = model(pos_img)
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / recon_img.shape[0]
        recon_loss = mseloss(recon_img, pos_img) / recon_img.shape[0]
        elbo = recon_loss + beta * kl_loss  # Loss from Beta-VAE
        noise = torch.randn([pos_img.shape[0], 128]).cuda()
        samples = model.decode(noise)  # sampling from VAE.
        requires_grad(parameters, False)
        calibrated_samples = sgld(model=energy_fn, img=samples.detach(), condition_img=samples.detach(),
                                  steps=sample_step, step_size=step_size)
        calibration_loss = mseloss(calibrated_samples.detach(), samples) / calibrated_samples.shape[0]
        final_loss = lam * calibration_loss + elbo
        final_loss.backward()
        optimizer_vae.step()

        # Learning EBMs.
        optimizer.zero_grad()
        requires_grad(parameters, True)
        with torch.no_grad():
            samples = model.decode(noise)  # sampling from VAE.
        neg_img = sgld(model=energy_fn, img=samples.detach(), condition_img=samples.detach(), steps=sample_step,
                       step_size=step_size)
        pos_out = energy_fn(pos_img.detach())
        neg_out = energy_fn(neg_img.detach())
        ebm_loss = pos_out - neg_out
        # loss += alpha * (pos_out ** 2 + neg_out ** 2) # optional
        ebm_loss = ebm_loss.mean()
        ebm_loss.backward()
        optimizer.step()

        loader.set_description(
            f'Iteration: {i + 1}, recon_loss: {recon_loss.item():.3f}, kl_loss: {kl_loss.item():.3f}, calibration: {calibration_loss.item():.3f},real energy: {pos_out.mean().item() * args.tau:.3f}, '
            f'fake energy: {neg_out.mean().item() * args.tau:.5f}, lam: {lam:.4f}')
        # dual step
        eps = args.eps
        lam = dual_step(lam, calibration_loss, eps=eps, lr=1e-2)
        if i % 1000 == 0:
            save_img(samples.detach().cpu(), folder=join_dir(folder_name, "vis"),
                     file_name=f'samples_{str(i).zfill(5)}.png')


if __name__ == '__main__':
    train()
