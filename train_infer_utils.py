import torch
from tqdm import tqdm
from torch.nn import functional as F

import utils
from consts import Consts
from config import Config

def get_noisy_image(x_start, step, noise=None):
    if noise == None:
        noise = torch.randn_like(x_start)

    sqrt_alphas_cumprod_t = utils.extract(Consts.sqrt_alphas_cumprod, step, x_start.shape)
    sqrt_one_minus_alphas_cumprod_t = utils.extract(Consts.sqrt_one_minus_alphas_cumprod, step, x_start.shape)

    return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

def get_denoised_image(x, step, model):
    betas_t = utils.extract(Consts.betas, step, x.shape)
    sqrt_one_minus_alphas_cumprod_t = utils.extract(Consts.sqrt_one_minus_alphas_cumprod, step, x.shape)
    sqrt_recip_alphas_t = utils.extract(Consts.sqrt_recip_alphas, step, x.shape)

    model_mean = sqrt_recip_alphas_t * (x - betas_t  * model(x, step) / sqrt_one_minus_alphas_cumprod_t)

    if step[0] == 0:
        return model_mean
    else:
        noise = torch.randn_like(x)
        posterior_variance_t = utils.extract(Consts.posterior_variance, step, x.shape)
        return model_mean + noise * Consts.posterior_variance
    
def get_denoised_images(shape, timesteps, model):
    device = next(model.parameters()).device
    b_size = shape[0]

    img = torch.randn(shape, device=device)
    imgs = []

    for i in tqdm(reversed(range(0, timesteps)), desc='sampling loop time step', total=timesteps):
        img = get_denoised_image(img, torch.full((b_size,), i, device=device, dtype=torch.long), model)
        imgs.append(img.cpu().numpy())
    return imgs

def sample(model, img_size, batch_size=16, channels=3):
    return get_denoised_images((batch_size, channels, img_size, img_size), Config.timesteps, model)

def diff_loss(x_start, step, model, loss_type='huber'):
    noise = torch.randn_like(x_start)

    noisy_img = get_noisy_image(x_start, step, noise=noise)
    # print(noisy_img.shape)
    predicted_noise = model(noisy_img, step)

    if loss_type == 'l1':
        loss = F.l1_loss(noise, predicted_noise)
    elif loss_type == 'l2':
        loss = F.mse_loss(noise, predicted_noise)
    elif loss_type == 'huber':
        loss = F.smooth_l1_loss(noise, predicted_noise)
    else:
        raise NotImplementedError()

    return loss