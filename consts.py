import torch
from torch.nn import functional as F

from config import Config


class Consts(object):
    def beta_schedule(timesteps):
        return torch.linspace(Config.beta_start, Config.beta_end, timesteps)
    
    betas = beta_schedule(Config.timesteps)
    alphas = 1.0 - betas

    alphas_cumprod = torch.cumprod(alphas, axis=0)
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)
    alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

    sqrt_recip_alphas = torch.sqrt(1 / alphas)

    posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)