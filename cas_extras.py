# this needs refactoring. On the to-do-list 

import comfy.model_management
import torch
from torch import FloatTensor
from tqdm.auto import trange, tqdm
import math
import comfy.utils
import numpy as np
import comfy.k_diffusion.sampling
from comfy.k_diffusion.sampling import get_ancestral_step, to_d
import comfy.model_patcher
import torch.nn.functional as F
import comfy.model_sampling
import functools
from comfy.model_sampling import ModelSamplingDiscrete, EPS
from .noise_classes import *
#from comfy.samplers import CFGGuider, calc_cond_batch
import comfy.samplers
from comfy.utils import common_upscale
import math
from comfy.k_diffusion import sampling as k_diffusion_sampling, utils 
from comfy.ldm.modules.diffusionmodules.util import make_beta_schedule
from einops import rearrange, repeat
import nodes 
import node_helpers
from comfy.k_diffusion.utils import append_dims
import comfy

def initialize_or_scale(tensor, value, steps):
    if tensor is None:
        return torch.full((steps,), value)
    else:
        return value * tensor
    
def move_to_same_device(*tensors):
    if not tensors:
        return tensors

    device = tensors[0].device
    return tuple(tensor.to(device) for tensor in tensors)

def default_noise_sampler(x):
    return lambda sigma, sigma_next: torch.randn_like(x)

def sample_lcmcustom_gpu(model, x, sigmas, extra_args=None, callback=None, disable=None, noise_sampler=None):
    extra_args = {} if extra_args is None else extra_args
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    noise_sampler = default_noise_sampler(x) if noise_sampler is None else noise_sampler
    s_in = x.new_ones([x.shape[0]]).to(device)

    for i in trange(len(sigmas) - 1, disable=disable):
        denoised = model(x, sigmas[i] * s_in, **extra_args)
        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigmas[i], 'denoised': denoised})

        x = denoised
        if sigmas[i + 1] > 0:
            x = model.inner_model.inner_model.model_sampling.noise_scaling(sigmas[i + 1], noise_sampler(sigmas[i], sigmas[i + 1]), x)
    return x

class SamplerLCMCustomGPU:
    @classmethod
    def INPUT_TYPES(cls):
        from .noise_classes import NOISE_GENERATOR_NAMES
        return {
            "required": {
                "noise_sampler_type": (NOISE_GENERATOR_NAMES,)
            }
        }
    
    RETURN_TYPES = ("SAMPLER",)
    CATEGORY = "sampling/custom_sampling/samplers"
    FUNCTION = "get_sampler"

    def get_sampler(self, noise_sampler_type):
        sampler = comfy.samplers.KSAMPLER(sample_lcmcustom_gpu, extra_options={"noise_sampler_type": noise_sampler_type})
        return (sampler, )

def DDPMSampler_step_gpu(x, sigma, sigma_prev, noise, noise_sampler):
    device = x.device  # Get the device of the input tensor
    alpha_cumprod = 1 / ((sigma * sigma) + 1)
    alpha_cumprod_prev = 1 / ((sigma_prev * sigma_prev) + 1)
    alpha = (alpha_cumprod / alpha_cumprod_prev)

    mu = (1.0 / alpha).sqrt() * (x - (1 - alpha) * noise / (1 - alpha_cumprod).sqrt())
    if sigma_prev > 0:
        noise_sample = noise_sampler(sigma, sigma_prev).to(device)
        mu += ((1 - alpha) * (1. - alpha_cumprod_prev) / (1. - alpha_cumprod)).sqrt() * noise_sample
    return mu

def generic_step_sampler_gpu(model, x, sigmas, extra_args=None, callback=None, disable=None, noise_sampler=None, step_function=None):
    extra_args = {} if extra_args is None else extra_args
    noise_sampler = default_noise_sampler(x) if noise_sampler is None else noise_sampler
    s_in = x.new_ones([x.shape[0]]).to(x.device)  # Ensure s_in is on the same device as x

    for i in trange(len(sigmas) - 1, disable=disable):
        denoised = model(x, sigmas[i] * s_in, **extra_args)
        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigmas[i], 'denoised': denoised})
        x = step_function(x / torch.sqrt(1.0 + sigmas[i] ** 2.0).to(x.device), sigmas[i].to(x.device), sigmas[i + 1].to(x.device), (x - denoised) / sigmas[i].to(x.device), noise_sampler)
        if sigmas[i + 1] != 0:
            x *= torch.sqrt(1.0 + sigmas[i + 1] ** 2.0).to(x.device)
    return x

@torch.no_grad()
def sample_ddpm_gpu(model, x, sigmas, extra_args=None, callback=None, disable=None, noise_sampler=None):
    device = next(model.parameters()).device if hasattr(model, 'parameters') else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    x = x.to(device)
    sigmas = sigmas.to(device)
    return generic_step_sampler_gpu(model, x, sigmas, extra_args, callback, disable, noise_sampler, DDPMSampler_step_gpu)

@torch.no_grad()
def sample_lcm_upscaleW(model, x, sigmas, extra_args=None, callback=None, disable=None, total_upscale=2.0, upscale_method="bislerp", upscale_steps=None):
    extra_args = {} if extra_args is None else extra_args

    if upscale_steps is None:
        upscale_steps = max(len(sigmas) // 2 + 1, 2)
    else:
        upscale_steps += 1
        upscale_steps = min(upscale_steps, len(sigmas) + 1)

    upscales = np.linspace(1.0, total_upscale, upscale_steps)[1:]

    orig_shape = x.size()
    s_in = x.new_ones([x.shape[0]])
    for i in trange(len(sigmas) - 1, disable=disable):
        denoised = model(x, sigmas[i] * s_in, **extra_args)
        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigmas[i], 'denoised': denoised})

        x = denoised
        if i < len(upscales):
            x = comfy.utils.common_upscale(x, round(orig_shape[-1] * upscales[i]), round(orig_shape[-2] * upscales[i]), upscale_method, "disabled")

        if sigmas[i + 1] > 0:
            x += sigmas[i + 1] * torch.randn_like(x)
    return x

class SamplerLCMUpscaleW:
    upscale_methods = ["bislerp", "nearest-exact", "bilinear", "area", "bicubic"]

    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {"scale_ratio": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 20.0, "step": 0.01}),
                     "scale_steps": ("INT", {"default": -1, "min": -1, "max": 1000, "step": 1}),
                     "upscale_method": (s.upscale_methods,),
                      }
               }
    RETURN_TYPES = ("SAMPLER",)
    CATEGORY = "sampling/custom_sampling/samplers"

    FUNCTION = "get_sampler"

    def get_sampler(self, scale_ratio, scale_steps, upscale_method):
        if scale_steps < 0:
            scale_steps = None
        sampler = comfy.samplers.KSAMPLER(sample_lcm_upscaleW, extra_options={"total_upscale": scale_ratio, "upscale_steps": scale_steps, "upscale_method": upscale_method})
        return (sampler, )

def sample_lcm_upscaleW_gpu(model, x, sigmas, extra_args=None, callback=None, disable=None, total_upscale=2.0, upscale_method="nearest-exact", upscale_steps=None):
    extra_args = {} if extra_args is None else extra_args

    if upscale_steps is None:
        upscale_steps = max(len(sigmas) // 2 + 1, 2)
    else:
        upscale_steps += 1
        upscale_steps = min(upscale_steps, len(sigmas) + 1)

    upscales = np.linspace(1.0, total_upscale, upscale_steps)[1:]

    orig_shape = x.size()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    s_in = x.new_ones([x.shape[0]]).to(device)

    for i in trange(len(sigmas) - 1, disable=disable):
        denoised = model(x, sigmas[i] * s_in, **extra_args)
        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigmas[i], 'denoised': denoised})

        x = denoised
        if i < len(upscales):
            x = comfy.utils.common_upscale(x, round(orig_shape[-1] * upscales[i]), round(orig_shape[-2] * upscales[i]), upscale_method, "disabled")

        if sigmas[i + 1] > 0:
            x += sigmas[i + 1] * torch.randn_like(x)
    return x

class SamplerLCMUpscaleWGPU:
    upscale_methods = ["bislerp", "nearest-exact", "bilinear", "area", "bicubic"]

    @classmethod
    def INPUT_TYPES(cls):
        return {"required":
                    {"scale_ratio": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 20.0, "step": 0.01}),
                     "scale_steps": ("INT", {"default": -1, "min": -1, "max": 1000, "step": 1}),
                     "upscale_method": (cls.upscale_methods,),
                      }
               }
    RETURN_TYPES = ("SAMPLER",)
    CATEGORY = "sampling/custom_sampling/samplers"

    FUNCTION = "get_sampler"

    def get_sampler(self, scale_ratio, scale_steps, upscale_method):
        if scale_steps < 0:
            scale_steps = None
        sampler = comfy.samplers.KSAMPLER(sample_lcm_upscaleW_gpu, extra_options={"total_upscale": scale_ratio, "upscale_steps": scale_steps, "upscale_method": upscale_method})
        return (sampler, )
    
# David Löwenfels
@torch.no_grad()
def sample_tcd_euler_a_w(model, x, sigmas, extra_args=None, callback=None, disable=None, noise_sampler=None, gamma=0.3):
    extra_args = {} if extra_args is None else extra_args
    noise_sampler = default_noise_sampler(x) if noise_sampler is None else noise_sampler
    s_in = x.new_ones([x.shape[0]])
    for i in trange(len(sigmas) - 1, disable=disable):
        denoised = model(x, sigmas[i] * s_in, **extra_args)
        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigmas[i], 'denoised': denoised})

        d = to_d(x, sigmas[i], denoised)

        sigma_from = sigmas[i]
        sigma_to = sigmas[i + 1]

        t = model.inner_model.inner_model.model_sampling.timestep(sigma_from)
        down_t = (1 - gamma) * t
        sigma_down = model.inner_model.inner_model.model_sampling.sigma(down_t)

        if sigma_down > sigma_to:
            sigma_down = sigma_to

        sigma_up = (sigma_to ** 2 - sigma_down ** 2) ** 0.5
        
        d = to_d(x, sigma_from, denoised)
        dt = sigma_down - sigma_from
        x += d * dt

        if sigma_to > 0 and gamma > 0:
            x = model.inner_model.inner_model.model_sampling.noise_scaling(sigma_up, noise_sampler(sigma_from, sigma_to), x)
    return x

@torch.no_grad() # David Löwenfels
def sample_tcd_w(model, x, sigmas, extra_args=None, callback=None, disable=None, noise_sampler=None, gamma=0.3):
    extra_args = {} if extra_args is None else extra_args
    noise_sampler = default_noise_sampler(x) if noise_sampler is None else noise_sampler
    s_in = x.new_ones([x.shape[0]])
    
    for i in trange(len(sigmas) - 1, disable=disable):
        denoised = model(x, sigmas[i] * s_in, **extra_args)
        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigmas[i], 'denoised': denoised})

        sigma_from, sigma_to = sigmas[i], sigmas[i+1]

        t = model.inner_model.inner_model.model_sampling.timestep(sigma_from)
        t_s = (1 - gamma) * t
        sigma_to_s = model.inner_model.inner_model.model_sampling.sigma(t_s)

        # if sigma_to_s > sigma_to:
        #     sigma_to_s = sigma_to
        # if sigma_to_s < 0:
        #     sigma_to_s = torch.tensor(1.0)
        print(f"sigma_from: {sigma_from}, sigma_to: {sigma_to}, sigma_to_s: {sigma_to_s}")

        noise_est = (x - denoised) / sigma_from
        x /= torch.sqrt(1.0 + sigma_from ** 2.0)

        alpha_cumprod = 1 / ((sigma_from * sigma_from) + 1)
        alpha_cumprod_prev = 1 / ((sigma_to * sigma_to) + 1)
        alpha = (alpha_cumprod / alpha_cumprod_prev)

        x = (1.0 / alpha).sqrt() * (x - (1 - alpha) * noise_est / (1 - alpha_cumprod).sqrt())

        first_step = sigma_to == 0
        last_step = i == len(sigmas) - 2

        if not first_step:
            if gamma > 0 and not last_step:
                noise = noise_sampler(sigma_from, sigma_to)

                variance = ((1 - alpha_cumprod_prev) / (1 - alpha_cumprod)) * (1 - alpha_cumprod / alpha_cumprod_prev)
                x += variance.sqrt() * noise

            x *= torch.sqrt(1.0 + sigma_to ** 2.0)
    return x

# David Löwenfels
class SamplerTCD_w:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                     "gamma": ("FLOAT", {"default": 0.3, "min": 0, "max": 1.0, "step": 0.01, "round": False})
                    }
               }
    RETURN_TYPES = ("SAMPLER",)
    CATEGORY = "sampling/custom_sampling/samplers"

    FUNCTION = "get_sampler"

    def get_sampler(self, gamma):
        sampler = comfy.samplers.KSAMPLER(sample_tcd_w, extra_options={"gamma": gamma})
        return (sampler, )


# David Löwenfels
class SamplerTCDEulerA_w:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                     "gamma": ("FLOAT", {"default": 0.3, "min": 0, "max": 1.0, "step": 0.01, "round": False})
                    }
               }
    RETURN_TYPES = ("SAMPLER",)
    CATEGORY = "sampling/custom_sampling/samplers"

    FUNCTION = "get_sampler"

    def get_sampler(self, gamma):
        sampler = comfy.samplers.KSAMPLER(sample_tcd_euler_a_w, extra_options={"gamma": gamma})
        return (sampler, )
    
def DDPMSampler_step_gpu(x, sigma, sigma_prev, noise, noise_sampler):
    device = x.device  # Get the device of the input tensor
    alpha_cumprod = 1 / ((sigma * sigma) + 1)
    alpha_cumprod_prev = 1 / ((sigma_prev * sigma_prev) + 1)
    alpha = (alpha_cumprod / alpha_cumprod_prev)

    mu = (1.0 / alpha).sqrt() * (x - (1 - alpha) * noise / (1 - alpha_cumprod).sqrt())
    if sigma_prev > 0:
        noise_sample = noise_sampler(sigma, sigma_prev).to(device)
        mu += ((1 - alpha) * (1. - alpha_cumprod_prev) / (1. - alpha_cumprod)).sqrt() * noise_sample
    return mu

def generic_step_sampler_gpu(model, x, sigmas, extra_args=None, callback=None, disable=None, noise_sampler=None, step_function=None):
    extra_args = {} if extra_args is None else extra_args
    noise_sampler = default_noise_sampler(x) if noise_sampler is None else noise_sampler
    s_in = x.new_ones([x.shape[0]]).to(x.device)  # Ensure s_in is on the same device as x

    for i in trange(len(sigmas) - 1, disable=disable):
        denoised = model(x, sigmas[i] * s_in, **extra_args)
        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigmas[i], 'denoised': denoised})
        x = step_function(x / torch.sqrt(1.0 + sigmas[i] ** 2.0).to(x.device), sigmas[i].to(x.device), sigmas[i + 1].to(x.device), (x - denoised) / sigmas[i].to(x.device), noise_sampler)
        if sigmas[i + 1] != 0:
            x *= torch.sqrt(1.0 + sigmas[i + 1] ** 2.0).to(x.device)
    return x

@torch.no_grad()
def sample_ddpm_gpu(model, x, sigmas, extra_args=None, callback=None, disable=None, noise_sampler=None):
    device = next(model.parameters()).device if hasattr(model, 'parameters') else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    x = x.to(device)
    sigmas = sigmas.to(device)
    return generic_step_sampler_gpu(model, x, sigmas, extra_args, callback, disable, noise_sampler, DDPMSampler_step_gpu)

# --- Helper Functions --- 

def append_zero(x):
    return torch.cat([x, x.new_zeros([1])])

def get_ancestral_step(sigma_from, sigma_to, eta=1.):
    if not eta:
        return sigma_to, 0.
    sigma_up = min(sigma_to, eta * (sigma_to ** 2 * (sigma_from ** 2 - sigma_to ** 2) / sigma_from ** 2) ** 0.5)
    sigma_down = (sigma_to ** 2 - sigma_up ** 2) ** 0.5
    return sigma_down, sigma_up

# --- Custom `to_d` Functions ---

def to_d(x, sigma, denoised):  # Original
    return (x - denoised) / utils.append_dims(sigma, x.ndim)

def to_d_style_transfer(x, sigma, denoised_cond, style_latent, weight):
    """
    Calculate the derivative using a weighted average of the 
    conditional denoised output and the style latent.
    """
    return (x - (denoised_cond * weight + style_latent * (1 - weight))) / utils.append_dims(sigma, x.ndim) 


# --- Attention-Based Blurring Function ---
# comfy.model_patcher.
def create_blur_map(x0, attn, sigma=3.0, threshold=1.0):
    """
    Creates a blur map based on attention scores and applies Gaussian blur
    to selected regions of the image.
    """
    # Reshape attention map and apply threshold
    b, _, lh, lw = x0.shape
    ah = int(math.sqrt(attn.shape[1]))
    aw = int(math.sqrt(attn.shape[1]))
    attn = attn.reshape(b, -1, ah, aw)
    mask = attn.mean(1, keepdim=False).sum(1, keepdim=False) > threshold

    # Upscale the mask to the original image resolution
    mask = mask.unsqueeze(1).type(attn.dtype)
    mask = F.interpolate(mask, (lh, lw))

    # Apply gaussian blur
    blurred = gaussian_blur_2d(x0, kernel_size=9, sigma=sigma)

    # Combine blurred and original based on the mask
    blurred = blurred * mask + x0 * (1 - mask)
    return blurred

def gaussian_blur_2d(img, kernel_size, sigma):
    ksize_half = (kernel_size - 1) * 0.5
    x = torch.linspace(-ksize_half, ksize_half, steps=kernel_size, device=img.device, dtype=img.dtype)
    pdf = torch.exp(-0.5 * (x / sigma).pow(2))
    kernel1d = pdf / pdf.sum()
    kernel2d = torch.mm(kernel1d[:, None], kernel1d[None, :])
    kernel2d = kernel2d.expand(img.shape[-3], 1, kernel2d.shape[0], kernel2d.shape[1])
    padding = [kernel_size // 2, kernel_size // 2, kernel_size // 2, kernel_size // 2]
    img = F.pad(img, padding, mode="reflect")
    img = F.conv2d(img, kernel2d, groups=img.shape[-3])
    return img


@torch.no_grad()
def sample_custom_method(model, x, sigmas, extra_args=None, callback=None, disable=None):
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])

    for i in trange(len(sigmas) - 1, disable=disable):
        denoised = model(x, sigmas[i] * s_in, **extra_args)
        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigmas[i], 'denoised': denoised})

        x = denoised
        if sigmas[i + 1] > 0:
            x += sigmas[i + 1] * torch.randn_like(x)
    return x

def to_d_weighted(x, sigma, denoised_cond, denoised_uncond, weight):
    """
    Calculate the derivative using a weighted average.
    """
    return (x - (denoised_cond * weight + denoised_uncond * (1 - weight))) / utils.append_dims(sigma, x.ndim)

@torch.no_grad()
def sample_euler_weighted_cfg_pp(model, x, sigmas, weight, extra_args=None, callback=None, disable=None):
    extra_args = {} if extra_args is None else extra_args

    temp = [0]
    def post_cfg_function(args):
        temp[0] = args["uncond_denoised"]
        return args["denoised"]

    model_options = extra_args.get("model_options", {}).copy()
    extra_args["model_options"] = comfy.model_patcher.set_model_options_post_cfg_function(model_options, post_cfg_function, disable_cfg1_optimization=True)

    s_in = x.new_ones([x.shape[0]])
    for i in trange(len(sigmas) - 1, disable=disable):
        sigma_hat = sigmas[i]
        denoised = model(x, sigma_hat * s_in, **extra_args)
        d = to_d_weighted(x, sigma_hat, denoised, temp[0], weight)
        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigma_hat, 'denoised': denoised})
        dt = sigmas[i + 1] - sigma_hat
        x = x + d * dt
    return x

@torch.no_grad()
def sample_euler_dynamic_cfg_pp(model, x, sigmas, extra_args=None, callback=None, disable=None):
    extra_args = {} if extra_args is None else extra_args

    temp = [0]
    def post_cfg_function(args):
        temp[0] = args["uncond_denoised"]
        return args["denoised"]

    model_options = extra_args.get("model_options", {}).copy()
    extra_args["model_options"] = comfy.model_patcher.set_model_options_post_cfg_function(
        model_options, post_cfg_function, disable_cfg1_optimization=True
    )

    s_in = x.new_ones([x.shape[0]])
    for i in trange(len(sigmas) - 1, disable=disable):
        sigma_hat = sigmas[i]
        denoised = model(x, sigma_hat * s_in, ** extra_args)

        # Calculate dynamic weight
        weight = i / (len(sigmas) - 1)  # Linear increase 

        d = to_d_weighted(x, sigma_hat, denoised, temp[0], weight)  # Use weighted derivative

        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigma_hat, 'denoised': denoised})
        dt = sigmas[i + 1] - sigma_hat
        x = x + d * dt
    return x

@torch.no_grad()
def sample_euler_attn_cfg_pp(model, x, sigmas, extra_args=None, callback=None, disable=None, **extra_options):
    extra_args = {} if extra_args is None else extra_args
    cfg_scale = extra_options.get('cfg_scale', 1.0)
    blur_sigma = extra_options.get('blur_sigma', 3.0)
    blur_threshold = extra_options.get('blur_threshold', 1.0)
    temp = [0]
    uncond_attn = None

    def post_cfg_function(args):
        nonlocal uncond_attn, temp
        temp[0] = args["uncond_denoised"]
        uncond_attn = attn_scores  # Capture attn scores from uncond
        return args["denoised"]

    #  --- Modified attention_basic that returns attention scores ---
    from comfy.ldm.modules.attention import attention_basic 
    def attention_basic_with_sim(*args, **kwargs):
        global attn_scores
        out, attn_scores = attention_basic(*args, **kwargs) 
        return out, attn_scores

    model_options = extra_args.get("model_options", {}).copy()
    extra_args["model_options"] = comfy.model_patcher.set_model_options_post_cfg_function(
        model_options, post_cfg_function, disable_cfg1_optimization=True
    )

    model.attention_basic = attention_basic_with_sim  # Replace attention for this sampler 

    s_in = x.new_ones([x.shape[0]])
    for i in trange(len(sigmas) - 1, disable=disable):
        global attn_scores  
        attn_scores = None  
        sigma_hat = sigmas[i]
        denoised = model(x, sigma_hat * s_in, **extra_args)
        d = (x - denoised + cfg_scale * (denoised - temp[0])) / sigmas[i]

        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigma_hat, 'denoised': denoised})

        # --- Apply attention-based blurring ---
        if uncond_attn is not None:  
            denoised = create_blur_map(
                denoised, uncond_attn, blur_sigma, blur_threshold
            )

        x = denoised + d * sigmas[i + 1]
    return x

@torch.no_grad()
def sample_euler_step_control_cfg_pp(model, x, sigmas, extra_args=None, callback=None, disable=None):
    """
    CFG++ with adaptive step size control.
    """
    extra_args = {} if extra_args is None else extra_args
    temp = [0]

    def post_cfg_function(args):
        temp[0] = args["uncond_denoised"]
        return args["denoised"]

    model_options = extra_args.get("model_options", {}).copy()
    extra_args["model_options"] = comfy.model_patcher.set_model_options_post_cfg_function(
        model_options, post_cfg_function, disable_cfg1_optimization=True
    )

    s_in = x.new_ones([x.shape[0]])
    for i in trange(len(sigmas) - 1, disable=disable):
        sigma_hat = sigmas[i]
        denoised = model(x, sigma_hat * s_in, **extra_args)

        d = to_d(x, sigma_hat, temp[0])

        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigma_hat, 'denoised': denoised})
        dt = sigmas[i + 1] - sigma_hat

        # Calculate adaptive step size here based on ||denoised - temp[0]||
        # Example: Scale dt inversely proportional to the difference
        diff_norm = torch.norm(denoised - temp[0])
        dt = dt / (diff_norm + 1e-6)  # Avoid division by zero

        x = denoised + d * dt 
    return x

@torch.no_grad()
def sample_heun_cfg_pp(model, x, sigmas, extra_args=None, callback=None, disable=None):
    """
    Heun's method with CFG++.
    """
    extra_args = {} if extra_args is None else extra_args

    temp = [0]
    def post_cfg_function(args):
        temp[0] = args["uncond_denoised"]
        return args["denoised"]

    model_options = extra_args.get("model_options", {}).copy()
    extra_args["model_options"] = comfy.model_patcher.set_model_options_post_cfg_function(
        model_options, post_cfg_function, disable_cfg1_optimization=True
    )

    s_in = x.new_ones([x.shape[0]])
    for i in trange(len(sigmas) - 1, disable=disable):
        sigma_hat = sigmas[i]
        denoised = model(x, sigma_hat * s_in, **extra_args)
        d = to_d(x, sigma_hat, temp[0])  # CFG++
        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigma_hat, 'denoised': denoised})
        dt = sigmas[i + 1] - sigma_hat

        if sigmas[i + 1] == 0:
            # Euler method for the last step
            x = x + d * dt
        else:
            # Heun's method (with CFG++)
            x_2 = x + d * dt
            denoised_2 = model(x_2, sigmas[i + 1] * s_in, **extra_args) 
            d_2 = to_d(x_2, sigmas[i + 1], denoised_2)  # CFG++
            d_prime = (d + d_2) / 2
            x = x + d_prime * dt
    return x

@torch.no_grad()
def sample_dpm_solver_cfg_pp(model, x, sigmas, extra_args=None, callback=None, disable=None):
    extra_args = {} if extra_args is None else extra_args

    temp = [0]
    def post_cfg_function(args):
        temp[0] = args["uncond_denoised"]
        return args["denoised"]

    model_options = extra_args.get("model_options", {}).copy()
    extra_args["model_options"] = comfy.model_patcher.set_model_options_post_cfg_function(
        model_options, post_cfg_function, disable_cfg1_optimization=True
    )

    s_in = x.new_ones([x.shape[0]])
    for i in trange(len(sigmas) - 1, disable=disable):
        sigma_hat = sigmas[i]
        denoised = model(x, sigma_hat * s_in, **extra_args)
        
        # Implement DPM-Solver logic
        d = to_d(x - denoised + temp[0], sigmas[i], denoised)
        
        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigma_hat, 'denoised': denoised})
        
        dt = sigmas[i + 1] - sigma_hat
        x = x + d * dt
    return x

class CustomEPS(EPS):
    def calculate_denoised(self, sigma, model_output, model_input, cfg_scale=1.0):
        sigma = sigma.view(sigma.shape[:1] + (1,) * (model_output.ndim - 1))
        return model_input - model_output * sigma + cfg_scale * (model_input - model_output * sigma)

class CustomModelSamplingDiscrete(ModelSamplingDiscrete):
    def calculate_denoised(self, sigma, model_output, model_input, uncond_denoised, cfg_scale=1.0):
        return model_input - model_output * sigma + cfg_scale * (model_input - uncond_denoised)
    
# Define the custom sampling functions

@torch.no_grad()
def sample_custom_lcm_cfg_pp(model, x, sigmas, extra_args=None, callback=None, disable=None, **extra_options):
    extra_args = {} if extra_args is None else extra_args
    cfg_scale = extra_options.get('cfg_scale', 1.0)
    temp = [0]

    def post_cfg_function(args):
        temp[0] = args["uncond_denoised"]
        return args["denoised"]

    model_options = extra_args.get("model_options", {}).copy()
    extra_args["model_options"] = comfy.model_patcher.set_model_options_post_cfg_function(
        model_options, post_cfg_function, disable_cfg1_optimization=True
    )

    s_in = x.new_ones([x.shape[0]])
    for i in trange(len(sigmas) - 1, disable=disable):
        sigma_hat = sigmas[i]
        denoised = model(x, sigma_hat * s_in, **extra_args)

        # Apply CFG++ scale
        d = (x - denoised + cfg_scale * (denoised - temp[0])) / append_dims(sigma_hat, x.ndim)

        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigma_hat, 'denoised': denoised})

        dt = sigmas[i + 1] - sigma_hat
        x = x + d * dt

    return x

@torch.no_grad()
def sample_custom_x0_cfg_pp(model, x, sigmas, extra_args=None, callback=None, disable=None, **extra_options):
    extra_args = {} if extra_args is None else extra_args
    cfg_scale = extra_options.get('cfg_scale', 1.0)
    temp = [0]

    def post_cfg_function(args):
        temp[0] = args["uncond_denoised"]
        return args["denoised"]

    model_options = extra_args.get("model_options", {}).copy()
    extra_args["model_options"] = comfy.model_patcher.set_model_options_post_cfg_function(
        model_options, post_cfg_function, disable_cfg1_optimization=True
    )

    s_in = x.new_ones([x.shape[0]])
    for i in range(len(sigmas) - 1):
        sigma_hat = sigmas[i]
        denoised = model(x, sigma_hat * s_in, **extra_args)

        d = denoised + cfg_scale * (denoised - temp[0])

        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigma_hat, 'denoised': denoised})

        dt = sigmas[i + 1] - sigma_hat
        x = x + d * dt
    return x

@torch.no_grad()
def sample_custom_model_sampling_discrete_distilled_cfg_pp(model, x, sigmas, extra_args=None, callback=None, disable=None, **extra_options):
    extra_args = {} if extra_args is None else extra_args
    cfg_scale = extra_options.get('cfg_scale', 1.0)
    temp = [0]

    def post_cfg_function(args):
        temp[0] = args["uncond_denoised"]
        return args["denoised"]

    model_options = extra_args.get("model_options", {}).copy()
    extra_args["model_options"] = comfy.model_patcher.set_model_options_post_cfg_function(
        model_options, post_cfg_function, disable_cfg1_optimization=True
    )

    s_in = x.new_ones([x.shape[0]])
    for i in trange(len(sigmas) - 1, disable=disable):
        sigma_hat = sigmas[i]
        denoised = model(x, sigma_hat * s_in, **extra_args)

        # Calculate denoised output
        sigma_hat_expanded = sigma_hat.view(sigma_hat.shape[:1] + (1,) * (denoised.ndim - 1))
        denoised_output = x - denoised * sigma_hat_expanded

        # Apply CFG++ scale
        d = denoised_output + cfg_scale * (denoised_output - temp[0])

        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigma_hat, 'denoised': denoised})

        dt = sigmas[i + 1] - sigma_hat
        x = denoised + d * dt
    return x

class SamplerWeightedCFGPP:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "weight": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01, "round": False}),
            }
        }

    RETURN_TYPES = ("SAMPLER",)
    CATEGORY = "sampling/custom_sampling/samplers"

    FUNCTION = "get_sampler"

    def get_sampler(self, weight):
        sampler = comfy.samplers.KSAMPLER(sample_euler_weighted_cfg_pp, extra_options={"weight": weight})
        return (sampler, )

class SamplerDynamicCFGPP:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {}}
    
    RETURN_TYPES = ("SAMPLER",)
    CATEGORY = "sampling/custom_sampling/samplers"

    FUNCTION = "get_sampler"

    def get_sampler(self):
        sampler = comfy.samplers.KSAMPLER(sample_euler_dynamic_cfg_pp)
        return (sampler, )

class SamplerEulerAttnCFGPP:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"cfg_scale": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.1}),
                             "blur_sigma": ("FLOAT", {"default": 3.0, "min": 0.1, "max": 10.0, "step": 0.1}),
                             "blur_threshold": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 10.0, "step": 0.1})}}

    RETURN_TYPES = ("SAMPLER",)
    CATEGORY = "sampling/custom_sampling/samplers"

    FUNCTION = "get_sampler"

    def get_sampler(self, cfg_scale, blur_sigma, blur_threshold):
        sampler = comfy.samplers.KSAMPLER(sample_euler_attn_cfg_pp, extra_options={"cfg_scale": cfg_scale, "blur_sigma": blur_sigma, "blur_threshold": blur_threshold})
        return (sampler, )

class SamplerStepSizeControlCFGPP:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {}}

    RETURN_TYPES = ("SAMPLER",)
    CATEGORY = "sampling/custom_sampling/samplers"

    FUNCTION = "get_sampler"

    def get_sampler(self):
        sampler = comfy.samplers.KSAMPLER(sample_euler_step_control_cfg_pp)
        return (sampler, )

class SamplerHeunCFGPP:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {}}

    RETURN_TYPES = ("SAMPLER",)
    CATEGORY = "sampling/custom_sampling/samplers"

    FUNCTION = "get_sampler"

    def get_sampler(self):
        sampler = comfy.samplers.KSAMPLER(sample_heun_cfg_pp)
        return (sampler, )

class SamplerDPMCFGPP:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {}}

    RETURN_TYPES = ("SAMPLER",)
    CATEGORY = "sampling/custom_sampling/samplers"

    FUNCTION = "get_sampler"

    def get_sampler(self):
        sampler = comfy.samplers.KSAMPLER(sample_dpm_solver_cfg_pp)
        return (sampler, )

# Define the nodes for the custom samplers
class SamplerCustomLCMCFGPP:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"cfg_scale": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.1})}}

    RETURN_TYPES = ("SAMPLER",)
    CATEGORY = "sampling/custom_sampling/samplers"

    FUNCTION = "get_sampler"

    def get_sampler(self, cfg_scale):
        sampler = comfy.samplers.KSAMPLER(sample_custom_lcm_cfg_pp, extra_options={"cfg_scale": cfg_scale})
        return (sampler, )

class SamplerCustomX0CFGPP:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"cfg_scale": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.1})}}

    RETURN_TYPES = ("SAMPLER",)
    CATEGORY = "sampling/custom_sampling/samplers"

    FUNCTION = "get_sampler"

    def get_sampler(self, cfg_scale):
        sampler = comfy.samplers.KSAMPLER(sample_custom_x0_cfg_pp, extra_options={"cfg_scale": cfg_scale})
        return (sampler, )

class SamplerCustomModelSamplingDiscreteDistilledCFGPP:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"cfg_scale": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.1})}}

    RETURN_TYPES = ("SAMPLER",)
    CATEGORY = "sampling/custom_sampling/samplers"

    FUNCTION = "get_sampler"

    def get_sampler(self, cfg_scale):
        sampler = comfy.samplers.KSAMPLER(sample_custom_model_sampling_discrete_distilled_cfg_pp, extra_options={"cfg_scale": cfg_scale})
        return (sampler, )
    
@torch.no_grad()
def sample_euler_step_control_ancestral_cfg_pp(model, x, sigmas, extra_args=None, callback=None, disable=None, **extra_options):
    extra_args = {} if extra_args is None else extra_args
    cfg_scale = extra_options.get('cfg_scale', 1.0)
    eta = extra_options.get('eta', 1.0)
    s_noise = extra_options.get('s_noise', 1.0)
    temp = [0]

    def post_cfg_function(args):
        temp[0] = args["uncond_denoised"]
        return args["denoised"]

    model_options = extra_args.get("model_options", {}).copy()
    extra_args["model_options"] = comfy.model_patcher.set_model_options_post_cfg_function(
        model_options, post_cfg_function, disable_cfg1_optimization=True
    )

    s_in = x.new_ones([x.shape[0]])
    for i in trange(len(sigmas) - 1, disable=disable):
        sigma_hat = sigmas[i]
        denoised = model(x, sigma_hat * s_in, **extra_args)

        # Apply CFG++ scale
        d = (x - denoised + cfg_scale * (denoised - temp[0])) / sigmas[i]

        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigma_hat, 'denoised': denoised})

        # Calculate adaptive step size here based on ||denoised - temp[0]||
        diff_norm = torch.norm(denoised - temp[0])
        dt = (sigmas[i + 1] - sigma_hat) / (diff_norm + 1e-6)  # Avoid division by zero

        x = denoised + d * dt
        x = x + eta * torch.randn_like(x) * sigmas[i + 1] * s_noise  # Apply eta and s_noise
    return x

class SamplerEulerStepControlAncestralCFGPP:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"cfg_scale": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.1}),
                             "eta": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.1}),
                             "s_noise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.1})}}

    RETURN_TYPES = ("SAMPLER",)
    CATEGORY = "sampling/custom_sampling/samplers"

    FUNCTION = "get_sampler"

    def get_sampler(self, cfg_scale, eta, s_noise):
        sampler = comfy.samplers.KSAMPLER(sample_euler_step_control_ancestral_cfg_pp, extra_options={"cfg_scale": cfg_scale, "eta": eta, "s_noise": s_noise})
        return (sampler, )


@torch.no_grad()
def sample_custom_model_sampling_discrete_distilled_ancestral_cfg_pp(model, x, sigmas, extra_args=None, callback=None, disable=None, **extra_options):
    extra_args = {} if extra_args is None else extra_args
    cfg_scale = extra_options.get('cfg_scale', 1.0)
    eta = extra_options.get('eta', 1.0)
    s_noise = extra_options.get('s_noise', 1.0)
    temp = [0]

    def post_cfg_function(args):
        temp[0] = args["uncond_denoised"]
        return args["denoised"]

    model_options = extra_args.get("model_options", {}).copy()
    extra_args["model_options"] = comfy.model_patcher.set_model_options_post_cfg_function(
        model_options, post_cfg_function, disable_cfg1_optimization=True
    )

    s_in = x.new_ones([x.shape[0]])
    for i in trange(len(sigmas) - 1, disable=disable):
        sigma_hat = sigmas[i]
        denoised = model(x, sigma_hat * s_in, **extra_args)

        # Calculate denoised output
        sigma_hat_expanded = sigma_hat.view(sigma_hat.shape[:1] + (1,) * (denoised.ndim - 1))
        denoised_output = x - denoised * sigma_hat_expanded

        # Apply CFG++ scale
        d = denoised_output + cfg_scale * (denoised_output - temp[0])

        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigma_hat, 'denoised': denoised})

        dt = sigmas[i + 1] - sigma_hat
        x = denoised + d * dt
        x = x + eta * torch.randn_like(x) * sigmas[i + 1] * s_noise  # Apply eta and s_noise
    return x

class SamplerCustomModelSamplingDiscreteDistilledAncestralCFGPP:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"cfg_scale": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.1}),
                             "eta": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.1}),
                             "s_noise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.1})}}

    RETURN_TYPES = ("SAMPLER",)
    CATEGORY = "sampling/custom_sampling/samplers"

    FUNCTION = "get_sampler"

    def get_sampler(self, cfg_scale, eta, s_noise):
        sampler = comfy.samplers.KSAMPLER(sample_custom_model_sampling_discrete_distilled_ancestral_cfg_pp, extra_options={"cfg_scale": cfg_scale, "eta": eta, "s_noise": s_noise})
        return (sampler, )



@torch.no_grad()
def sample_custom_x0_ancestral_cfg_pp(model, x, sigmas, extra_args=None, callback=None, disable=None, **extra_options):
    extra_args = {} if extra_args is None else extra_args
    cfg_scale = extra_options.get('cfg_scale', 1.0)
    eta = extra_options.get('eta', 1.0)
    s_noise = extra_options.get('s_noise', 1.0)
    temp = [0]

    def post_cfg_function(args):
        temp[0] = args["uncond_denoised"]
        return args["denoised"]

    model_options = extra_args.get("model_options", {}).copy()
    extra_args["model_options"] = comfy.model_patcher.set_model_options_post_cfg_function(
        model_options, post_cfg_function, disable_cfg1_optimization=True
    )

    s_in = x.new_ones([x.shape[0]])
    for i in trange(len(sigmas) - 1, disable=disable):
        sigma_hat = sigmas[i]
        denoised = model(x, sigma_hat * s_in, **extra_args)

        # Apply CFG++ scale
        d = (denoised + cfg_scale * (denoised - temp[0])) / sigmas[i]

        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigma_hat, 'denoised': denoised})

        dt = sigmas[i + 1] - sigma_hat
        x = x + d * dt
        x = x + eta * torch.randn_like(x) * sigmas[i + 1] * s_noise  # Apply eta and s_noise
    return x

class SamplerCustomX0AncestralCFGPP:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"cfg_scale": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.1}),
                             "eta": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.1}),
                             "s_noise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.1})}}

    RETURN_TYPES = ("SAMPLER",)
    CATEGORY = "sampling/custom_sampling/samplers"

    FUNCTION = "get_sampler"

    def get_sampler(self, cfg_scale, eta, s_noise):
        sampler = comfy.samplers.KSAMPLER(sample_custom_x0_ancestral_cfg_pp, extra_options={"cfg_scale": cfg_scale, "eta": eta, "s_noise": s_noise})
        return (sampler, )



@torch.no_grad()
def sample_lcm_upscaleW_cfgpp(model, x, sigmas, extra_args=None, callback=None, disable=None, total_upscale=2.0, upscale_method="bislerp", upscale_steps=None, **extra_options):
    extra_args = {} if extra_args is None else extra_args
    cfg_scale = extra_options.get('cfg_scale', 1.0)
    temp = [0]

    def post_cfg_function(args):
        temp[0] = args["uncond_denoised"]
        return args["denoised"]

    model_options = extra_args.get("model_options", {}).copy()
    extra_args["model_options"] = comfy.model_patcher.set_model_options_post_cfg_function(
        model_options, post_cfg_function, disable_cfg1_optimization=True
    )

    if upscale_steps is None:
        upscale_steps = max(len(sigmas) // 2 + 1, 2)
    else:
        upscale_steps += 1
        upscale_steps = min(upscale_steps, len(sigmas) + 1)

    upscales = np.linspace(1.0, total_upscale, upscale_steps)[1:]

    orig_shape = x.size()
    s_in = x.new_ones([x.shape[0]])
    for i in trange(len(sigmas) - 1, disable=disable):
        denoised = model(x, sigmas[i] * s_in, **extra_args)

        # Apply CFG++ scale
        d = denoised + cfg_scale * (denoised - temp[0])

        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigmas[i], 'denoised': d})

        x = d
        if i < len(upscales):
            new_size = (round(orig_shape[-2] * upscales[i]), round(orig_shape[-1] * upscales[i]))
            if upscale_method == "linear":
                x = torch.nn.functional.interpolate(x, size=new_size, mode="bilinear", align_corners=False)
            elif upscale_method == "trilinear":
                x = x.unsqueeze(2)
                x = torch.nn.functional.interpolate(x, size=(1, new_size[0], new_size[1]), mode="trilinear", align_corners=False)
                x = x.squeeze(2)
            else:
                x = common_upscale(x, new_size[1], new_size[0], upscale_method, "disabled")

        if sigmas[i + 1] > 0:
            x += sigmas[i + 1] * torch.randn_like(x)
    return x

class SamplerLCMUpscaleWCFGPP:
    upscale_methods = ["bislerp", "nearest-exact", "bilinear", "area", "bicubic", "linear", "trilinear"]

    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {"cfg_scale": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.1}),
                     "scale_ratio": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 20.0, "step": 0.01}),
                     "scale_steps": ("INT", {"default": -1, "min": -1, "max": 1000, "step": 1}),
                     "upscale_method": (s.upscale_methods,),
                      }
               }
    RETURN_TYPES = ("SAMPLER",)
    CATEGORY = "sampling/custom_sampling/samplers"

    FUNCTION = "get_sampler"

    def get_sampler(self, cfg_scale, scale_ratio, scale_steps, upscale_method):
        if scale_steps < 0:
            scale_steps = None
        sampler = comfy.samplers.KSAMPLER(sample_lcm_upscaleW_cfgpp, extra_options={"cfg_scale": cfg_scale, "total_upscale": scale_ratio, "upscale_steps": scale_steps, "upscale_method": upscale_method})
        return (sampler, )
    
class EPSCFGPPScheduler:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "num_timesteps": ("INT", {"default": 10, "min": 1, "max": 10000, "step": 1}),
                    "beta_schedule": (["linear", "cosine"], {"default": "linear"}),
                    "linear_start": ("FLOAT", {"default": 0.00085, "min": 0.00001, "max": 1.0, "step": 0.00001}),
                    "linear_end": ("FLOAT", {"default": 0.012, "min": 0.00001, "max": 1.0, "step": 0.00001}),
                    "cosine_s": ("FLOAT", {"default": 8e-3, "min": 0.00001, "max": 1.0, "step": 0.00001}),
                }}

    RETURN_TYPES = ("SIGMAS",)
    CATEGORY = "sampling/custom_sampling/sigmas"

    FUNCTION = "get_sigmas"

    def get_sigmas(self, num_timesteps, beta_schedule, linear_start, linear_end, cosine_s):
        sigmas = self._generate_sigmas(num_timesteps, beta_schedule, linear_start, linear_end, cosine_s)
        return (sigmas, )

    def _generate_sigmas(self, num_timesteps, beta_schedule, linear_start, linear_end, cosine_s):
        if beta_schedule == "linear":
            betas = self._make_linear_beta_schedule(num_timesteps, linear_start, linear_end)
        elif beta_schedule == "cosine":
            betas = self._make_cosine_beta_schedule(num_timesteps, cosine_s)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        sigmas = ((1.0 - alphas_cumprod) / alphas_cumprod).sqrt()
        sigmas = torch.flip(sigmas, dims=[0])
        return sigmas

    def _make_linear_beta_schedule(self, num_timesteps, start, end):
        return torch.linspace(start, end, num_timesteps)

    def _make_cosine_beta_schedule(self, num_timesteps, s):
        steps = num_timesteps + 1
        x = torch.linspace(0, num_timesteps, steps)
        alphas_cumprod = (torch.cos(((x / num_timesteps) + s) / (1 + s) * torch.tensor(math.pi * 0.5)) ** 2) / (torch.cos((s / (1 + s)) * torch.tensor(math.pi * 0.5)) ** 2)
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1.0 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return betas.clamp(0, 0.999)
    
# The following function adds the samplers during initialization, in __init__.py
def add_samplers():
    from comfy.samplers import KSampler, k_diffusion_sampling
    if hasattr(KSampler, "DISCARD_PENULTIMATE_SIGMA_SAMPLERS"):
        KSampler.DISCARD_PENULTIMATE_SIGMA_SAMPLERS |= discard_penultimate_sigma_samplers
    added = 0
    for sampler in extra_samplers:
        if sampler not in KSampler.SAMPLERS:
            try:
                idx = KSampler.SAMPLERS.index("uni_pc_bh2")  # Last item in the samplers list
                KSampler.SAMPLERS.insert(idx + 1, sampler)  # Add our custom samplers
                setattr(k_diffusion_sampling, "sample_{}".format(sampler), extra_samplers[sampler])
                added += 1
            except ValueError as _err:
                pass
    if added > 0:
        import importlib
        importlib.reload(k_diffusion_sampling)

# The following function adds the samplers during initialization, in __init__.py
def add_schedulers():
    from comfy.samplers import KSampler, k_diffusion_sampling
    added = 0
    for scheduler in extra_schedulers: #getattr(self, "sample_{}".format(extra_samplers))
        if scheduler not in KSampler.SCHEDULERS:
            try:
                idx = KSampler.SCHEDULERS.index("ddim_uniform") # Last item in the samplers list
                KSampler.SCHEDULERS.insert(idx+1, scheduler) # Add our custom samplers
                setattr(k_diffusion_sampling, "get_sigmas_{}".format(scheduler), extra_schedulers[scheduler])
                added += 1
            except ValueError as err:
                pass
    if added > 0:
        import importlib
        importlib.reload(k_diffusion_sampling)


# Add any extra samplers to the following dictionary

extra_samplers = {
    "ddpm_gpu": sample_ddpm_gpu,
    "SamplerLCMUpscaleW": sample_lcm_upscaleW,
    "SamplerLCMUpscaleWGPU": sample_lcm_upscaleW_gpu,
    "tcd_w": sample_tcd_w,  # Add TCD sampler
    "tcd_euler_a_w": sample_tcd_euler_a_w,  # Add TCD Euler A sampler
    "SamplerWeightedCFGPP": sample_euler_weighted_cfg_pp,
    "SamplerDynamicCFGPP": sample_euler_dynamic_cfg_pp,
    "SamplerEulerAttnCFGPP": sample_euler_attn_cfg_pp,
    "SamplerStepSizeControlCFGPP": sample_euler_step_control_cfg_pp,
    "SamplerHeunCFGPP": sample_heun_cfg_pp,
    "SamplerDPMCFGPP": sample_dpm_solver_cfg_pp,
    "SamplerCMCFGPP": sample_custom_lcm_cfg_pp,
    "SamplerX0CFGPP": sample_custom_x0_cfg_pp,
    "SamplerCustomModelSamplingDiscreteDistilledCFGPP": sample_custom_model_sampling_discrete_distilled_cfg_pp,
    "SamplerLCMUpscaleWCFGPP": sample_lcm_upscaleW_cfgpp,
    "SamplerEulerStepControlAncestralCFGPP": sample_euler_step_control_ancestral_cfg_pp,
    "SamplerCustomModelSamplingDiscreteDistilledAncestralCFGPP": sample_custom_model_sampling_discrete_distilled_ancestral_cfg_pp,
    "SamplerCustomX0AncestralCFGPP": sample_custom_x0_ancestral_cfg_pp,
}

# Add any extra schedulers to the following dictionary
discard_penultimate_sigma_samplers = set(())

extra_schedulers = {}

NODE_CLASS_MAPPINGS = {
    "SamplerLCMUpscaleW": SamplerLCMUpscaleW,
    "SamplerLCMUpscaleWGPU": SamplerLCMUpscaleWGPU,
    "tcd_w": SamplerTCD_w,
    "tcd_euler_a_w": SamplerTCDEulerA_w,
    "SamplerWeightedCFGPP": SamplerWeightedCFGPP,
    "SamplerDynamicCFGPP": SamplerDynamicCFGPP,
    "SamplerEulerAttnCFGPP": SamplerEulerAttnCFGPP,
    "SamplerStepSizeControlCFGPP": SamplerStepSizeControlCFGPP,
    "SamplerHeunCFGPP": SamplerHeunCFGPP,
    "SamplerDPMCFGPP": SamplerDPMCFGPP,
    "CustomLCMCFGPP": SamplerCustomLCMCFGPP,
    "CustomX0CFGPP": SamplerCustomX0CFGPP,
    "CustomModelSamplingDiscreteDistilledCFGPP": SamplerCustomModelSamplingDiscreteDistilledCFGPP,
    "SamplerLCMUpscaleWCFGPP": SamplerLCMUpscaleWCFGPP,
    "SamplerEulerStepControlAncestralCFGPP": SamplerEulerStepControlAncestralCFGPP,
    "CustomModelSamplingDiscreteDistilledAncestralCFGPP": SamplerCustomModelSamplingDiscreteDistilledAncestralCFGPP,
    "CustomX0AncestralCFGPP": SamplerCustomX0AncestralCFGPP,
    
}    
