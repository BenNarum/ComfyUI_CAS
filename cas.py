# The main sampler. It's a little messy right now.

# samplers.py contains the custom nodes for the SigmaWaveFormSampler node. The SigmaWaveFormSampler node is a custom node
from .noise_classes import *
from .noise_classes import prepare_noise, NOISE_GENERATOR_NAMES
import comfy.samplers
import comfy.sample
import comfy.sampler_helpers
from comfy import model_management, utils
from comfy.k_diffusion import sampling as k_diffusion_sampling
import node_helpers
import latent_preview
import torch
import math
from tqdm.auto import trange
import kornia
import functools
import matplotlib.pyplot as plt
import numpy as np
from comfy_extras.nodes_latent import reshape_latent_to
import os
import sys
import random
import node_helpers
import comfy.utils
from torch.nn import functional as F, MultiheadAttention, Linear

def initialize_or_scale(tensor, value, steps):
    if tensor is None:
        # Create the tensor on the same device as 'value' if it's a tensor
        device = value.device if torch.is_tensor(value) else 'cpu' 
        return torch.full((steps,), value, device=device) 
    else:
        return value * tensor
    
def move_to_same_device(*tensors):
    if not tensors:
        return tensors

    device = tensors[0].device
    return tuple(tensor.to(device) for tensor in tensors)

sys.path.insert(0, os.path.join(os.path.dirname(os.path.realpath(__file__)), "comfy"))

class CustomAdvancedSampler:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "guider": ("GUIDER", ),
                "sampler": ("SAMPLER", ),
                "sigmas": ("SIGMAS", ),
                "latent_image": ("LATENT", ),
                "add_noise": ("BOOLEAN", {"default": True}),
                "noise_is_latent": ("BOOLEAN", {"default": False}),
                "noise_type": (NOISE_GENERATOR_NAMES, ),
                "noise_seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "use_alpha": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("LATENT", "LATENT")
    RETURN_NAMES = ("output", "denoised_output")
    FUNCTION = "sample"
    CATEGORY = "sampling/custom_sampling"

    def sample(self, guider, sampler, sigmas, latent_image, use_alpha, add_noise, noise_is_latent, noise_type, noise_seed):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        latent = latent_image.copy()
        latent_image_samples = latent["samples"].to(device)
        latent_image_fixed = comfy.sample.fix_empty_latent_channels(guider.model_patcher, latent_image_samples)
        latent["samples"] = latent_image_fixed

        noise = None
        alpha_layer = None

        if not add_noise:
            torch.manual_seed(noise_seed)
            noise = torch.zeros(latent_image_fixed.size(), dtype=latent_image_fixed.dtype, layout=latent_image_fixed.layout, device=latent_image_fixed.device)
        else:
            print(f"Using default noise type: {noise_type}")
            noise = prepare_noise(latent_image_fixed, noise_seed, noise_type, use_alpha=use_alpha).to(latent_image_fixed.device)
            if use_alpha:
                noise, alpha_layer = noise[..., 0], noise[..., 1]

        if noise_is_latent:
            intensity_map = latent_image_fixed.abs()
            if use_alpha: 
                noise_channels = noise[..., :-1]
                alpha_layer = noise[..., -1:]
                noise_channels = reshape_latent_to(intensity_map.shape, noise_channels)
                noise = noise_channels * (0.5 + intensity_map)
                noise = torch.cat((noise, alpha_layer), dim=-1)
            else:
                noise = reshape_latent_to(intensity_map.shape, noise)
                noise = noise * (0.5 + intensity_map)
            noise_fft = torch.fft.fft2(noise)
            latent_fft = torch.fft.fft2(latent_image_fixed)
            combined_fft = noise_fft * latent_fft
            noise = torch.fft.ifft2(combined_fft).real
            noise = noise / noise.std(dim=(1, 2, 3), keepdim=True)

        noise = torch.nan_to_num(noise, nan=0.0, posinf=0.0, neginf=0.0)
        latent_image_fixed = torch.nan_to_num(latent_image_fixed, nan=0.0, posinf=0.0, neginf=0.0)

        if use_alpha and alpha_layer is not None:
            noise = noise * alpha_layer

        noise_mask = None
        if "noise_mask" in latent:
            noise_mask = latent["noise_mask"]

        x0_output = {}
        callback = latent_preview.prepare_callback(guider.model_patcher, sigmas.shape[-1] - 1, x0_output)

        disable_pbar = not utils.PROGRESS_BAR_ENABLED

        # Perform sampling with the guider, ensuring sigmas are used correctly
        samples = guider.sample(
            noise, latent_image_fixed, sampler, sigmas,
            denoise_mask=noise_mask, callback=callback, disable_pbar=disable_pbar, seed=noise_seed
        )

        samples = samples.to(model_management.intermediate_device())

        out = latent.copy()
        out["samples"] = samples
        if "x0" in x0_output:
            out_denoised = latent.copy()
            out_denoised["samples"] = guider.model_patcher.model.process_latent_out(x0_output["x0"].cpu())
        else:
            out_denoised = out

        return out, out_denoised
    
# Define the Guider_Basic class
class Guider_Basic(comfy.samplers.CFGGuider):
    def __init__(self, model):
        super().__init__(model)

    def set_conds(self, positive):
        # Only set the positive conditioning
        self.inner_set_conds({"positive": positive})

# Define the BasicGuider class
class BasicCFGGuider:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "conditioning": ("CONDITIONING",),
                "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0, "step": 0.1, "round": 0.01}),
            }
        }

    RETURN_TYPES = ("GUIDER",)

    FUNCTION = "get_guider"
    CATEGORY = "sampling/custom_sampling/guiders"

    def get_guider(self, model, conditioning, cfg):
        guider = Guider_Basic(model)
        guider.set_conds(conditioning)
        guider.set_cfg(cfg)
        return (guider,)

# these aren't working yet. It's a concept

class LatentSelfAttention:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "latent": ("LATENT",) }} 
    
    RETURN_TYPES = ("LATENT", "MASK") 
    RETURN_NAMES = ("latent", "attention_map")
    FUNCTION = "compute_attention"
    CATEGORY = "latent/attention" 

    def compute_attention(self, latent):
        # Extract the latent tensor and device
        x = latent["samples"]
        device = x.device

        # Reshape for multihead attention
        b, c, h, w = x.shape
        x = x.reshape(b, c, h * w).permute(0, 2, 1)  # (b, h*w, c)

        # Initialize the MultiheadAttention module 
        attention_layer = MultiheadAttention(embed_dim=c, num_heads=8).to(device)

        # Compute self-attention
        attn_output, attn_weights = attention_layer(x, x, x)

        # Reshape attn_output back to the original latent shape
        attn_output = attn_output.permute(0, 2, 1).reshape(b, c, h, w)
        latent_out = latent.copy()
        latent_out["samples"] = attn_output 

        # Process attention weights to get a spatial attention map 
        attention_map = attn_weights.mean(dim=1).reshape(b, h, w)

        return (latent_out, attention_map) 
    
class AttentionToSigmas:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { 
                              "model": ("MODEL",),
                              "latent": ("LATENT",),  # Take latent as input
                              "base_scheduler": (comfy.samplers.SCHEDULER_NAMES, ),
                              "scaling_factor": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.1}),  
                             }}

    RETURN_TYPES = ("SIGMAS",)
    FUNCTION = "get_sigmas"
    CATEGORY = "sampling/custom_sampling/schedulers"

    def get_sigmas(self, model, latent, base_scheduler, scaling_factor):
        # Get device, number of steps, and latent channels 
        model_sampling = model.get_model_object("model_sampling")
        device = model_sampling.sigmas.device
        steps = model_sampling.num_timesteps
        latent_channels = model.model.latent_format.latent_channels

        # Generate base sigmas
        base_sigmas = comfy.samplers.calculate_sigmas(model_sampling, base_scheduler, steps).to(device)

        # Calculate the adjusted dimension for MultiheadAttention
        num_heads = 8
        adjusted_dim = ((latent_channels + num_heads - 1) // num_heads) * num_heads

        # Linear layer to adjust dimensions 
        dim_adjustment_layer = Linear(latent_channels, adjusted_dim).to(device)

        # Initialize the MultiheadAttention module (using adjusted_dim)
        attention_layer = MultiheadAttention(embed_dim=adjusted_dim, num_heads=num_heads).to(device) 

        # Modify sigmas based on attention maps
        sigmas = []
        for i in range(steps):
            # Get the latent at this step using partial denoising
            current_latent = self.get_latent_at_step(model, latent, base_sigmas[i]) 

            # Reshape the latent and adjust dimensions for multihead attention
            b, c, h, w = current_latent.shape
            current_latent = current_latent.reshape(b, c, h * w).permute(0, 2, 1)  # (b, h*w, c)
            current_latent = dim_adjustment_layer(current_latent) # Apply linear layer

            # Compute self-attention
            attn_output, attn_weights = attention_layer(current_latent, current_latent, current_latent)

            # Reshape attn_output back to the original latent shape
            current_latent = attn_output.permute(0, 2, 1).reshape(b, adjusted_dim, h, w) 

            # Process attention weights to get a spatial attention map 
            attn_map = attn_weights.mean(dim=1).reshape(b, h, w)

            # Normalize attention map
            attn_map = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min())

            # Create weighting factor from attention map
            weighting_factor = 1.0 + attn_map * scaling_factor 

            # Scale the base sigma
            sigmas.append(base_sigmas[i] * weighting_factor)

        sigmas = torch.stack(sigmas, dim=0) 
        sigmas[-1] = 0.0 

        return (sigmas, )

      
    def get_latent_at_step(self, model, latent, sigma):
            # Get the noise tensor from CustomAdvancedSampler 
            noise = model.get_model_object("custom_advanced_sampler").noise

            # Get the noisy latent input
            latent_in = latent.copy()
            latent_in['samples'] = model.get_model_object("model_sampling").noise_scaling(sigma, noise, latent_in['samples'])
            empty_conds = {}

            # Perform one denoising step
            latent_out = model.apply_model(latent_in["samples"], sigma, **empty_conds)
            return latent_out  # Return the denoised latent at this step

    
