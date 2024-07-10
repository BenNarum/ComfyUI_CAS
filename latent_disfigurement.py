# A bunch of math nodes than will directly alter latent tensors. Will work on proper documentation

import torch
import random
import comfy.utils
import comfy.model_management
import numpy as np
import torch.nn.functional as F

import logging
import random 
import nodes
import json


MAX_RESOLUTION = 16384

# ----------------------------------------------------------------------------
# Convolution Node
# ----------------------------------------------------------------------------

class LatentConvolution:    
    PRESET_KERNELS = {
        "Edge Detection": [[1, 1, 1], [1, -8, 1], [1, 1, 1]],
        "Sharpen": [[0, -1, 0], [-1, 5, -1], [0, -1, 0]],
        "Box Blur": [[1/9, 1/9, 1/9], [1/9, 1/9, 1/9], [1/9, 1/9, 1/9]],
        "Gaussian Blur": [[1/16, 2/16, 1/16], [2/16, 4/16, 2/16], [1/16, 2/16, 1/16]],
        "Sobel X": [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
        "Sobel Y": [[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
        "Emboss": [[-2, -1, 0], [-1, 1, 1], [0, 1, 2]],
        "Outline": [[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]],
        "Identity": [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
        "Laplacian": [[0, -1, 0], [-1, 4, -1], [0, -1, 0]],
        "Motion Blur Horizontal": [[1/9, 1/9, 1/9], [1/9, 1/9, 1/9], [1/9, 1/9, 1/9]],
        "Motion Blur Vertical": [[1/9, 1/9, 1/9], [1/9, 1/9, 1/9], [1/9, 1/9, 1/9]],
        "Prewitt X": [[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]],
        "Prewitt Y": [[-1, -1, -1], [0, 0, 0], [1, 1, 1]],
        "Ridge Detection": [[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]],
        "High Pass Filter": [[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]],
        "Low Pass Filter": [[1/16, 2/16, 1/16], [2/16, 4/16, 2/16], [1/16, 2/16, 1/16]],
        "Diagonal Edge Detection": [[2, -1, -1], [-1, 2, -1], [-1, -1, 2]]
    }

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": { 
                "latent": ("LATENT",),
                "use_preset": ("BOOLEAN", {"default": True}),
                "kernel_preset": (list(s.PRESET_KERNELS.keys()),),
                "custom_kernel": ("STRING", {
                    "multiline": True, 
                    "default": "[[1, 1, 1],\n [1, -8, 1],\n [1, 1, 1]]"  # Example edge detection kernel
                }),
                "stride": ("INT", {"default": 1, "min": 1, "max": 5}),
                "padding": ("INT", {"default": 1, "min": 0, "max": 10}),
            }
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "convolve"
    CATEGORY = "latent/custom"

    def convolve(self, latent, use_preset, kernel_preset, custom_kernel, stride, padding):
        device = comfy.model_management.get_torch_device()
        latent_tensor = latent["samples"]

        # Ensure the latent tensor is on the correct device
        latent_tensor = latent_tensor.to(device)

        # Select the kernel based on the toggle and choice
        if use_preset:
            kernel = self.PRESET_KERNELS.get(kernel_preset)
        else:
            try:
                kernel = json.loads(custom_kernel)
            except Exception as e:
                logging.warning(f"Error parsing custom kernel: {e}")
                return (latent,)

        # Convert the kernel to a PyTorch tensor
        kernel_tensor = torch.tensor(kernel, dtype=torch.float32, device=device)

        # Ensure the kernel is 2D and has odd dimensions
        if kernel_tensor.ndim != 2 or kernel_tensor.shape[0] % 2 == 0 or kernel_tensor.shape[1] % 2 == 0:
            logging.warning("Invalid kernel. Kernel must be 2D with odd dimensions.")
            return (latent,) 

        # Reshape for convolution
        kernel_tensor = kernel_tensor.view(1, 1, *kernel_tensor.shape).repeat(latent_tensor.shape[1], 1, 1, 1)

        # Apply convolution with user-defined stride and padding
        convolved_latent = F.conv2d(latent_tensor, kernel_tensor, stride=(stride, stride), padding=(padding, padding), groups=latent_tensor.shape[1])

        return ({"samples": convolved_latent}, ) 

# ----------------------------------------------------------------------------
# Activation Node
# ----------------------------------------------------------------------------

MAX_RESOLUTION = 16384 

class LatentActivation:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { 
                              "samples": ("LATENT",),
                              "activation": (["ReLU", "Sigmoid", "Tanh", "LeakyReLU", "ELU", "Softplus", "Swish", "PReLU", "GELU", "SELU", "Mish"],),
                              "strength": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                              "leaky_relu_negative_slope": ("FLOAT", {"default": 0.01, "min": 0.0, "max": 1.0, "step": 0.01}),
                              "elu_alpha": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.1}),
                              "softplus_beta": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 10.0, "step": 0.1}),
                              "softplus_threshold": ("FLOAT", {"default": 20.0, "min": 1.0, "max": 100.0, "step": 1.0}),
                              "swish_beta": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 10.0, "step": 0.1}),
                              "prelu_init": ("FLOAT", {"default": 0.25, "min": 0.0, "max": 1.0, "step": 0.01}),
                              "add_to_original": ("BOOLEAN", {"default": True}),
                              "normalize": ("BOOLEAN", {"default": False}),
                              "clamp": ("BOOLEAN", {"default": False}),
                              "clamp_min": ("FLOAT", {"default": -3.0, "min": -10.0, "max": 0.0, "step": 0.1}),
                              "clamp_max": ("FLOAT", {"default": 3.0, "min": 0.0, "max": 10.0, "step": 0.1}),
                              "composite": ("BOOLEAN", {"default": False}),
                              "blend_amount": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                              }}

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "apply_activation"

    CATEGORY = "latent/experimental"

    def apply_activation(self, samples, activation, strength, leaky_relu_negative_slope, elu_alpha,
                         softplus_beta, softplus_threshold, swish_beta, prelu_init,
                         add_to_original, normalize, clamp, clamp_min, clamp_max, composite, blend_amount):
        latent = samples["samples"]

        if activation == "ReLU":
            transformed_latent = torch.relu(latent)
        elif activation == "Sigmoid":
            transformed_latent = torch.sigmoid(latent)
        elif activation == "Tanh":
            transformed_latent = torch.tanh(latent)
        elif activation == "LeakyReLU":
            transformed_latent = torch.nn.functional.leaky_relu(latent, negative_slope=leaky_relu_negative_slope)
        elif activation == "ELU":
            transformed_latent = torch.nn.functional.elu(latent, alpha=elu_alpha)
        elif activation == "Softplus":
            transformed_latent = torch.nn.functional.softplus(latent, beta=softplus_beta, threshold=softplus_threshold)
        elif activation == "Swish":
            transformed_latent = latent * torch.sigmoid(swish_beta * latent)
        elif activation == "PReLU":
            prelu = torch.nn.PReLU(init=prelu_init)
            transformed_latent = prelu(latent)
        elif activation == "GELU":
            transformed_latent = torch.nn.functional.gelu(latent)
        elif activation == "SELU":
            transformed_latent = torch.nn.functional.selu(latent)
        elif activation == "Mish":
            transformed_latent = latent * torch.tanh(F.softplus(latent))
        else:
            raise ValueError(f"Invalid activation function: {activation}")

        if normalize:
            transformed_latent = (transformed_latent - transformed_latent.mean()) / transformed_latent.std()

        if clamp:
            transformed_latent = torch.clamp(transformed_latent, min=clamp_min, max=clamp_max)

        samples_out = samples.copy()
        if add_to_original:
            output_latent = latent + strength * transformed_latent
        else:
            output_latent = transformed_latent

        if composite:
            # Correctly upscale the latent tensor using F.interpolate for correct spatial dimensions
            latent_height, latent_width = latent.shape[2], latent.shape[3]
            upscaled_latent = F.interpolate(latent, size=(latent_height, latent_width), mode='nearest')

            # Ensure the dimensions of output_latent match those of latent
            if output_latent.shape[2:] != latent.shape[2:]:
                output_latent = F.interpolate(output_latent, size=(latent_height, latent_width), mode='nearest')

            # Blend the tensors
            samples_out["samples"] = upscaled_latent * (1 - blend_amount) + output_latent * blend_amount
        else:
            samples_out["samples"] = output_latent

        return (samples_out, )

# ----------------------------------------------------------------------------
# Math Node
# ----------------------------------------------------------------------------

# Define LatentMath
class LatentMath:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
                    "code": ("STRING", {"multiline": True, "default": "result = latent * 2.0"}),
                  },
                "optional": {
                    "latent": ("LATENT", ),
                    "latent_shape": ("LATENT", ),
                    "preset": ("LATENT_MATH_FORMULA", ), 
                }}
    RETURN_TYPES = ("LATENT",)
    FUNCTION = "apply_code"
    CATEGORY = "latent/custom"

    def apply_code(self, code, latent=None, preset=None, latent_shape=None):
        device = comfy.model_management.get_torch_device()

        if latent is not None:
            latent_tensor = latent["samples"].to(device)
        elif latent_shape is not None:
            latent_tensor = torch.zeros_like(latent_shape["samples"], device=device)
        else:
            logging.warning("Either 'latent' or 'latent_shape' must be provided.")
            return (None,)

        local_dict = {'latent': latent_tensor, 'torch': torch}

        if preset is not None:
            code = preset["formula"]
            variables = preset.get("variables", "") 

            try:
                exec(variables, globals(), local_dict)
            except Exception as e:
                logging.warning(f"Error executing preset variables: {e}")
                return (latent,)

        try:
            exec(code, globals(), local_dict)
            result = local_dict.get("result", None)

            if not isinstance(result, torch.Tensor):
                raise ValueError("Code must return a PyTorch tensor named 'result'.")
            if result.shape != latent_tensor.shape:
                raise ValueError("Result tensor shape must match the latent shape.")

            return ({"samples": result},)
        except Exception as e:
            logging.warning(f"Error executing code: {e}")
            return (latent,)

class LatentWarpPresets:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"preset": (["swirl", "pinch", "horizontal_shift", "vertical_flip", "random_tile"], ),
                              "params": ("STRING", {"multiline": True, "default": ""}), # Empty by default!
                             }}
    RETURN_TYPES = ("LATENT_MATH_FORMULA",)
    FUNCTION = "get_preset"
    CATEGORY = "latent/custom/presets"

    def get_preset(self, preset, params):
        local_dict = {} 
        for line in params.splitlines():
            if line.startswith("set params:"): 
                try:
                    exec(line[len("set params:"):].strip(), globals(), local_dict) 
                except Exception as e:
                    logging.warning(f"Error evaluating parameter: {line.strip()} - {e}")

        if preset == "swirl":
            code = f"""
grid_x, grid_y = torch.meshgrid(torch.linspace(-1, 1, latent.shape[3], device=latent.device), 
                                torch.linspace(-1, 1, latent.shape[2], device=latent.device), indexing='ij')
dist = torch.sqrt(grid_x**2 + grid_y**2)
angle = dist * {local_dict.get('swirl_strength', 5.0)} * 3.14159  
warped_x = grid_x * torch.cos(angle) - grid_y * torch.sin(angle)
warped_y = grid_x * torch.sin(angle) + grid_y * torch.cos(angle)
grid = torch.stack((warped_x, warped_y), dim=-1).unsqueeze(0).repeat(latent.shape[0], 1, 1, 1)
result = torch.nn.functional.grid_sample(latent, grid, mode='bilinear', padding_mode='zeros', align_corners=False)
            """ 
        elif preset == "pinch":
            code = f"""
grid_x, grid_y = torch.meshgrid(torch.linspace(-1, 1, latent.shape[3], device=latent.device), 
                                torch.linspace(-1, 1, latent.shape[2], device=latent.device), indexing='ij')
dist = torch.sqrt(grid_x**2 + grid_y**2)
pinch_factor = 1.0 + (1.0 - dist) * {local_dict.get('pinch_strength', 0.5)} 
warped_x = grid_x * pinch_factor
warped_y = grid_y * pinch_factor
grid = torch.stack((warped_x, warped_y), dim=-1).unsqueeze(0).repeat(latent.shape[0], 1, 1, 1)
result = torch.nn.functional.grid_sample(latent, grid, mode='bilinear', padding_mode='zeros', align_corners=False)
            """  
        elif preset == "horizontal_shift":
            code = f"result = torch.roll(latent, shifts=(0, {int(local_dict.get('amount', 8))}), dims=(2, 3))" 
        elif preset == "vertical_flip":
            code = "result = torch.flip(latent, dims=[2])" 
        elif preset == "random_tile":
            code = f"""
tile_size = {int(local_dict.get('tile_size', 16))} 
tiles = latent.chunk(latent.shape[2] // tile_size, dim=2)
random.shuffle(tiles)
result = torch.cat(tiles, dim=2) 
            """ 
        else:
            code = "" 

        return ({"formula": code, "variables": ""}, )  # Return only the dictionary


class LatentChannelPresets:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"preset": (["channel_swap", "rgb_to_bgr", "channel_scale"], ),
                              "params": ("STRING", {"multiline": True, "default": ""}),  
                             }}

    RETURN_TYPES = ("LATENT_MATH_FORMULA",)
    FUNCTION = "get_preset"
    CATEGORY = "latent/custom/presets"

    def get_preset(self, preset, params):
        local_dict = {} 
        for line in params.splitlines():
            if line.startswith("set params:"): 
                try:
                    exec(line[len("set params:"):].strip(), globals(), local_dict) 
                except Exception as e:
                    logging.warning(f"Error evaluating parameter: {line.strip()} - {e}")

        if preset == "channel_swap":
            code = "result = latent.index_select(1, torch.tensor([1, 0, 2, 3], device=latent.device))"  
        elif preset == "rgb_to_bgr":
            code = "result = latent.index_select(1, torch.tensor([2, 1, 0, 3], device=latent.device))"  
        elif preset == "channel_scale":
            code = f"""
result = latent.clone()
result[:, 0] *= {local_dict.get('red_scale', 1.5)}  
result[:, 1] *= {local_dict.get('green_scale', 0.75)}  
result[:, 2] *= {local_dict.get('blue_scale', 1.0)}  
            """
        else:
            code = "" 

        return ({"formula": code, "variables": ""}, )  


class LatentValuePresets:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "preset": (["square", "sqrt", "exp", "invert", "clamp", "normalise", "logarithmic transformation"], ),
                              "params": ("STRING", {"multiline": True, "default": ""}),  
                             }}

    RETURN_TYPES = ("LATENT_MATH_FORMULA",)
    FUNCTION = "get_preset"
    CATEGORY = "latent/custom/presets"

    def get_preset(self, preset, params):
        local_dict = {} 
        for line in params.splitlines():
            if line.startswith("set params:"): 
                try:
                    exec(line[len("set params:"):].strip(), globals(), local_dict) 
                except Exception as e:
                    logging.warning(f"Error evaluating parameter: {line.strip()} - {e}")

        if preset == "square":
            code = f"result = torch.pow(latent, {local_dict.get('exponent', 2.0)})"   
        elif preset == "sqrt":
            code = f"result = torch.pow(torch.abs(latent), 1.0 / {local_dict.get('root', 2.0)}) * torch.sign(latent)"
        elif preset == "exp":
            code = f"result = torch.exp(latent * {local_dict.get('strength', 1.0)}) - 1.0"  
        elif preset == "invert": 
            code = "result = 1.0 - latent"  
        elif preset == "clamp":
            code = f"result = torch.clamp(latent, {local_dict.get('min_value', -0.5)}, {local_dict.get('max_value', 0.5)})"
        elif preset == "normalize":
            code = f"result = (latent - latent.min()) / (latent.max() - latent.min())"
        elif preset == "logarithmic transformation":
            code = f"result = torch.log(torch.abs(latent) + {local_dict.get('base', 1.0)}) * torch.sign(latent)"
       
        else:
            code = "" 

        return ({"formula": code, "variables": ""}, )  


class LatentFrequencyPresets:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "preset": (["high_pass", "low_pass", "high_boost", "low_cut"], ),
                              "params": ("STRING", {"multiline": True, "default": ""}), 
                             }}

    RETURN_TYPES = ("LATENT_MATH_FORMULA",)
    FUNCTION = "get_preset"
    CATEGORY = "latent/custom/presets"

    def get_preset(self, preset, params):
        local_dict = {} 
        for line in params.splitlines():
            if line.startswith("set params:"): 
                try:
                    exec(line[len("set params:"):].strip(), globals(), local_dict) 
                except Exception as e:
                    logging.warning(f"Error evaluating parameter: {line.strip()} - {e}")

        if preset == "high_pass":
            code = f"""
fft_result = torch.fft.rfft2(latent, dim=(-2, -1), norm='ortho')
magnitudes = fft_result.abs()
high_pass_filter = 1 - torch.exp(-((torch.arange(magnitudes.shape[2])[:, None] / magnitudes.shape[2])**2 + (torch.arange(magnitudes.shape[3])[None, :] / magnitudes.shape[3])**2) * {local_dict.get('strength', 5.0)})
high_pass_filter = high_pass_filter.to(latent.device)
magnitudes *= (1.0 + high_pass_filter.unsqueeze(0).unsqueeze(0))
fft_result.real = magnitudes * torch.cos(fft_result.angle())
fft_result.imag = magnitudes * torch.sin(fft_result.angle())
result = torch.fft.irfft2(fft_result, dim=(-2, -1), norm='ortho')
            """ 
        elif preset == "low_pass":
            code = f"""
fft_result = torch.fft.rfft2(latent, dim=(-2, -1), norm='ortho')
magnitudes = fft_result.abs()
low_pass_filter = torch.exp(-((torch.arange(magnitudes.shape[2])[:, None] / magnitudes.shape[2])**2 + (torch.arange(magnitudes.shape[3])[None, :] / magnitudes.shape[3])**2) * {local_dict.get('strength', 5.0)})
low_pass_filter = low_pass_filter.to(latent.device)
magnitudes *= low_pass_filter.unsqueeze(0).unsqueeze(0)
fft_result.real = magnitudes * torch.cos(fft_result.angle())
fft_result.imag = magnitudes * torch.sin(fft_result.angle())
result = torch.fft.irfft2(fft_result, dim=(-2, -1), norm='ortho') 
            """
        elif preset == "high_boost":
            code = f"""
fft_result = torch.fft.rfft2(latent, dim=(-2, -1), norm='ortho')
magnitudes = fft_result.abs()
high_pass_filter = 1 - torch.exp(-((torch.arange(magnitudes.shape[2])[:, None] / magnitudes.shape[2])**2 + (torch.arange(magnitudes.shape[3])[None, :] / magnitudes.shape[3])**2) * {local_dict.get('strength', 5.0)})
high_pass_filter = high_pass_filter.to(latent.device)
magnitudes *= (1.0 + high_pass_filter.unsqueeze(0).unsqueeze(0))
fft_result.real = magnitudes * torch.cos(fft_result.angle())
fft_result.imag = magnitudes * torch.sin(fft_result.angle())
high_passed = torch.fft.irfft2(fft_result, dim=(-2, -1), norm='ortho')
result = latent * (1.0 - {local_dict.get('cutoff', 0.25)}) + high_passed * {local_dict.get('cutoff', 0.25)}
            """
        elif preset == "low_cut":
            code = f"""
fft_result = torch.fft.rfft2(latent, dim=(-2, -1), norm='ortho')
magnitudes = fft_result.abs()
low_pass_filter = torch.exp(-((torch.arange(magnitudes.shape[2])[:, None] / magnitudes.shape[2])**2 + (torch.arange(magnitudes.shape[3])[None, :] / magnitudes.shape[3])**2) * {local_dict.get('strength', 5.0)})
low_pass_filter = low_pass_filter.to(latent.device)
magnitudes *= low_pass_filter.unsqueeze(0).unsqueeze(0)
fft_result.real = magnitudes * torch.cos(fft_result.angle())
fft_result.imag = magnitudes * torch.sin(fft_result.angle())
low_passed = torch.fft.irfft2(fft_result, dim=(-2, -1), norm='ortho')
result = latent - low_passed * {local_dict.get('cutoff', 0.25)} 
            """
        else:
            code = "" 

        return ({"formula": code, "variables": ""}, )  

# ----------------------------------------------------------------------------
# Latent Noise Presets 
# ----------------------------------------------------------------------------

import torch
import comfy.utils
import logging
import pywt
import random
import opensimplex

class LatentNoisePresets:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"preset": (["fractal", "pyramid", "gaussian", "uniform", "brownian",
                                          "laplacian", "studentt", "wavelet", "perlin", "voronoi", "poisson", "opensimplex",
                                          "white", "pink", "brown", "salt_and_pepper"], ),
                              "scale": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.1}),
                              "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                              "params": ("STRING", {"multiline": True, "default": ""}),  # Empty by default
                             }}
    RETURN_TYPES = ("LATENT_MATH_FORMULA",)
    FUNCTION = "get_preset"
    CATEGORY = "latent/custom/presets"

    def get_preset(self, preset, scale, seed, params):
        local_dict = {} 
        for line in params.splitlines():
            if line.startswith("set params:"): 
                try:
                    exec(line[len("set params:"):].strip(), globals(), local_dict) 
                except Exception as e:
                    logging.warning(f"Error evaluating parameter: {line.strip()} - {e}")

        # --- Formula Generation based on Preset ---

        if preset == "fractal":
            code = f"""
# Use latent_shape if latent is None
if latent is None and latent_shape is not None:
    y_freq = torch.fft.fftfreq(latent_shape.shape[2], d=1.0/latent_shape.shape[2], device=latent_shape.device)
    x_freq = torch.fft.fftfreq(latent_shape.shape[3], d=1.0/latent_shape.shape[3], device=latent_shape.device)
    freq = torch.sqrt(y_freq[:, None]**2 + x_freq[None, :]**2).clamp(min=1e-10)
    spectral_density = {local_dict.get('k', 1.0)} / torch.pow(freq, {local_dict.get('alpha', 0.0)} * {local_dict.get('scale', 0.1)})
    spectral_density[0, 0] = 0
    noise = torch.normal(mean=0.0, std=1.0, size=latent_shape.shape, dtype=latent_shape.dtype, device=latent_shape.device, generator=torch.manual_seed({seed}))
    noise_fft = torch.fft.fft2(noise)
    modified_fft = noise_fft * spectral_density.unsqueeze(0).unsqueeze(0)
    noise = torch.fft.ifft2(modified_fft).real
    result = latent_shape + (noise / torch.std(noise)) * {scale}
elif latent is not None:
    y_freq = torch.fft.fftfreq(latent.shape[2], d=1.0/latent.shape[2], device=latent.device)
    x_freq = torch.fft.fftfreq(latent.shape[3], d=1.0/latent.shape[3], device=latent.device)
    freq = torch.sqrt(y_freq[:, None]**2 + x_freq[None, :]**2).clamp(min=1e-10)
    spectral_density = {local_dict.get('k', 1.0)} / torch.pow(freq, {local_dict.get('alpha', 0.0)} * {local_dict.get('scale', 0.1)})
    spectral_density[0, 0] = 0
    noise = torch.normal(mean=0.0, std=1.0, size=latent.shape, dtype=latent.dtype, device=latent.device, generator=torch.manual_seed({seed}))
    noise_fft = torch.fft.fft2(noise)
    modified_fft = noise_fft * spectral_density.unsqueeze(0).unsqueeze(0)
    noise = torch.fft.ifft2(modified_fft).real
    result = latent + (noise / torch.std(noise)) * {scale}
else:
    raise ValueError("Either 'latent' or 'latent_shape' must be provided.")
            """
        elif preset == "pyramid":
            code = f"""
if latent is None:
    result = comfy.utils.pyramid_noise_like(latent_shape.shape, latent_shape.dtype, latent_shape.layout, generator=torch.manual_seed({seed}), device=latent_shape.device) * {scale}
else:
    result = comfy.utils.pyramid_noise_like(latent.shape, latent.dtype, latent.layout, generator=torch.manual_seed({seed}), device=latent.device) * {scale}
            """ 
        elif preset == "gaussian":
            code = f"result = latent + torch.randn_like(latent) * {scale}" 
        elif preset == "uniform":
            code = f"result = latent + (torch.rand_like(latent) - 0.5) * 2 * {scale}"  
        elif preset == "brownian":
            code = f"""
if latent is None:
    timesteps = torch.linspace(0, 1, latent_shape.shape[-1], device=latent_shape.device)
    dt = timesteps[1] - timesteps[0]
    result = torch.cumsum(torch.randn_like(latent_shape) * dt, dim=-1) * {scale} 
else:
    timesteps = torch.linspace(0, 1, latent.shape[-1], device=latent.device)
    dt = timesteps[1] - timesteps[0]
    result = torch.cumsum(torch.randn_like(latent) * dt, dim=-1) * {scale} 
            """ 
        elif preset == "laplacian":
            code = f"result = latent + torch.distributions.laplace.Laplace(0, {local_dict.get('scale', 1.0)}).sample(latent.shape).to(latent.device) * {scale}" 
        elif preset == "studentt":
            code = f"result = latent + torch.distributions.studentT.StudentT({local_dict.get('df', 1.0)}).sample(latent.shape).to(latent.device) * {scale}"
        elif preset == "wavelet": 
            code = f"""
import pywt

if latent is None:
    coeffs = pywt.dwt2(torch.zeros(latent_shape.shape, dtype=torch.float32, device='cpu').numpy(), '{local_dict.get('wavelet', 'db4')}')
    cA, (cH, cV, cD) = coeffs
    # ... (You can manipulate wavelet coefficients here)
    noise = torch.from_numpy(pywt.idwt2(coeffs, '{local_dict.get('wavelet', 'db4')}')).to(latent_shape.device) 
    result = latent_shape + (noise / noise.std()) * {scale} 
else:
    coeffs = pywt.dwt2(latent.cpu().numpy(), '{local_dict.get('wavelet', 'db4')}')
    cA, (cH, cV, cD) = coeffs
    # ... (You can manipulate wavelet coefficients here)
    noise = torch.from_numpy(pywt.idwt2(coeffs, '{local_dict.get('wavelet', 'db4')}')).to(latent.device)
    result = latent + (noise / noise.std()) * {scale}
            """
        elif preset == "perlin":
            # ... (Add your PyTorch-based Perlin noise implementation)
            code = f"""
def perlin_noise(shape, scale=1.0, device='cpu'):
    # ... (Your PyTorch code here)
    return noise_tensor

if latent is None:
    result = latent_shape + perlin_noise(latent_shape.shape, scale={local_dict.get('perlin_scale', 1.0)}, device=latent.device) * {scale}
else:
    result = latent + perlin_noise(latent.shape, scale={local_dict.get('perlin_scale', 1.0)}, device=latent.device) * {scale} 
            """
        elif preset == "voronoi":
            code = f"""
cells = {local_dict.get('cells', 16)} 
if latent is None:
    cell_points = torch.rand(cells, 2, device=latent_shape.device) * torch.tensor([latent_shape.shape[2], latent_shape.shape[3]], device=latent_shape.device)
    grid_x, grid_y = torch.meshgrid(torch.arange(latent_shape.shape[3], device=latent_shape.device), 
                                    torch.arange(latent_shape.shape[2], device=latent_shape.device), indexing='ij')
    grid = torch.stack((grid_y, grid_x), dim=-1).unsqueeze(0).repeat(cells, 1, 1, 1) 
    distances = torch.cdist(grid.view(cells, -1, 2), cell_points.unsqueeze(1))
    closest_cell_indices = torch.argmin(distances, dim=0).view(latent_shape.shape[2], latent_shape.shape[3])
    cell_values = torch.rand(cells, device=latent_shape.device)
    result = latent_shape + cell_values[closest_cell_indices].unsqueeze(0).unsqueeze(0).repeat(latent_shape.shape[0], latent_shape.shape[1], 1, 1) * {scale} 
else:
    cell_points = torch.rand(cells, 2, device=latent.device) * torch.tensor([latent.shape[2], latent.shape[3]], device=latent.device)
    grid_x, grid_y = torch.meshgrid(torch.arange(latent.shape[3], device=latent.device), 
                                    torch.arange(latent.shape[2], device=latent.device), indexing='ij')
    grid = torch.stack((grid_y, grid_x), dim=-1).unsqueeze(0).repeat(cells, 1, 1, 1) 
    distances = torch.cdist(grid.view(cells, -1, 2), cell_points.unsqueeze(1))
    closest_cell_indices = torch.argmin(distances, dim=0).view(latent.shape[2], latent.shape[3])
    cell_values = torch.rand(cells, device=latent.device)
    result = latent + cell_values[closest_cell_indices].unsqueeze(0).unsqueeze(0).repeat(latent.shape[0], latent.shape[1], 1, 1) * {scale}
            """
        elif preset == "poisson":
            code = f"result = latent + torch.poisson(torch.ones_like(latent) * {local_dict.get('lambda', 1.0)}) * {scale}"
        elif preset == "opensimplex":
            code = f"""
import opensimplex            
gen = opensimplex.OpenSimplex(seed)

def opensimplex_noise(shape, scale=1.0):
    noise = torch.zeros(shape, device='cpu')
    for i in range(shape[2]):
        for j in range(shape[3]):
            noise[0, 0, i, j] = gen.noise2(i / scale, j / scale)
    return noise.to('cuda' if torch.cuda.is_available() else 'cpu')

if latent is None:
    result = latent_shape + opensimplex_noise(latent_shape.shape, scale={local_dict.get('opensimplex_scale', 1.0)}) * {scale}
else:
    result = latent + opensimplex_noise(latent.shape, scale={local_dict.get('opensimplex_scale', 1.0)}) * {scale}
            """
        elif preset == "white":
            code = f"result = torch.randn(latent.shape, device=latent.device) * {scale}"
        elif preset == "pink":
            code = f"""
# Pink noise generation using inverse FFT
if latent is None:
    freqs = torch.fft.fftfreq(latent_shape.shape[2], d=1.0/latent_shape.shape[2], device=latent_shape.device)
    spectral_density = 1 / torch.sqrt(freqs[:, None]**2 + freqs[None, :]**2).clamp(min=1e-10)
    noise = torch.normal(mean=0.0, std=1.0, size=latent_shape.shape, dtype=latent_shape.dtype, device=latent_shape.device, generator=torch.manual_seed({seed}))
    noise_fft = torch.fft.fft2(noise)
    modified_fft = noise_fft * spectral_density.unsqueeze(0).unsqueeze(0)
    noise = torch.fft.ifft2(modified_fft).real
    result = latent_shape + (noise / torch.std(noise)) * {scale}
else:
    freqs = torch.fft.fftfreq(latent.shape[2], d=1.0/latent.shape[2], device=latent.device)
    spectral_density = 1 / torch.sqrt(freqs[:, None]**2 + freqs[None, :]**2).clamp(min=1e-10)
    noise = torch.normal(mean=0.0, std=1.0, size=latent.shape, dtype=latent.dtype, device=latent.device, generator=torch.manual_seed({seed}))
    noise_fft = torch.fft.fft2(noise)
    modified_fft = noise_fft * spectral_density.unsqueeze(0).unsqueeze(0)
    noise = torch.fft.ifft2(modified_fft).real
    result = latent + (noise / torch.std(noise)) * {scale}
            """
        elif preset == "brown":
            code = f"""
# Brown noise generation
if latent is None:
    cumsum = torch.cumsum(torch.randn(latent_shape.shape, device=latent_shape.device), dim=-1)
    result = latent_shape + (cumsum / torch.std(cumsum)) * {scale}
else:
    cumsum = torch.cumsum(torch.randn(latent.shape, device=latent.device), dim=-1)
    result = latent + (cumsum / torch.std(cumsum)) * {scale}
            """
        elif preset == "salt_and_pepper":
            code = f"""
# Salt and pepper noise generation
if latent is None:
    result = latent_shape.clone()
    noise = torch.randint(0, 2, latent_shape.shape, device=latent_shape.device).float()
    result[noise == 0] = 0
    result[noise == 1] = 1
else:
    result = latent.clone()
    noise = torch.randint(0, 2, latent.shape, device=latent.device).float()
    result[noise == 0] = 0
    result[noise == 1] = 1
            """

        else:
            code = ""

        return ({"formula": code, "variables": ""}, )

# ----------------------------------------------------------------------------
# Formula Builder
# ----------------------------------------------------------------------------

class LatentMathFormulaBuilder:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "formula1": ("LATENT_MATH_FORMULA",),  
                },
                "optional": {
                    "formula2": ("LATENT_MATH_FORMULA",), 
                    "formula3": ("LATENT_MATH_FORMULA",),
                    "formula4": ("LATENT_MATH_FORMULA",),
                    "formula5": ("LATENT_MATH_FORMULA",),
                    # ... (add more formula inputs as needed) 
                    "variables": ("STRING", {"multiline": True, "default": ""}),  
                }
               }
    RETURN_TYPES = ("LATENT_MATH_FORMULA",)
    FUNCTION = "build_formula"
    CATEGORY = "latent/custom"

    def build_formula(self, formula1, formula2=None, formula3=None, formula4=None, formula5=None, variables=""):
        combined_formula = formula1.get("formula", "")  # Access the 'formula' key from the dictionary
        if formula2:
            combined_formula += "\n\n" + formula2.get("formula", "")  # Access 'formula' 
        if formula3:
            combined_formula += "\n\n" + formula3.get("formula", "")  # Access 'formula'
        if formula4:
            combined_formula += "\n\n" + formula4.get("formula", "")  # Access 'formula'
        if formula5:
            combined_formula += "\n\n" + formula5.get("formula", "")  # Access 'formula'

        return {"formula": combined_formula.strip(), "variables": variables}

# ----------------------------------------------------------------------------
# Standalone Latent Disfigurement Nodes
# ----------------------------------------------------------------------------

class LatentFFT:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "latent": ("LATENT",),
                              "operation": (["low_pass", "high_pass", "band_pass"], {"default": "low_pass"}),
                              "cutoff_frequency": ("FLOAT", {"default": 0.25, "min": -10.0, "max": 10.0, "step": 0.01}),
                              "strength": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                             }}
    RETURN_TYPES = ("LATENT",)
    FUNCTION = "apply_fft"
    CATEGORY = "latent/custom"

    def apply_fft(self, latent, operation, cutoff_frequency, strength):
        device = comfy.model_management.get_torch_device()
        latent_tensor = latent["samples"].to(device)

        # Apply FFT (real to complex)
        fft_result = torch.fft.rfft2(latent_tensor, dim=(-2, -1), norm='ortho')

        # Create a mask in the frequency domain based on the operation
        mask = self.create_frequency_mask(fft_result.shape, operation, cutoff_frequency, device)

        # Apply the mask to the FFT result
        filtered_fft = fft_result * mask

        # Inverse FFT (complex to real)
        filtered_latent = torch.fft.irfft2(filtered_fft, s=latent_tensor.shape[-2:], dim=(-2, -1), norm='ortho')

        # Blend the original and filtered latent
        output_latent = latent_tensor * (1.0 - strength) + filtered_latent * strength

        return ({"samples": output_latent}, )

    def create_frequency_mask(self, shape, operation, cutoff, device):
        # Create a grid of frequencies
        freq_x = torch.fft.rfftfreq(shape[-1] * 2 - 2, d=1.0, device=device)  # Adjust for rfft2 size reduction
        freq_y = torch.fft.fftfreq(shape[-2], d=1.0, device=device)
        freq_grid_y, freq_grid_x = torch.meshgrid(freq_y, freq_x, indexing='ij')
        frequencies = torch.sqrt(freq_grid_x**2 + freq_grid_y**2)

        # Create the mask based on the selected operation
        if operation == "low_pass":
            mask = (frequencies <= cutoff).float()
        elif operation == "high_pass":
            mask = (frequencies >= cutoff).float()
        elif operation == "band_pass":
            cutoff_low = cutoff - 0.1  # Adjust band width as needed
            cutoff_high = cutoff + 0.1
            mask = ((frequencies >= cutoff_low) & (frequencies <= cutoff_high)).float()
        else:
            raise ValueError(f"Invalid FFT operation: {operation}")

        # Ensure mask matches the shape of the FFT result
        batch, channels, height, width = shape
        mask = mask.unsqueeze(0).unsqueeze(0).expand(batch, channels, height, mask.shape[-1])

        return mask

class LatentGlitch:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"latent": ("LATENT", ),
                             "block_size": ("INT", {"default": 16, "min": 1, "max": 128, "step": 1}),
                             "shift_amount": ("INT", {"default": 8, "min": -512, "max": 512, "step": 1}),
                             "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                            },
                "optional": {"glitch_mask": ("MASK", ),}
               }
    RETURN_TYPES = ("LATENT",)
    FUNCTION = "glitch"
    CATEGORY = "latent/custom"

    def glitch(self, latent, block_size, shift_amount, seed, glitch_mask=None):
        device = comfy.model_management.get_torch_device()
        latent_tensor = latent["samples"].to(device) 
        torch.manual_seed(seed)
        random.seed(seed)

        if glitch_mask is not None:
            # Load the mask from NumPy array 
            glitch_mask = torch.from_numpy(glitch_mask["samples"]).float().to(device)

            # Resize mask to match latent dimensions
            glitch_mask = comfy.utils.resize_to_batch_size(glitch_mask, latent_tensor.shape[0])
            glitch_mask = glitch_mask.reshape((-1, 1, glitch_mask.shape[-2], glitch_mask.shape[-1])).to(device)

            # Downsample the mask to the block size
            glitch_mask = F.interpolate(glitch_mask, size=(latent_tensor.shape[-2], latent_tensor.shape[-1] // block_size), mode="bilinear")

            # Apply glitch based on mask
            blocks = latent_tensor.chunk(latent_tensor.shape[-1] // block_size, dim=-1)
            for i in range(len(blocks)):
                # Apply random roll only to blocks that are within the mask
                if random.random() < 0.5 and glitch_mask[:, 0, i // block_size, i % block_size] > 0:
                    blocks[i] = torch.roll(blocks[i], shifts=shift_amount, dims=-1)
            latent_tensor = torch.cat(list(blocks), dim=-1)

        else:
            # Apply glitch to the entire image (same as before)
            blocks = latent_tensor.chunk(latent_tensor.shape[-1] // block_size, dim=-1) 
            blocks = [torch.roll(block, shifts=shift_amount, dims=-1) if random.random() < 0.5 else block for block in blocks]
            latent_tensor = torch.cat(blocks, dim=-1)

        return ({"samples": latent_tensor},)

class LatentTwist:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"latent": ("LATENT", ),
                              "angle": ("FLOAT", {"default": 45.0, "min": -180.0, "max": 180.0, "step": 1.0}),
                              "center_x": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                              "center_y": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                             }}
    RETURN_TYPES = ("LATENT",)
    FUNCTION = "twist"
    CATEGORY = "latent/custom"

    def twist(self, latent, angle, center_x, center_y):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        latent_tensor = latent["samples"].to(device)

        # Create grid for transformation
        theta = torch.deg2rad(torch.tensor(angle, device=device))
        grid_x, grid_y = torch.meshgrid(torch.linspace(-1, 1, latent_tensor.shape[3], device=device), 
                                        torch.linspace(-1, 1, latent_tensor.shape[2], device=device))
        
        # Center the grid 
        centered_x = grid_x - (center_x * 2 - 1) 
        centered_y = grid_y - (center_y * 2 - 1)
        
        # Apply rotation
        rotated_x = centered_x * torch.cos(theta) - centered_y * torch.sin(theta)
        rotated_y = centered_x * torch.sin(theta) + centered_y * torch.cos(theta)

        # Create the transformation grid
        grid = torch.stack((rotated_x, rotated_y), dim=-1).unsqueeze(0).repeat(latent_tensor.shape[0], 1, 1, 1) 
        
        # Apply warping
        twisted_latent = torch.nn.functional.grid_sample(latent_tensor, grid, mode="bilinear", padding_mode="zeros", align_corners=False) 
        
        return ({"samples": twisted_latent}, ) 

class LatentMosaic:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "latent": ("LATENT", ),
            "tile_size": ("INT", {"default": 16, "min": 1, "max": 128, "step": 1}),
            }}
    RETURN_TYPES = ("LATENT",)
    FUNCTION = "mosaic"
    CATEGORY = "latent/custom"

    def mosaic(self, latent, tile_size):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')        
        latent_tensor = latent["samples"].to(device)
        
        # Downsample and then upscale
        downsampled = torch.nn.functional.interpolate(latent_tensor, scale_factor=(1.0/tile_size), mode="nearest")
        mosaic_latent = torch.nn.functional.interpolate(downsampled, size=latent_tensor.shape[2:], mode="nearest")
        
        return ({"samples": mosaic_latent}, )

class LatentPixelSort:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"latent": ("LATENT", ),
                              "channel": ("INT", {"default": 0, "min": 0, "max": 3, "step": 1}),
                              "reverse": ("BOOLEAN", {"default": False}),
                             }}
    RETURN_TYPES = ("LATENT",)
    FUNCTION = "sort"
    CATEGORY = "latent/custom"

    def sort(self, latent, channel, reverse):
        latent_tensor = latent["samples"]
        
        # Sort along the selected channel
        sorted_indices = torch.argsort(latent_tensor[:, channel], dim=-1, descending=reverse)
        
        # Expand sorted_indices to match latent_tensor dimensions
        sorted_indices = comfy.utils.repeat_to_batch_size(sorted_indices.unsqueeze(1), latent_tensor.shape[1], dim=1)
        
        # Gather pixels based on sorted indices 
        pixel_sorted_latent = torch.gather(latent_tensor, dim=-1, index=sorted_indices) #.unsqueeze(1).expand(-1, latent_tensor.shape[1], -1))

        return ({"samples": pixel_sorted_latent}, )
    
