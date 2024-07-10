import torch
import nodes 
from torch.nn import functional as F
from comfy.k_diffusion import sampling as k_diffusion_sampling
import comfy.model_patcher 

class SoftmaxScheduler:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "model": ("MODEL",),
                              "steps": ("INT", {"default": 50, "min": 1, "max": 10000}),
                              "decay_factor": ("FLOAT", {"default": 0.95, "min": 0.0, "max": 1.0, "step": 0.01}),  
                             }}

    RETURN_TYPES = ("SIGMAS",)
    FUNCTION = "get_sigmas"
    CATEGORY = "sampling/custom_sampling/schedulers"

    def get_sigmas(self, model, steps, decay_factor):
        # Get the device of the model (GPU or CPU)
        device = model.get_model_object("model_sampling").sigmas.device  

        # Generate geometric series weights
        step_weights = [decay_factor**i for i in range(50)] 

        # Apply softmax to get probabilities
        probabilities = F.softmax(torch.tensor(step_weights, device=device), dim=0)

        # Create a tensor of evenly spaced values from 0 to 1 
        t_values = torch.linspace(0, 1, steps + 1, device=device)

        # Map probabilities to sigma values using linear interpolation
        sigma_max = model.get_model_object("model_sampling").sigma_max 
        sigma_min = model.get_model_object("model_sampling").sigma_min 

        # Interpolation using F.interpolate (ensuring tensors are on the same device)
        probabilities_cumsum = probabilities.cumsum(dim=0).unsqueeze(0).unsqueeze(0)
        interp_values = F.interpolate(probabilities_cumsum, size=(steps + 1,), mode='linear', align_corners=False).squeeze().to(device)

        sigmas = sigma_max - (sigma_max - sigma_min) * interp_values

        sigmas[-1] = 0.0291675  # Set the last sigma

        return (sigmas, )

# Register the node
nodes.NODE_CLASS_MAPPINGS["SoftmaxScheduler"] = SoftmaxScheduler

class HardTanhScheduler:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "model": ("MODEL",),
                              "steps": ("INT", {"default": 10, "min": 1, "max": 10000}), 
                              "base_scheduler": (comfy.samplers.SCHEDULER_NAMES, ),
                              "min_sigma_scale": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 1.0, "step": 0.01}),
                              "max_sigma_scale": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                             }}

    RETURN_TYPES = ("SIGMAS",)
    FUNCTION = "get_sigmas"
    CATEGORY = "sampling/custom_sampling/schedulers"

    def get_sigmas(self, model, steps, base_scheduler, min_sigma_scale, max_sigma_scale):
        # Generate sigmas from the base scheduler
        base_sigmas = comfy.samplers.calculate_sigmas(model.get_model_object("model_sampling"), base_scheduler, steps).cpu()

        # Create a tensor of evenly spaced values from -1 to 1
        x_values = torch.linspace(-1, 1, steps + 1)

        # Apply HardTanh to scale values between -1 and 1
        scaled_values = F.hardtanh(x_values) 

        # Scale the values to the desired sigma scale range
        sigma_scale_range = max_sigma_scale - min_sigma_scale
        scaled_sigmas = min_sigma_scale + sigma_scale_range * (scaled_values + 1.0) / 2.0 

        # Multiply the base sigmas by the scaled values
        sigmas = base_sigmas * scaled_sigmas 
        sigmas[-1] = 0.0 # Ensure the last sigma is zero

        return (sigmas, )

nodes.NODE_CLASS_MAPPINGS["HardTanhScheduler"] = HardTanhScheduler
