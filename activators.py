# from custom_nodes/latent_activator/activation.py

import torch
import torch.nn.functional as F
from .latent_blender import LatentBlender
import inspect

class Activation:
    @staticmethod
    def blend(samples, latent, transformed_latent, strength, add_to_original, normalize, clamp, clamp_min, clamp_max, composite, blend_amount, inplace=False):
        return LatentBlender.blend_latent(samples, latent, transformed_latent, strength, add_to_original, composite, blend_amount, normalize, clamp, clamp_min, clamp_max, inplace)

    def apply_activation(self, samples, strength, add_to_original, normalize, clamp, clamp_min, clamp_max, composite, blend_amount, inplace=False, **kwargs):
        latent = samples["samples"].clone() if inplace else samples["samples"]

        # Check if the activation function supports the inplace argument
        sig = inspect.signature(self.activation_function)
        if 'inplace' in sig.parameters:
            transformed_latent = self.activation_function(latent, inplace=inplace, **kwargs)
        else:
            transformed_latent = self.activation_function(latent, **kwargs)

        return self.blend(samples, latent, transformed_latent, strength, add_to_original, normalize, clamp, clamp_min, clamp_max, composite, blend_amount, inplace)

class Threshold(Activation):
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "samples": ("LATENT",),
            "threshold": ("FLOAT", {"default": 0.0, "min": -10.0, "max": 10.0, "step": 0.1}),
            "value": ("FLOAT", {"default": 0.0, "min": -10.0, "max": 10.0, "step": 0.1}),
            "inplace": ("BOOLEAN", {"default": False}),
            "strength": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
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

    def activation_function(self, latent, threshold, value, inplace):
        return F.threshold(latent, threshold=threshold, value=value, inplace=inplace)

class ReLU(Activation):
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "samples": ("LATENT",),
            "inplace": ("BOOLEAN", {"default": False}),
            "strength": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
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

    def activation_function(self, latent, inplace):
        return F.relu(latent, inplace=inplace)

class RReLU(Activation):
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "samples": ("LATENT",),
            "lower": ("FLOAT", {"default": 1.0 / 8.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            "upper": ("FLOAT", {"default": 1.0 / 3.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            "inplace": ("BOOLEAN", {"default": False}),
            "strength": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
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

    def activation_function(self, latent, lower, upper, inplace):
        return F.rrelu(latent, lower=lower, upper=upper, inplace=inplace)

class Hardtanh(Activation):
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "samples": ("LATENT",),
            "min_val": ("FLOAT", {"default": -1.0, "min": -10.0, "max": 10.0, "step": 0.1}),
            "max_val": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.1}),
            "inplace": ("BOOLEAN", {"default": False}),
            "strength": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
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

    def activation_function(self, latent, min_val, max_val, inplace):
        return F.hardtanh(latent, min_val=min_val, max_val=max_val, inplace=inplace)

class ReLU6(Activation):
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "samples": ("LATENT",),
            "inplace": ("BOOLEAN", {"default": False}),
            "strength": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
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

    def activation_function(self, latent, inplace):
        return F.relu6(latent, inplace=inplace)

class Hardsigmoid(Activation):
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "samples": ("LATENT",),
            "inplace": ("BOOLEAN", {"default": False}),
            "strength": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
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

    def activation_function(self, latent, inplace):
        return F.hardsigmoid(latent, inplace=inplace)

class Tanh(Activation):
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "samples": ("LATENT",),
            "strength": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
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

    def activation_function(self, latent):
        return torch.tanh(latent)

class SiLU(Activation):
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "samples": ("LATENT",),
            "inplace": ("BOOLEAN", {"default": False}),
            "strength": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
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

    def activation_function(self, latent, inplace):
        return F.silu(latent, inplace=inplace)

class Mish(Activation):
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "samples": ("LATENT",),
            "inplace": ("BOOLEAN", {"default": False}),
            "strength": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
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

    def activation_function(self, latent, inplace):
        return F.mish(latent, inplace=inplace)

class Hardswish(Activation):
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "samples": ("LATENT",),
            "inplace": ("BOOLEAN", {"default": False}),
            "strength": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
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

    def activation_function(self, latent, inplace):
        return F.hardswish(latent, inplace=inplace)

class ELU(Activation):
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "samples": ("LATENT",),
            "alpha": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
            "inplace": ("BOOLEAN", {"default": False}),
            "strength": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
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

    def activation_function(self, latent, alpha, inplace):
        return F.elu(latent, alpha=alpha, inplace=inplace)

class CELU(Activation):
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "samples": ("LATENT",),
            "alpha": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
            "inplace": ("BOOLEAN", {"default": False}),
            "strength": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
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

    def activation_function(self, latent, alpha, inplace):
        return F.celu(latent, alpha=alpha, inplace=inplace)

class SELU(Activation):
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "samples": ("LATENT",),
            "inplace": ("BOOLEAN", {"default": False}),
            "strength": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
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

    def activation_function(self, latent, inplace):
        return F.selu(latent, inplace=inplace)

class GLU(Activation):
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "samples": ("LATENT",),
            "dim": ("INT", {"default": -1, "choices": [-1, 2, 3]}),
            "strength": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
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

    def activation_function(self, latent, dim):
        return F.glu(latent, dim=dim)

# from custom_nodes/latent_activator/activation.py

class GELU(Activation):
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "samples": ("LATENT",),
            "approximate": ("STRING", {"default": "none", "options": ["none", "tanh"]}),
            "strength": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
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

    def activation_function(self, latent, approximate):
        return F.gelu(latent, approximate=approximate)

class Hardshrink(Activation):
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "samples": ("LATENT",),
            "lambd": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 10.0, "step": 0.1}),
            "strength": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
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

    def activation_function(self, latent, lambd):
        return F.hardshrink(latent, lambd=lambd)

class LeakyReLU(Activation):
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "samples": ("LATENT",),
            "negative_slope": ("FLOAT", {"default": 0.01, "min": 0.0, "max": 1.0, "step": 0.01}),
            "inplace": ("BOOLEAN", {"default": False}),
            "strength": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
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

    def activation_function(self, latent, negative_slope, inplace):
        return F.leaky_relu(latent, negative_slope=negative_slope, inplace=inplace)            

class LogSigmoid(Activation):
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "samples": ("LATENT",),
            "strength": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
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

    def activation_function(self, latent):
        return F.logsigmoid(latent)

class Softplus(Activation):
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "samples": ("LATENT",),
            "beta": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.1}),
            "threshold": ("FLOAT", {"default": 20.0, "min": 0.0, "max": 100.0, "step": 0.1}),
            "strength": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
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

    def activation_function(self, latent, beta, threshold):
        return F.softplus(latent, beta=beta, threshold=threshold)

class Softshrink(Activation):
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "samples": ("LATENT",),
            "lambd": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 10.0, "step": 0.1}),
            "strength": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
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

    def activation_function(self, latent, lambd):
        return F.softshrink(latent, lambd=lambd)

class PReLU(Activation):
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "samples": ("LATENT",),
            "num_parameters": ("INT", {"default": 1, "min": 1, "max": 100}),
            "strength": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
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

    def activation_function(self, latent, num_parameters):
        prelu = torch.nn.PReLU(num_parameters=num_parameters)
        return prelu(latent)

class Softsign(Activation):
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "samples": ("LATENT",),
            "strength": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
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

    def activation_function(self, latent):
        return F.softsign(latent)

class Tanhshrink(Activation):
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "samples": ("LATENT",),
            "strength": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
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

    def activation_function(self, latent):
        return F.tanhshrink(latent)
    
class Softmin(Activation):
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "samples": ("LATENT",),
            "dim": ("INT", {"default": -1, "min": -1, "max": 10, "step": 1}),
            "strength": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
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

    def activation_function(self, latent, dim):
        return F.softmin(latent, dim=dim)

class Softmax(Activation):
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "samples": ("LATENT",),
            "dim": ("INT", {"default": -1, "min": -1, "max": 10, "step": 1}),
            "strength": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
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

    def activation_function(self, latent, dim):
        return F.softmax(latent, dim=dim)

class Softmax2D(Activation):
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "samples": ("LATENT",),
            "strength": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
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

    def activation_function(self, latent):
        return F.softmax(latent, dim=1)

class LogSoftmax(Activation):
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "samples": ("LATENT",),
            "dim": ("INT", {"default": -1, "min": -1, "max": 10, "step": 1}),
            "strength": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
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

    def activation_function(self, latent, dim):
        return F.log_softmax(latent, dim=dim)
    
import torch.nn as nn

