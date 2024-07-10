# I've had some errors with device selection while using this. on the to-do list

import torch

class AttenuatorNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "sigma_input": ("SIGMAS",),
                "attenuation_factor": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 10.0, "step": 0.01}),
            },
        }

    RETURN_TYPES = ("SIGMAS",)
    RETURN_NAMES = ("attenuated_sigmas",)
    FUNCTION = "apply_attenuation"
    CATEGORY = "sampling/custom_sampling/utilities"

    def __init__(self, inverse_fourier=False):
        self.inverse_fourier = inverse_fourier

    def apply_attenuation(self, sigma_input, attenuation_factor):
        # Perform the calculations in complex128
        sigma = sigma_input.numpy().astype('complex128')
        attenuated_sigma = sigma * attenuation_factor
        
        # Optionally perform the inverse Fourier transform
        if self.inverse_fourier:
            attenuated_sigma = torch.fft.ifft(torch.tensor(attenuated_sigma, dtype=torch.complex128)).numpy()

        # Convert the result to torch.float64
        sigma_tensor = torch.tensor(attenuated_sigma, dtype=torch.float64)
        return (sigma_tensor,)
