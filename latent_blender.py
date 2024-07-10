import torch
import torch.nn.functional as F

class LatentBlender:
    @staticmethod
    def blend_latent(samples, latent, transformed_latent, strength, add_to_original, composite, blend_amount, normalize, clamp, clamp_min, clamp_max, inplace):
        # Normalize if specified
        if normalize:
            transformed_latent = (transformed_latent - transformed_latent.mean()) / transformed_latent.std()

        # Clamp if specified
        if clamp:
            transformed_latent = torch.clamp(transformed_latent, min=clamp_min, max=clamp_max)

        if add_to_original and not inplace:
            output_latent = latent + strength * transformed_latent
        else:
            output_latent = transformed_latent

        samples_out = samples.copy()
        if composite:
            latent_height, latent_width = latent.shape[2], latent.shape[3]
            upscaled_latent = F.interpolate(latent, size=(latent_height, latent_width), mode='nearest')

            if output_latent.shape[2:] != latent.shape[2:]:
                output_latent = F.interpolate(output_latent, size=(latent_height, latent_width), mode='nearest')

            samples_out["samples"] = upscaled_latent * (1 - blend_amount) + output_latent * blend_amount
        else:
            samples_out["samples"] = output_latent

        return (samples_out, )
