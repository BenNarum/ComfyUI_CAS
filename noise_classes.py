# Needs refactoring. 
# Really gotta cut down on the wavelets, as they're almost identical.
# Probably need a better prepare_noise, for dealing with the different nuances

import torch
import torch.nn as nn
from torch import Tensor, Generator, lerp
from torch.nn.functional import unfold
from typing import Callable, Tuple
import math 
from math import pi
from comfy.k_diffusion.sampling import BrownianTreeNoiseSampler
from torch.distributions import StudentT, Laplace
import numpy as np
import matplotlib.pyplot as plt
import pywt
import functools

def cast_fp64(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        target_device = None
        target_dtype = torch.float64
        for arg in args:
            if torch.is_tensor(arg):
                target_device = arg.device
                target_dtype = arg.dtype
                break
        if target_device is None:
            for v in kwargs.values():
                if torch.is_tensor(v):
                    target_device = v.device
                    target_dtype = v.dtype
                    break

        def cast_and_move_to_device(data):
            if torch.is_tensor(data):
                return data.to(target_dtype).to(target_device)
            elif isinstance(data, dict):
                return {k: cast_and_move_to_device(v) for k, v in data.items()}
            elif isinstance(data, (list, tuple)):
                return type(data)(cast_and_move_to_device(v) for v in data)
            return data

        new_args = [cast_and_move_to_device(arg) for arg in args]
        new_kwargs = {k: cast_and_move_to_device(v) for k, v in kwargs.items()}

        return func(*new_args, **new_kwargs)
    return wrapper



def like(x):
    return {'size': x.shape, 'dtype': x.dtype, 'layout': x.layout, 'device': x.device}

def scale_to_range(x, scaled_min=-1.73, scaled_max=1.73):
    return scaled_min + (x - x.min()) * (scaled_max - scaled_min) / (x.max() - x.min())

def normalize(x):
    return (x - x.mean()) / x.std()

class NoiseGenerator:
    def __init__(self, x=None, size=None, dtype=None, layout=None, device=None, seed=42, generator=None, sigma_min=None, sigma_max=None):
        self.seed = seed
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max

        if x is not None:
            self.x = x
            self.size = x.shape
            self.dtype = x.dtype
            self.layout = x.layout
            self.device = x.device
        else:
            self.x = torch.zeros(size, dtype=dtype, layout=layout, device=device)

        if size is not None:
            self.size = size
        if dtype is not None:
            self.dtype = dtype
        if layout is not None:
            self.layout = layout
        if device is not None:
            self.device = device

        if generator is None:
            self.generator = torch.Generator(device=self.device).manual_seed(seed)
        else:
            self.generator = generator

    def __call__(self):
        raise NotImplementedError("This method must be implemented in subclasses.")

    def update(self, **kwargs):
        updated_values = []
        for attribute_name, value in kwargs.items():
            if value is not None:
                setattr(self, attribute_name, value)
            updated_values.append(getattr(self, attribute_name))
        return tuple(updated_values)

    
# Define other noise generators here as needed, e.g., FractalNoiseGenerator

class BrownianNoiseGenerator(NoiseGenerator):
    def __init__(self, x=None, size=None, dtype=None, layout=None, device=None, seed=42, generator=None, sigma_min=None, sigma_max=None, 
                 hurst=0.5, length=1.0):
        super().__init__(x, size, dtype, layout, device, seed, generator, sigma_min, sigma_max)
        self.hurst = hurst
        self.length = length

    def generate_brownian_noise(self, size: Tuple[int, int], hurst: float, length: float) -> Tensor:
        t = torch.linspace(0, length, size[0] * size[1], device=self.device, dtype=self.dtype)
        t = t.view(size)
        dt = t[1] - t[0]

        gaussian_noise = torch.randn(size, device=self.device, dtype=self.dtype) * dt.sqrt()
        brownian_noise = torch.cumsum(gaussian_noise, dim=0)
        brownian_noise /= brownian_noise.std()

        brownian_noise = torch.pow(brownian_noise, hurst)
        return brownian_noise

    def __call__(self, *, hurst=None, length=None, **kwargs):
        self.update(hurst=hurst, length=length)
        
        b, c, h, w = self.size
        noise_shape = kwargs.get('size', (h, w))

        noise = torch.stack([self.generate_brownian_noise(noise_shape, self.hurst, self.length) for _ in range(b * c)], dim=0)
        noise = noise.view(b, c, h, w)
        noise = (noise - noise.mean()) / (noise.std() + 1e-5)  # Ensure normalization
        return noise

class FractalNoiseGenerator(NoiseGenerator):
    def __init__(self, x=None, size=None, dtype=None, layout=None, device=None, seed=42, generator=None, sigma_min=None, sigma_max=None,
                 alpha=0.0, k=1.0, scale=0.1):
        super().__init__(x, size, dtype, layout, device, seed, generator, sigma_min, sigma_max)
        self.alpha = alpha
        self.k = k
        self.scale = scale

    def __call__(self, *, alpha=None, k=None, scale=None, sigmas=None, **kwargs):
        if alpha is not None:
            self.alpha = alpha
        if k is not None:
            self.k = k
        if scale is not None:
            self.scale = scale

        b, c, h, w = self.size
        noise = torch.normal(mean=0.0, std=1.0, size=(b, c, h, w), dtype=self.dtype, layout=self.layout, device=self.device, generator=self.generator)
        
        y_freq = torch.fft.fftfreq(h, 1/h, device=self.device)
        x_freq = torch.fft.fftfreq(w, 1/w, device=self.device)
        freq = torch.sqrt(y_freq[:, None]**2 + x_freq[None, :]**2).clamp(min=1e-10)
        
        spectral_density = self.k / torch.pow(freq, self.alpha * self.scale)
        spectral_density[0, 0] = 0

        noise_fft = torch.fft.fft2(noise)
        modified_fft = noise_fft * spectral_density
        noise = torch.fft.ifft2(modified_fft).real

        return noise / torch.std(noise)

class PyramidNoiseGenerator(NoiseGenerator):
    def __init__(self, x=None, size=None, dtype=None, layout=None, device=None, seed=42, generator=None, sigma_min=None, sigma_max=None, 
                 discount=0.8, mode='nearest-exact'):
        self.update(discount=discount, mode=mode)
        super().__init__(x, size, dtype, layout, device, seed, generator, sigma_min, sigma_max)

    def __call__(self, *, discount=None, mode=None, **kwargs):
        self.update(discount=discount, mode=mode)

        b, c, h, w = self.size
        orig_h, orig_w = h, w

        noise_shape = kwargs.get('size', (b, c, orig_h, orig_w))
        device = kwargs.get('device', self.device)
        dtype = kwargs.get('dtype', self.dtype)
        layout = kwargs.get('layout', self.layout)
        generator = kwargs.get('generator', self.generator)

        x = torch.zeros(noise_shape, dtype=dtype, layout=layout, device=device)
        
        r = 1
        for i in range(5):
            r *= 2
            new_noise_shape = (b, c, orig_h * r, orig_w * r)
            noise = torch.normal(mean=0, std=0.5 ** i, size=new_noise_shape, dtype=dtype, layout=layout, device=device, generator=generator)
            if self.mode == 'linear':
                noise = noise.view(b * c, orig_h * r * orig_w * r)
                x += torch.nn.functional.interpolate(noise.unsqueeze(1), size=(orig_h * orig_w), mode=self.mode).view(b, c, orig_h, orig_w) * self.discount ** i
            elif self.mode == 'trilinear':
                noise = noise.view(b, c, 1, orig_h * r, orig_w * r)
                x += torch.nn.functional.interpolate(noise, size=(1, orig_h, orig_w), mode=self.mode).view(b, c, orig_h, orig_w) * self.discount ** i
            else:
                x += torch.nn.functional.interpolate(noise, size=(orig_h, orig_w), mode=self.mode) * self.discount ** i
        
        return x / x.std()

class HiresPyramidNoiseGenerator(NoiseGenerator):
    def __init__(self, x=None, size=None, dtype=None, layout=None, device=None, seed=42, generator=None, sigma_min=None, sigma_max=None, 
                 discount=0.7, mode='nearest-exact'):
        self.update(discount=discount, mode=mode)
        super().__init__(x, size, dtype, layout, device, seed, generator, sigma_min, sigma_max)

    def __call__(self, *, discount=None, mode=None, **kwargs):
        self.update(discount=discount, mode=mode)

        b, c, h, w = self.size
        orig_h, orig_w = h, w

        noise_shape = kwargs.get('size', (b, c, orig_h, orig_w))
        device = kwargs.get('device', self.device)
        dtype = kwargs.get('dtype', self.dtype)
        layout = kwargs.get('layout', self.layout)
        generator = kwargs.get('generator', self.generator)

        u = nn.Upsample(size=(orig_h, orig_w), mode=self.mode).to(device)

        noise = ((torch.rand(size=noise_shape, dtype=dtype, layout=layout, device=device, generator=generator) - 0.5) * 2 * 1.73)

        for i in range(4):
            r = torch.rand(1, device=device, generator=generator).item() * 2 + 2
            new_h, new_w = min(orig_h * 15, int(orig_h * (r ** i))), min(orig_w * 15, int(orig_w * (r ** i)))

            new_noise = torch.randn((b, c, new_h, new_w), dtype=dtype, layout=layout, device=device, generator=generator)
            if self.mode == 'linear':
                new_noise = new_noise.view(b * c, new_h * new_w)
                upsampled_noise = torch.nn.functional.interpolate(new_noise.unsqueeze(1), size=(orig_h * orig_w), mode=self.mode).view(b, c, orig_h, orig_w)
            elif self.mode == 'trilinear':
                new_noise = new_noise.view(b, c, 1, new_h, new_w)
                upsampled_noise = torch.nn.functional.interpolate(new_noise, size=(1, orig_h, orig_w), mode=self.mode).view(b, c, orig_h, orig_w)
            else:
                upsampled_noise = torch.nn.functional.interpolate(new_noise, size=(orig_h, orig_w), mode=self.mode)
            noise += upsampled_noise * self.discount ** i
            
            if new_h >= orig_h * 15 or new_w >= orig_w * 15:
                break  # if resolution is too high
        
        return noise / noise.std()
# Example Usage:
# hires_pyramid_generator = HiresPyramidNoiseGenerator(size=(1, 3, 256, 256))
# noise = hires_pyramid_generator(size=(1, 3, 512, 512), device='cuda')
        

class UniformNoiseGenerator(NoiseGenerator):
    def __init__(self, x=None, size=None, dtype=None, layout=None, device=None, seed=42, generator=None, sigma_min=None, sigma_max=None, 
                 mean=0.0, scale=1.73):
        self.update(mean=mean, scale=scale)

        super().__init__(x, size, dtype, layout, device, seed, generator, sigma_min, sigma_max)

    def __call__(self, *, mean=None, scale=None, **kwargs):
        self.update(mean=mean, scale=scale)

        noise = torch.rand(self.size, dtype=self.dtype, layout=self.layout, device=self.device, generator=self.generator)

        return self.scale * 2 * (noise - 0.5) + self.mean

class GaussianNoiseGenerator(NoiseGenerator):
    def __init__(self, x=None, size=None, dtype=None, layout=None, device=None, seed=42, generator=None, sigma_min=None, sigma_max=None, 
                 mean=0.0, std=1.0):
        self.update(mean=mean, std=std)
        super().__init__(x, size, dtype, layout, device, seed, generator, sigma_min, sigma_max)

    def __call__(self, *, mean=None, std=None, **kwargs):
        self.update(mean=mean, std=std)
        
        b, c, h, w = self.size
        
        noise_shape = kwargs.get('size', (b, c, h, w))

        noise = torch.randn(noise_shape, dtype=self.dtype, layout=self.layout, device=self.device, generator=self.generator)

        return noise * self.std + self.mean


class LaplacianNoiseGenerator(NoiseGenerator):
    def __init__(self, x=None, size=None, dtype=None, layout=None, device=None, seed=42, generator=None, sigma_min=None, sigma_max=None, 
                 loc=0, scale=1.0):
        self.update(loc=loc, scale=scale)
        super().__init__(x, size, dtype, layout, device, seed, generator, sigma_min, sigma_max)

    def __call__(self, *, loc=None, scale=None, **kwargs):
        self.update(loc=loc, scale=scale)
        
        b, c, h, w = self.size
        
        noise_shape = kwargs.get('size', (b, c, h, w))

        noise = torch.randn(noise_shape, dtype=self.dtype, layout=self.layout, device=self.device, generator=self.generator) / 4.0

        rng_state = torch.random.get_rng_state()
        torch.manual_seed(self.generator.initial_seed())
        laplacian_noise = Laplace(loc=self.loc, scale=self.scale).rsample(noise_shape).to(self.device)
        self.generator.manual_seed(self.generator.initial_seed() + 1)
        torch.random.set_rng_state(rng_state)

        noise += laplacian_noise
        return noise / noise.std()


class StudentTNoiseGenerator(NoiseGenerator):
    def __init__(self, x=None, size=None, dtype=None, layout=None, device=None, seed=42, generator=None, sigma_min=None, sigma_max=None, 
                 loc=0, scale=0.2, df=1):
        self.update(loc=loc, scale=scale, df=df)
        super().__init__(x, size, dtype, layout, device, seed, generator, sigma_min, sigma_max)

    def __call__(self, *, loc=None, scale=None, df=None, **kwargs):
        self.update(loc=loc, scale=scale, df=df)
        
        b, c, h, w = self.size
        
        noise_shape = kwargs.get('size', (b, c, h, w))

        rng_state = torch.random.get_rng_state()
        torch.manual_seed(self.generator.initial_seed())

        noise = StudentT(loc=self.loc, scale=self.scale, df=self.df).rsample(noise_shape)

        s = torch.quantile(noise.flatten(start_dim=1).abs(), 0.75, dim=-1)
        s = s.reshape(*s.shape, 1, 1, 1)
        noise = noise.clamp(-s, s)

        noise_latent = torch.copysign(torch.pow(torch.abs(noise), 0.5), noise).to(self.device)

        self.generator.manual_seed(self.generator.initial_seed() + 1)
        torch.random.set_rng_state(rng_state)
        return (noise_latent - noise_latent.mean()) / noise_latent.std()


# Class for generating noise using wavelet transforms
class WaveletNoiseGenerator(NoiseGenerator):
    def __init__(self, x=None, size=None, dtype=None, layout=None, device=None, seed=42, generator=None, sigma_min=None, sigma_max=None,
                 wavelet='haar', mode='symmetric'):
        self.update(wavelet=wavelet, mode=mode)
        super().__init__(x, size, dtype, layout, device, seed, generator, sigma_min, sigma_max)

    def __call__(self, *, wavelet=None, mode=None, sigmas=None, **kwargs):
        self.update(wavelet=wavelet, mode=mode)

        noise_shape = kwargs.get('size', self.size)
        device = kwargs.get('device', self.device)
        dtype = kwargs.get('dtype', self.dtype)
        layout = kwargs.get('layout', self.layout)
        generator = kwargs.get('generator', self.generator)

        b, c, h, w = noise_shape

        # Generate random noise
        noise_tensor = torch.randn((b * c, h, w), dtype=dtype, layout=layout, device=device, generator=generator)

        if wavelet in pywt.wavelist(kind='continuous'):
            # Perform continuous wavelet transform
            scales = np.arange(1, min(h, w) + 1)
            coeffs, _ = pywt.cwt(noise_tensor.cpu().numpy(), scales=scales, wavelet=wavelet)
            noise = np.sum(coeffs, axis=0)
        else:
            # Perform discrete wavelet decomposition and reconstruction with specified mode
            coeffs = pywt.wavedecn(noise_tensor.cpu().numpy(), wavelet=self.wavelet, mode=self.mode)
            noise = pywt.waverecn(coeffs, wavelet=self.wavelet, mode=self.mode)

        noise_tensor = torch.tensor(noise, dtype=dtype, device=device).view(b, c, h, w)

        # Apply sigma scaling
        if sigmas is not None:
            noise_tensor = noise_tensor * sigmas.view(b, 1, 1, 1)

        noise_tensor = (noise_tensor - noise_tensor.mean()) / noise_tensor.std()

        return noise_tensor

class PerlinNoiseGenerator(NoiseGenerator):
    def __init__(self, x=None, size=None, dtype=None, layout=None, device=None, seed=42, generator=None, sigma_min=None, sigma_max=None, scale=1):
        super().__init__(x, size, dtype, layout, device, seed, generator, sigma_min, sigma_max)
        self.scale = scale
        self.perm = torch.randint(0, 256, (256,), device=self.device, dtype=torch.int)
        self.perm = torch.cat([self.perm, self.perm])

    def fade(self, t):
        return t * t * t * (t * (t * 6 - 15) + 10)

    def lerp(self, t, a, b):
        return a + t * (b - a)

    def grad(self, hash, x, y):
        h = hash & 3
        u = torch.where(h < 2, x, y)
        v = torch.where(h < 2, y, x)
        return torch.where(h & 1 == 0, u, -u) + torch.where(h & 2 == 0, v, -v)

    def perlin(self, x, y):
        X = x.int()
        Y = y.int()
        x = x - X
        y = y - Y
        u = self.fade(x)
        v = self.fade(y)
        X = X & 255
        Y = Y & 255
        p = self.perm
        aa = p[X + p[Y]]
        ab = p[X + p[Y + 1]]
        ba = p[X + 1 + p[Y]]
        bb = p[X + 1 + p[Y + 1]]
        return self.lerp(v, self.lerp(u, self.grad(aa, x, y), self.grad(ba, x - 1, y)),
                         self.lerp(u, self.grad(ab, x, y - 1), self.grad(bb, x - 1, y - 1)))

    def __call__(self, *, scale=None, **kwargs):
        self.update(scale=scale, **kwargs)
        b, c, h, w = self.size

        x = torch.linspace(0, h / self.scale, h, device=self.device, dtype=self.dtype)
        y = torch.linspace(0, w / self.scale, w, device=self.device, dtype=self.dtype)
        grid_x, grid_y = torch.meshgrid(x.clone(), y.clone())

        noise = torch.zeros((b, c, h, w), device=self.device, dtype=self.dtype)

        for i in range(b):
            for j in range(c):
                noise[i, j, :, :] = self.perlin(grid_x.clone(), grid_y.clone())

        return noise

class OpenSimplexNoiseGenerator(NoiseGenerator):
    def __init__(self, x=None, size=None, dtype=None, layout=None, device=None, seed=None, generator=None, sigma_min=None, sigma_max=None):
        seed = seed % (2**32)  # Ensure seed is within the allowed range
        super().__init__(x, size, dtype, layout, device, seed, generator, sigma_min, sigma_max)
        self.noise_generator = OpenSimplexNoise(seed)

    def generate_noise(self, shape: Tuple[int, int], scale: float = 1.0) -> Tensor:
        height, width = shape
        coords = np.indices((height, width)).reshape(2, -1).T
        coords = coords / scale
        noise = np.apply_along_axis(lambda p: self.noise_generator.noise2d(p[0], p[1]), 1, coords)
        noise = noise.reshape(height, width)
        return torch.tensor(noise, device=self.device, dtype=self.dtype)

    def __call__(self, *, scale=1.0, **kwargs):
        b, c, h, w = self.size
        noise_shape = kwargs.get('size', (h, w))
        noise = torch.stack([self.generate_noise(noise_shape, scale=scale) for _ in range(b * c)], dim=0)
        return noise.view(b, c, h, w)

class OpenSimplexNoise:
    def __init__(self, seed: int):
        self.perm = np.arange(256, dtype=int)
        np.random.seed(seed)
        np.random.shuffle(self.perm)
        self.perm = np.stack([self.perm, self.perm]).flatten()

    def noise2d(self, x: float, y: float) -> float:
        STRETCH_CONSTANT_2D = -0.211324865405187
        SQUISH_CONSTANT_2D = 0.366025403784439
        NORM_CONSTANT_2D = 47

        stretch_offset = (x + y) * STRETCH_CONSTANT_2D
        xs = x + stretch_offset
        ys = y + stretch_offset
        xsb = int(np.floor(xs))
        ysb = int(np.floor(ys))

        squish_offset = (xsb + ysb) * SQUISH_CONSTANT_2D
        xb = xsb + squish_offset
        yb = ysb + squish_offset
        xins = xs - xsb
        yins = ys - ysb
        in_sum = xins + yins

        dx0 = x - xb
        dy0 = y - yb

        value = 0.0
        if in_sum <= 1:
            a_score = xins
            b_score = yins
            a_point = 0x01
            b_point = 0x02
            if a_score >= b_score and yins > 0:
                a_score = yins
                a_point = 0x02
                b_score = xins
                b_point = 0x01
            elif a_score < b_score and xins > 0:
                a_score = xins
                a_point = 0x01
                b_score = yins
                b_point = 0x02

            wins = 1 - in_sum
            if wins > a_score or wins > b_score:
                if wins > a_score:
                    c = (xsb + 1, ysb + 1)
                else:
                    c = (xsb + (a_point & 1), ysb + (a_point >> 1))
                dx_ext = dx0 + 1 - SQUISH_CONSTANT_2D
                dy_ext = dy0 + 1 - SQUISH_CONSTANT_2D
                value += (2 - dx_ext * dx_ext - dy_ext * dy_ext) ** 4 * self.extrapolate(c[0], c[1], dx_ext, dy_ext)
            if wins > b_score:
                c = (xsb + (b_point & 1), ysb + (b_point >> 1))
                dx_ext = dx0 + (b_point & 1) - SQUISH_CONSTANT_2D
                dy_ext = dy0 + (b_point >> 1) - SQUISH_CONSTANT_2D
                value += (2 - dx_ext * dx_ext - dy_ext * dy_ext) ** 4 * self.extrapolate(c[0], c[1], dx_ext, dy_ext)
        else:
            a_score = 1 - xins
            b_score = 1 - yins
            a_point = 0x01
            b_point = 0x02
            if a_score <= b_score and xins < 1:
                a_score = xins
                a_point = 0x01
                b_score = 1 - yins
                b_point = 0x02
            elif a_score > b_score and yins < 1:
                a_score = yins
                a_point = 0x02
                b_score = 1 - xins
                b_point = 0x01

            wins = 2 - in_sum
            if wins > a_score or wins > b_score:
                if wins > a_score:
                    c = (xsb - 1, ysb - 1)
                else:
                    c = (xsb - (a_point & 1), ysb - (a_point >> 1))
                dx_ext = dx0 - 1 - SQUISH_CONSTANT_2D
                dy_ext = dy0 - 1 - SQUISH_CONSTANT_2D
                value += (2 - dx_ext * dx_ext - dy_ext * dy_ext) ** 4 * self.extrapolate(c[0], c[1], dx_ext, dy_ext)
            if wins > b_score:
                c = (xsb - (b_point & 1), ysb - (b_point >> 1))
                dx_ext = dx0 - (b_point & 1) - SQUISH_CONSTANT_2D
                dy_ext = dy0 - (b_point >> 1) - SQUISH_CONSTANT_2D
                value += (2 - dx_ext * dx_ext - dy_ext * dy_ext) ** 4 * self.extrapolate(c[0], c[1], dx_ext, dy_ext)

        return value / NORM_CONSTANT_2D

    def extrapolate(self, xsb, ysb, dx, dy):
        index = self.perm[(self.perm[xsb & 0xff] + ysb) & 0xff] & 0x0e
        grad2 = [
            (5, 2), (2, 5), (-5, 2), (-2, 5), (5, -2), (2, -5), (-5, -2), (-2, -5),
            (5, 2), (2, 5), (-5, 2), (-2, 5), (5, -2), (2, -5), (-5, -2), (-2, -5)
        ]
        g = grad2[index >> 1]
        return g[0] * dx + g[1] * dy


class SuperSimplexNoiseGenerator(NoiseGenerator):
    def __init__(self, x=None, size=None, dtype=None, layout=None, device=None, seed=None, generator=None, sigma_min=None, sigma_max=None):
        seed = seed % (2**32)  # Ensure seed is within the allowed range
        super().__init__(x, size, dtype, layout, device, seed, generator, sigma_min, sigma_max)
        self.noise_generator = SuperSimplexNoise(seed)

    def generate_noise(self, shape: Tuple[int, int, int], scale: float = 1.0) -> Tensor:
        depth, height, width = shape
        coords = np.indices((depth, height, width)).reshape(3, -1).T
        coords = coords / scale
        noise = np.apply_along_axis(lambda p: self.noise_generator.noise3d(p[0], p[1], p[2]), 1, coords)
        noise = noise.reshape(depth, height, width)
        return torch.tensor(noise, device=self.device, dtype=self.dtype)

    def generate_noise_with_alpha(self, shape: Tuple[int, int, int], scale: float = 1.0) -> Tensor:
        depth, height, width = shape
        coords = np.indices((depth, height, width)).reshape(3, -1).T
        coords = coords / scale
        noise = np.apply_along_axis(lambda p: self.noise_generator.noise3d(p[0], p[1], p[2]), 1, coords)
        noise = noise.reshape(depth, height, width)
        alpha = np.apply_along_axis(lambda p: self.noise_generator.noise3d(p[0], p[1], p[2] + 1000), 1, coords)  # Offset z for alpha layer
        alpha = alpha.reshape(depth, height, width)
        combined = np.stack((noise, alpha), axis=-1)  # Stack along the last dimension to keep height and width dimensions intact
        return torch.tensor(combined, device=self.device, dtype=self.dtype)

    def __call__(self, *, scale=1.0, with_alpha=False, **kwargs):
        b, c, h, w = self.size
        if with_alpha:
            depth = 1  # We use depth = 1 because we're adding an alpha layer, not a 3D noise cube
            noise = torch.stack([self.generate_noise_with_alpha((depth, h, w), scale=scale) for _ in range(b * c)], dim=0)
            noise = noise.view(b, c, h, w, 2)  # (batch, channels, height, width, 2 (noise+alpha))
            return noise
        else:
            noise = torch.stack([self.generate_noise((1, h, w), scale=scale) for _ in range(b * c)], dim=0)
            return noise.view(b, c, h, w)

class SuperSimplexNoise:
    STRETCH_CONSTANT_2D = -0.211324865405187
    SQUISH_CONSTANT_2D = 0.366025403784439
    STRETCH_CONSTANT_3D = -1.0 / 6.0
    SQUISH_CONSTANT_3D = 1.0 / 3.0
    NORM_CONSTANT_2D = 47
    NORM_CONSTANT_3D = 103

    def __init__(self, seed: int):
        self.perm = np.arange(256, dtype=int)
        np.random.seed(seed)
        np.random.shuffle(self.perm)
        self.perm = np.stack([self.perm, self.perm]).flatten()

    def extrapolate(self, xsb, ysb, dx, dy):
        index = self.perm[(self.perm[xsb & 0xff] + ysb) & 0xff] & 0x0e
        grad2 = [
            (5, 2), (2, 5), (-5, 2), (-2, 5), (5, -2), (2, -5), (-5, -2), (-2, -5),
            (5, 2), (2, 5), (-5, 2), (-2, 5), (5, -2), (2, -5), (-5, -2), (-2, -5)
        ]
        g = grad2[index >> 1]
        return g[0] * dx + g[1] * dy

    def extrapolate3(self, xsb, ysb, zsb, dx, dy, dz):
        index = self.perm[(self.perm[(self.perm[xsb & 0xff] + ysb) & 0xff] + zsb) & 0xff] & 0x1e
        grad3 = [
            (1, 1, 0), (-1, 1, 0), (1, -1, 0), (-1, -1, 0),
            (1, 0, 1), (-1, 0, 1), (1, 0, -1), (-1, 0, -1),
            (0, 1, 1), (0, -1, 1), (0, 1, -1), (0, -1, -1),
            (1, 1, 0), (-1, 1, 0), (0, -1, 1), (0, -1, -1)
        ]
        g = grad3[index >> 1]
        return g[0] * dx + g[1] * dy + g[2] * dz

    def noise3d(self, x: float, y: float, z: float) -> float:
        stretch_offset = (x + y + z) * self.STRETCH_CONSTANT_3D
        xs = x + stretch_offset
        ys = y + stretch_offset
        zs = z + stretch_offset
        xsb = int(np.floor(xs))
        ysb = int(np.floor(ys))
        zsb = int(np.floor(zs))

        squish_offset = (xsb + ysb + zsb) * self.SQUISH_CONSTANT_3D
        xb = xsb + squish_offset
        yb = ysb + squish_offset
        zb = zsb + squish_offset
        xins = xs - xsb
        yins = ys - ysb
        zins = zs - zsb
        in_sum = xins + yins + zins

        dx0 = x - xb
        dy0 = y - yb
        dz0 = z - zb

        value = 0.0

        # Logic for 3D noise calculation (simplified for brevity)
        # More logic should be added here based on the requirements for generating SuperSimplex noise

        return value / self.NORM_CONSTANT_3D

class SuperSimplexEinsumNoiseGenerator(NoiseGenerator):
    STRETCH_CONSTANT_3D = -1.0 / 6.0
    SQUISH_CONSTANT_3D = 1.0 / 3.0
    NORM_CONSTANT_3D = 103

    def __init__(self, x=None, size=None, dtype=None, layout=None, device=None, seed=None, generator=None, sigma_min=None, sigma_max=None):
        seed = seed % (2**32)  # Ensure seed is within the allowed range
        super().__init__(x, size, dtype, layout, device, seed, generator, sigma_min, sigma_max)
        self.perm = torch.tensor(np.arange(256, dtype=int))
        np.random.seed(self.seed)
        np.random.shuffle(self.perm.numpy())
        self.perm = torch.tensor(np.stack([self.perm.numpy(), self.perm.numpy()]).flatten()).to(device)

    def extrapolate3(self, xsb, ysb, zsb, dx, dy, dz):
        index = self.perm[(self.perm[(self.perm[xsb.cpu() & 0xff] + ysb.cpu()) & 0xff] + zsb.cpu()) & 0xff] & 0x1e
        index = index.to(self.device)
        grad3 = torch.tensor([
            (1, 1, 0), (-1, 1, 0), (1, -1, 0), (-1, -1, 0),
            (1, 0, 1), (-1, 0, 1), (1, 0, -1), (-1, 0, -1),
            (0, 1, 1), (0, -1, 1), (0, 1, -1), (0, -1, -1),
            (1, 1, 0), (-1, 1, 0), (0, -1, 1), (0, -1, -1)
        ], device=self.device, dtype=self.dtype)
        g = grad3[index >> 1]
        return torch.einsum('ij,ij->i', g, torch.stack([dx, dy, dz], dim=-1))

    def super_simplex_noise3d(self, coords):
        stretch_offset = torch.sum(coords, dim=-1, keepdim=True) * self.STRETCH_CONSTANT_3D
        stretched_coords = coords + stretch_offset
        xs, ys, zs = stretched_coords[..., 0], stretched_coords[..., 1], stretched_coords[..., 2]
        xsb, ysb, zsb = torch.floor(xs).long(), torch.floor(ys).long(), torch.floor(zs).long()

        squish_offset = (xsb + ysb + zsb).float() * self.SQUISH_CONSTANT_3D
        xb, yb, zb = xsb.float() + squish_offset, ysb.float() + squish_offset, zsb.float() + squish_offset
        xins, yins, zins = xs - xsb.float(), ys - ysb.float(), zs - zsb.float()
        in_sum = xins + yins + zins

        dx0, dy0, dz0 = coords[..., 0] - xb, coords[..., 1] - yb, coords[..., 2] - zb

        value = torch.zeros_like(dx0)

        if (in_sum <= 1).all():
            dx1, dy1, dz1 = dx0 - 1 - self.SQUISH_CONSTANT_3D, dy0 - 0 - self.SQUISH_CONSTANT_3D, dz0 - 0 - self.SQUISH_CONSTANT_3D
            attn1 = 2 - dx1 * dx1 - dy1 * dy1 - dz1 * dz1
            attn1 = torch.clamp(attn1, min=0)
            attn1 *= attn1
            value += attn1 * attn1 * self.extrapolate3(xsb + 1, ysb, zsb, dx1, dy1, dz1)

            dx2, dy2, dz2 = dx0 - 0 - self.SQUISH_CONSTANT_3D, dy0 - 1 - self.SQUISH_CONSTANT_3D, dz0 - 0 - self.SQUISH_CONSTANT_3D
            attn2 = 2 - dx2 * dx2 - dy2 * dy2 - dz2 * dz2
            attn2 = torch.clamp(attn2, min=0)
            attn2 *= attn2
            value += attn2 * attn2 * self.extrapolate3(xsb, ysb + 1, zsb, dx2, dy2, dz2)

            dx3, dy3, dz3 = dx0 - 0 - self.SQUISH_CONSTANT_3D, dy0 - 0 - self.SQUISH_CONSTANT_3D, dz0 - 1 - self.SQUISH_CONSTANT_3D
            attn3 = 2 - dx3 * dx3 - dy3 * dy3 - dz3 * dz3
            attn3 = torch.clamp(attn3, min=0)
            attn3 *= attn3
            value += attn3 * attn3 * self.extrapolate3(xsb, ysb, zsb + 1, dx3, dy3, dz3)

        else:
            dx1, dy1, dz1 = dx0 - 1 + self.SQUISH_CONSTANT_3D, dy0 - 1 + self.SQUISH_CONSTANT_3D, dz0 - 0 + self.SQUISH_CONSTANT_3D
            attn1 = 2 - dx1 * dx1 - dy1 * dy1 - dz1 * dz1
            attn1 = torch.clamp(attn1, min=0)
            attn1 *= attn1
            value += attn1 * attn1 * self.extrapolate3(xsb + 1, ysb + 1, zsb, dx1, dy1, dz1)

            dx2, dy2, dz2 = dx0 - 1 + self.SQUISH_CONSTANT_3D, dy0 - 0 + self.SQUISH_CONSTANT_3D, dz0 - 1 + self.SQUISH_CONSTANT_3D
            attn2 = 2 - dx2 * dx2 - dy2 * dy2 - dz2 * dz2
            attn2 = torch.clamp(attn2, min=0)
            attn2 *= attn2
            value += attn2 * attn2 * self.extrapolate3(xsb + 1, ysb, zsb + 1, dx2, dy2, dz2)

            dx3, dy3, dz3 = dx0 - 0 + self.SQUISH_CONSTANT_3D, dy0 - 1 + self.SQUISH_CONSTANT_3D, dz0 - 1 + self.SQUISH_CONSTANT_3D
            attn3 = 2 - dx3 * dx3 - dy3 * dy3 - dz3 * dz3
            attn3 = torch.clamp(attn3, min=0)
            attn3 *= attn3
            value += attn3 * attn3 * self.extrapolate3(xsb, ysb + 1, zsb + 1, dx3, dy3, dz3)

        return value / self.NORM_CONSTANT_3D

    def __call__(self, **kwargs):
        self.update(**kwargs)
        b, c, h, w = self.size

        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        # Create a grid of coordinates
        coords = torch.stack(torch.meshgrid(
            torch.linspace(0, 1, h, device=self.device),
            torch.linspace(0, 1, w, device=self.device),
            torch.linspace(0, 1, 1, device=self.device)), -1).reshape(-1, 3)

        # Generate noise using the super simplex noise function
        noise = self.super_simplex_noise3d(coords)
        noise = noise.view(h, w)

        # Expand the noise to match the number of channels
        noise = noise.unsqueeze(0).expand(c, h, w)
        noise = noise.unsqueeze(0).expand(b, c, h, w)

        return noise

class DottedLineNoiseGenerator(NoiseGenerator):
    def __init__(self, x=None, size=None, dtype=None, layout=None, device=None, seed=None, generator=None, sigma_min=None, sigma_max=None, 
                 numLines=60, iterations=500, repulsion_strength=0.1):
        seed = seed % (2**32)  # Ensure seed is within the allowed range
        self.update(numLines=numLines, iterations=iterations, repulsion_strength=repulsion_strength)
        super().__init__(x, size, dtype, layout, device, seed, generator, sigma_min, sigma_max)

    def __call__(self, **kwargs):
        self.update(**kwargs)
        depth, height, width = self.size[-3:]
        iterations = self.iterations
        numLines = self.numLines
        repulsion_strength = self.repulsion_strength

        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        # Initialize a blank canvas for each channel
        canvas = np.zeros((depth, height, width))

        # Store all line paths
        all_line_paths = []

        # Generate multiple lines
        for _ in range(numLines):
            # Initialize the line with a random starting position across the entire canvas
            current_position = np.random.uniform(0, [width, height])
            direction = np.random.uniform(-1, 1, 2)
            direction = direction / np.linalg.norm(direction)

            # Store the line path
            line_path = [current_position.copy()]

            # Iteratively adjust the line
            for _ in range(iterations):
                random_deviation = np.random.uniform(-0.5, 0.5, 2)
                new_direction = direction + random_deviation
                
                # Add repulsion force to maintain distance between lines
                for other_path in all_line_paths:
                    other_position = other_path[-1]
                    dist = np.linalg.norm(current_position - other_position)
                    if dist < 10:  # Adjust the threshold distance as needed
                        repulsion = (current_position - other_position) / dist * repulsion_strength
                        new_direction += repulsion
                
                new_direction = new_direction / np.linalg.norm(new_direction)
                
                step_size = np.random.uniform(2, 7)  # Add randomness to step size
                new_position = current_position + step_size * new_direction
                new_position[0] = np.clip(new_position[0], 0, width - 1)
                new_position[1] = np.clip(new_position[1], 0, height - 1)
                
                line_path.append(new_position.copy())
                current_position = new_position
                direction = new_direction

            # Store the line path for later use
            all_line_paths.append(line_path)

        # Convert all line paths to numpy arrays and plot them on the canvas
        for line_path in all_line_paths:
            line_path = np.array(line_path)
            for point in line_path:
                if not np.isnan(point).any():  # Skip NaN values
                    x, y = int(point[0]), int(point[1])
                    if 0 <= x < width and 0 <= y < height:
                        for c in range(depth):
                            canvas[c, y, x] = 1.0  # Set the pixel value to 1 for line points

        # Convert the canvas to a torch tensor and normalize
        noise = torch.tensor(canvas, dtype=self.dtype, device=self.device)
        noise = noise.unsqueeze(0)  # Add batch dimension
        return noise / noise.std()


class ContinuousLineNoiseGenerator(NoiseGenerator):
    def __init__(self, x=None, size=None, dtype=None, layout=None, device=None, seed=None, generator=None, sigma_min=None, sigma_max=None, 
                 numLines=60, iterations=500, repulsion_strength=0.1):
        self.update(numLines=numLines, iterations=iterations, repulsion_strength=repulsion_strength)
        super().__init__(x, size, dtype, layout, device, seed, generator, sigma_min, sigma_max)

    def __call__(self, **kwargs):
        self.update(**kwargs)
        height, width = self.size[-2], self.size[-1]  # Use height and width from the size
        iterations = self.iterations
        numLines = self.numLines
        repulsion_strength = self.repulsion_strength

        seed = self.seed % (2**32)  # Ensure seed is within the allowed range
        torch.manual_seed(seed)
        np.random.seed(seed)

        channels = self.size[1] if len(self.size) > 1 else 1
        canvas = torch.zeros((channels, height, width), dtype=self.dtype, device=self.device)
        all_line_paths = []

        for _ in range(numLines):
            current_position = np.random.uniform(0, [height, width])
            direction = np.random.uniform(-1, 1, 2)
            if np.linalg.norm(direction) != 0:
                direction = direction / np.linalg.norm(direction)
            else:
                direction = np.random.uniform(-1, 1, 2)
                direction = direction / np.linalg.norm(direction)
            line_path = [current_position.copy()]

            for _ in range(iterations):
                random_deviation = np.random.uniform(-0.5, 0.5, 2)
                new_direction = direction + random_deviation
                if np.linalg.norm(new_direction) != 0:
                    new_direction = new_direction / np.linalg.norm(new_direction)
                for other_path in all_line_paths:
                    other_position = other_path[-1]
                    dist = np.linalg.norm(current_position - other_position)
                    if dist < 10 and dist != 0:  # Avoid division by zero
                        repulsion = (current_position - other_position) / dist * repulsion_strength
                        new_direction += repulsion
                if np.linalg.norm(new_direction) != 0:
                    new_direction = new_direction / np.linalg.norm(new_direction)
                new_position = current_position + 5 * new_direction
                new_position[0] = np.clip(new_position[0], 0, height - 1)
                new_position[1] = np.clip(new_position[1], 0, width - 1)
                line_path.append(new_position.copy())
                current_position = new_position
                direction = new_direction
            all_line_paths.append(line_path)

        for line_path in all_line_paths:
            line_path = np.array(line_path)
            for point in range(len(line_path) - 1):
                if not np.isnan(line_path[point]).any() and not np.isnan(line_path[point + 1]).any():
                    y0, x0 = int(line_path[point][0]), int(line_path[point][1])
                    y1, x1 = int(line_path[point + 1][0]), int(line_path[point + 1][1])
                    if 0 <= x0 < width and 0 <= y0 < height and 0 <= x1 < width and 0 <= y1 < height:
                        for c in range(channels):
                            self.draw_line(canvas[c], y0, x0, y1, x1)

        noise = canvas.unsqueeze(0)
        return noise / noise.std()

    @staticmethod
    def draw_line(canvas, y0, x0, y1, x1):
        """Draw a line on the canvas using Bresenham's line algorithm."""
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy

        while True:
            canvas[y0, x0] = 1.0
            if x0 == x1 and y0 == y1:
                break
            e2 = err * 2
            if e2 > -dy:
                err -= dy
                x0 += sx
            if e2 < dx:
                err += dx
                y0 += sy

def modulated_gaussian_depth(x, y, gaussian_params):
    x0, y0, sigma = gaussian_params
    gaussian = np.exp(-((x - x0)**2 + (y - y0)**2) / (2 * sigma**2))
    sinusoidal = np.sin(x / 10.0) * np.cos(y / 10.0)
    return gaussian * (1 + 0.5 * sinusoidal)  # Modulate Gaussian with sinusoidal

# Function to generate 2D combined modulated frequency noise
def generate_2d_combined_modulated_frequency_noise(rows, cols, base_theta1, base_theta2, variance, depth_function, gaussian_params):
    x = np.arange(rows)
    y = np.arange(cols)
    xv, yv = np.meshgrid(x, y, indexing='ij')
    
    depth_map = depth_function(xv, yv, gaussian_params)
    modulated_theta1 = base_theta1 * np.sin(depth_map * np.pi)
    modulated_theta2 = base_theta2 * np.cos(depth_map * np.pi)
    noise = np.sin(modulated_theta1 * xv + modulated_theta2 * yv) + np.random.normal(0, variance, (rows, cols))
    
    return noise

# Define the new noise generator class
class CombinedModulatedFrequencyNoiseGenerator(NoiseGenerator):
    def __init__(self, x=None, size=None, dtype=None, layout=None, device=None, seed=None, generator=None, sigma_min=None, sigma_max=None, 
                 theta1=2, theta2=1, variance=0.1):
        seed = seed % (2**32)  # Ensure seed is within the allowed range
        self.update(theta1=theta1, theta2=theta2, variance=variance)
        super().__init__(x, size, dtype, layout, device, seed, generator, sigma_min, sigma_max)

    def __call__(self, **kwargs):
        self.update(**kwargs)
        rows, cols = self.size[-2], self.size[-1]  # Assuming the last two dimensions are height and width
        theta1 = self.theta1
        theta2 = self.theta2
        variance = self.variance

        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        # Randomize Gaussian parameters
        x0 = np.random.uniform(0, rows)
        y0 = np.random.uniform(0, cols)
        sigma = np.random.uniform(10, 30)  # Randomize sigma within a range
        gaussian_params = (x0, y0, sigma)

        # Generate the noise
        noise_np = generate_2d_combined_modulated_frequency_noise(rows, cols, theta1, theta2, variance, modulated_gaussian_depth, gaussian_params)

        # Determine the number of channels from the size of the latent image
        channels = self.size[1] if len(self.size) > 1 else 1

        # Expand the noise to match the number of channels
        noise_np = np.expand_dims(noise_np, axis=0)  # Add channel dimension
        noise_np = np.repeat(noise_np, channels, axis=0)

        # Convert the noise to a torch tensor and normalize
        noise = torch.tensor(noise_np, dtype=self.dtype, device=self.device)
        noise = noise.unsqueeze(0)  # Add batch dimension

        return noise / noise.std()


# Ensure the seed is within the valid range
#if not 0 <= seed < 2**32:
#    seed = 42  # or any other default value within the valid range

from typing import Callable
from scipy.interpolate import splprep, splev

class SingleLineNoiseGenerator(NoiseGenerator):
    def __init__(self, x=None, size=None, dtype=None, layout=None, device=None, seed=None, generator=None, sigma_min=None, sigma_max=None, 
                 iterations=100, repulsion_strength=0.05, gravity_strength=0.01):
        seed = seed % (2**32)  # Ensure seed is within the allowed range
        self.update(iterations=iterations, repulsion_strength=repulsion_strength, gravity_strength=gravity_strength)
        super().__init__(x, size, dtype, layout, device, seed, generator, sigma_min, sigma_max)

    def reflect(self, position, direction, size):
        if position[0] < 0 or position[0] >= size[0]:
            direction[0] *= -1
            position[0] = np.clip(position[0], 0, size[0] - 1)
        if position[1] < 0 or position[1] >= size[1]:
            direction[1] *= -1
            position[1] = np.clip(position[1], 0, size[1] - 1)
        return direction

    def __call__(self, **kwargs):
        self.update(**kwargs)
        size = self.size[-2:]  # Assuming dimensions for width and height
        iterations = self.iterations
        repulsion_strength = self.repulsion_strength
        gravity_strength = self.gravity_strength

        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        # Determine the number of channels from the size of the latent image
        channels = self.size[1] if len(self.size) > 1 else 1

        # Initialize a blank canvas for each channel
        canvas = np.zeros((channels, size[0], size[1]))

        # 1. Initialization
        current_position = np.random.uniform(0, size, 2)
        direction = np.random.uniform(-1, 1, 2)
        direction /= np.linalg.norm(direction)

        # Store the line path
        line_path = [current_position.copy()]

        # Create a grid to keep track of density
        density_grid = np.zeros((size[0] // 10, size[1] // 10), dtype=int)

        # Iteratively adjust the line
        for i in range(iterations):
            # 2. Iterative Line Path Extension
            random_deviation = np.random.uniform(-0.5, 0.5, 2)
            direction = (direction + random_deviation)
            direction /= np.linalg.norm(direction)

            # Add repulsion force
            for other_pos in line_path:
                dist = np.linalg.norm(current_position - other_pos)
                if dist > 0 and dist < 10:  # Avoid division by zero and apply threshold distance for repulsion
                    repulsion = (current_position - other_pos) / dist * repulsion_strength
                    direction += repulsion

            direction /= np.linalg.norm(direction)  # Re-normalize direction

            # Update position
            new_position = current_position + 5 * direction

            # 3. Reflection off Borders
            direction = self.reflect(new_position, direction, size)

            # Ensure continuity by connecting points closely
            step_size = 1.0  # Adjust the step size to ensure points are connected
            steps = int(np.ceil(np.linalg.norm(new_position - current_position) / step_size))
            if steps > 1:
                for step in range(1, steps):
                    interp_position = current_position + (new_position - current_position) * (step / steps)
                    line_path.append(interp_position.copy())

            # 4. Density Computation
            grid_pos = np.round(new_position / 10).astype(int)
            # Ensure the indices are within valid bounds
            grid_pos = np.clip(grid_pos, 0, [size[0] // 10 - 1, size[1] // 10 - 1])
            density = density_grid[grid_pos[0], grid_pos[1]]
            density_grid[grid_pos[0], grid_pos[1]] += 1

            # 5. Gravitational Force
            if density > 0:  # Example threshold
                g = (np.array(size) / 2 - current_position)
                norm_g = np.linalg.norm(g)
                if norm_g > 0:  # Avoid division by zero
                    g /= norm_g
                    direction += g * gravity_strength

            # 6. Pulling Back to Start
            g = (line_path[0] - current_position)
            norm_g = np.linalg.norm(g)
            if norm_g > 0:  # Avoid division by zero
                g /= norm_g
                direction += g * gravity_strength * (i / iterations)

            # Update position and direction
            current_position = new_position
            line_path.append(current_position.copy())

        # 7. Closing the Loop
        line_path.append(line_path[0])

        # 8. Spline Conversion
        line_path = np.array(line_path)
        try:
            tck, u = splprep([line_path[:, 0], line_path[:, 1]], s=0)
        except Exception as e:
            print(f"Error in splprep: {e}")
            print(f"Line path array: {line_path}")
            raise e

        unew = np.linspace(0, 1.0, num=iterations * 10)
        out = splev(unew, tck)

        for point in zip(out[0], out[1]):
            x, y = int(np.nan_to_num(point[0])), int(np.nan_to_num(point[1]))
            if 0 <= x < size[0] and 0 <= y < size[1]:
                for c in range(channels):
                    canvas[c, y, x] = 1  # Set the pixel value to 1 for line points

        # Convert the canvas to a torch tensor and normalize
        noise = torch.tensor(canvas, dtype=self.dtype, device=self.device)
        noise = noise.unsqueeze(0)  # Add batch dimension
        return noise / noise.std()


# Factory function for noise generators
def noise_generator_factory(cls, **fixed_params):
    def create_instance(**kwargs):
        params = {**fixed_params, **kwargs}
        return cls(**params)
    return create_instance
    
# Wavelet modes for boundary handling
wavelet_modes = [
    'zero', 'constant', 'symmetric', 'reflect', 'periodic', 'smooth', 'periodization'
]

# List of commonly used and widely recognized wavelets
common_wavelets = [
    'haar', 'db2', 'sym3', 'coif1', 'bior2.2', 'rbio2.2', 'dmey', 'gaus1', 'mexh', 'morl'
]

# Create a dictionary of all common wavelet noise generators
base_noise_generators = {wavelet: type(f"{wavelet}NoiseGenerator", (WaveletNoiseGenerator,), {'__init__': functools.partialmethod(WaveletNoiseGenerator.__init__, wavelet=wavelet)}) for wavelet in common_wavelets}

# Add wavelet noise generators with different modes for common discrete wavelets
wavelet_noise_generators = {
    "wavelet-" + wavelet + "-" + mode: noise_generator_factory(WaveletNoiseGenerator, wavelet=wavelet, mode=mode)
    for wavelet in pywt.wavelist(kind='discrete') if wavelet in common_wavelets
    for mode in wavelet_modes
}

# Base noise generators without specific modes
base_noise_generators = {
    "fractal": FractalNoiseGenerator,
    "pyramid": PyramidNoiseGenerator,
    "hires-pyramid": HiresPyramidNoiseGenerator,
    "gaussian": GaussianNoiseGenerator,
    "uniform": UniformNoiseGenerator,
    "brownian": BrownianNoiseGenerator,
    "laplacian": LaplacianNoiseGenerator,
    "studentt": StudentTNoiseGenerator,
    "wavelet": WaveletNoiseGenerator,
    "perlin": PerlinNoiseGenerator,
    "opensimplex": OpenSimplexNoiseGenerator,
    #"supersimplex": SuperSimplexNoiseGenerator,
    "supersimplexeinsum": SuperSimplexEinsumNoiseGenerator,
    "dotted_line": DottedLineNoiseGenerator,
    "continuous_line": ContinuousLineNoiseGenerator,
    "modulated_frequency": CombinedModulatedFrequencyNoiseGenerator,
    "single_line": SingleLineNoiseGenerator,

}

# Add additional modes for other noise generators if needed
pyramid_modes = ['linear', 'bilinear', 'bicubic', 'trilinear', 'area', 'nearest', 'nearest-exact']
pyramid_noise_generators = {
    "pyramid-" + mode: noise_generator_factory(PyramidNoiseGenerator, mode=mode) for mode in pyramid_modes
}

hires_pyramid_noise_generators = {
    "hires-pyramid-" + mode: noise_generator_factory(HiresPyramidNoiseGenerator, mode=mode) for mode in pyramid_modes
}

# Combine all noise generators into one dictionary
NOISE_GENERATOR_CLASSES = {
    **base_noise_generators,
    **wavelet_noise_generators,
    **pyramid_noise_generators,
    **hires_pyramid_noise_generators
}

# Generate NOISE_GENERATOR_NAMES
NOISE_GENERATOR_NAMES = tuple(NOISE_GENERATOR_CLASSES.keys())

@cast_fp64
def prepare_noise(latent_image, seed, noise_type, noise_inds=None, alpha=1.0, k=1.0, use_alpha=False, **noise_params):
    device = latent_image.device
    noise_func = NOISE_GENERATOR_CLASSES.get(noise_type)(x=latent_image, seed=seed, sigma_min=0.0291675, sigma_max=14.614642, **noise_params)
    
    if noise_type == "fractal":
        noise_func.alpha = alpha
        noise_func.k = k

    if noise_inds is None:
        if use_alpha:
            return noise_func(sigma=14.614642, sigma_next=0.0291675, with_alpha=True).to(device)
        else:
            return noise_func(sigma=14.614642, sigma_next=0.0291675).to(device)

    unique_inds, inverse = np.unique(noise_inds, return_inverse=True)
    noises = []
    for i in range(unique_inds[-1] + 1):
        if use_alpha:
            noise = noise_func(size=[1] + list(latent_image.size())[1:] + [2], dtype=latent_image.dtype, layout=latent_image.layout, device=latent_image.device, with_alpha=True).to(device)
        else:
            noise = noise_func(size=[1] + list(latent_image.size())[1:], dtype=latent_image.dtype, layout=latent_image.layout, device=latent_image.device).to(device)
        if i in unique_inds:
            noises.append(noise)
    noises = [noises[i] for i in inverse]
    noises = torch.cat(noises, axis=0).to(device)
    return noises

