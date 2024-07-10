from . import cas
from . import cas_extra
from . import latent_disfigurement
from .latent_blender import LatentBlender  # Import the LatentBlender class from latent_blender.py
from . import activators  # Import the activators.py file from this directory
from . import schedulers
from . import attenuator
cas_extra.add_samplers()
cas_extra.add_schedulers()

NODE_CLASS_MAPPINGS = {
    'CustomAdvancedSampler': cas.CustomAdvancedSampler,
    'BasicCFGGuider': cas.BasicCFGGuider,
    
    'SamplerWeightedCFGPP': cas_extra.SamplerWeightedCFGPP,
    'SamplerDynamicCFGPP': cas_extra.SamplerDynamicCFGPP,
    'SamplerEulerAttnCFGPP': cas_extra.SamplerEulerAttnCFGPP,
    'SamplerStepSizeControlCFGPP': cas_extra.SamplerStepSizeControlCFGPP,
    'SamplerHeunCFGPP': cas_extra.SamplerHeunCFGPP,
    'SamplerDPMCFGPP': cas_extra.SamplerDPMCFGPP,
    'SamplerCustomLCMCFGPP': cas_extra.SamplerCustomLCMCFGPP,
    'SamplerCustomX0CFGPP': cas_extra.SamplerCustomX0CFGPP,
    'SamplerCustomModelSamplingDiscreteDistilledCFGPP': cas_extra.SamplerCustomModelSamplingDiscreteDistilledCFGPP,
    'SamplerCustomModelSamplingDiscreteDistilledAncestralCFGPP': cas_extra.SamplerCustomModelSamplingDiscreteDistilledAncestralCFGPP,
    'SamplerCustomX0AncestralCFGPP': cas_extra.SamplerCustomX0AncestralCFGPP,
    'SamplerEulerStepControlAncestralCFGPP': cas_extra.SamplerEulerStepControlAncestralCFGPP,
    'EPSCFGPPScheduler': cas_extra.EPSCFGPPScheduler,
    'SamplerLCMUpscaleWCFGPP': cas_extra.SamplerLCMUpscaleWCFGPP,
    
    'LatentConvolution': latent_disfigurement.LatentConvolution,
    'LatentActivation': latent_disfigurement.LatentActivation,
    'LatentMath': latent_disfigurement.LatentMath,
    'LatentWarpPresets': latent_disfigurement.LatentWarpPresets,
    'LatentChannelPresets': latent_disfigurement.LatentChannelPresets,
    'LatentValuePresets': latent_disfigurement.LatentValuePresets,
    'LatentFrequencyPresets': latent_disfigurement.LatentFrequencyPresets,
    'LatentNoisePresets': latent_disfigurement.LatentNoisePresets,
    'LatentMathFormulaBuilder': latent_disfigurement.LatentMathFormulaBuilder,
    'LatentFFT': latent_disfigurement.LatentFFT,
    'LatentGlitch': latent_disfigurement.LatentGlitch,
    'LatentTwist': latent_disfigurement.LatentTwist,
    'LatentMosaic': latent_disfigurement.LatentMosaic,
    'LatentPixelSort': latent_disfigurement.LatentPixelSort,
    
    'LatentSelfAttention': cas.LatentSelfAttention,
    'AttentionToSigmas': cas.AttentionToSigmas,
    # Activators    
    'Threshold': activators.Threshold,
    'ReLU': activators.ReLU,
    'RReLU': activators.RReLU,
    'Hardtanh': activators.Hardtanh,
    'ReLU6': activators.ReLU6,
    'Hardsigmoid': activators.Hardsigmoid,
    'Tanh': activators.Tanh,
    'SiLU': activators.SiLU,
    'Mish': activators.Mish,
    'Hardswish': activators.Hardswish,
    'ELU': activators.ELU,
    'CELU': activators.CELU,
    'SELU': activators.SELU,
    'GLU': activators.GLU,
    'GELU': activators.GELU,
    'Hardshrink': activators.Hardshrink,
    'LeakyReLU': activators.LeakyReLU,
    'LogSigmoid': activators.LogSigmoid,
    'Softplus': activators.Softplus,
    'Softshrink': activators.Softshrink,
    'PReLU': activators.PReLU,
    'Softsign': activators.Softsign,
    'Tanhshrink': activators.Tanhshrink,
    'Softmin': activators.Softmin,
    'Softmax': activators.Softmax,
    'Softmax2D': activators.Softmax2D,
    'LogSoftmax': activators.LogSoftmax,
    # Schedulers    
    'SoftmaxScheduler': schedulers.SoftmaxScheduler,
    'HardTanhScheduler': schedulers.HardTanhScheduler,
    # Attenuator
    'AttenuatorNode': attenuator.AttenuatorNode,

}    



NODE_DISPLAY_NAME_MAPPINGS = {
    'CustomAdvancedSampler': 'CAS',
    'BasicCFGGuider': 'CAS Guider',
    
    'SamplerWeightedCFGPP': 'Weighted CFGPP',
    'SamplerDynamicCFGPP': 'Dynamic CFGPP',
    'SamplerEulerAttnCFGPP': 'Euler Attn CFGPP',
    'SamplerStepSizeControlCFGPP': 'Step Size Control CFGPP',
    'SamplerHeunCFGPP': 'Heun CFGPP',
    'SamplerDPMCFGPP': 'DPM CFGPP',
    'SamplerCustomLCMCFGPP': 'Custom LCM CFGPP',
    'SamplerCustomX0CFGPP': 'Custom X0 CFGPP',
    'SamplerCustomModelSamplingDiscreteDistilledCFGPP': 'Custom Model Sampling Discrete Distilled CFGPP',
    'SamplerCustomModelSamplingDiscreteDistilledAncestralCFGPP': 'Custom Model Sampling Discrete Distilled Ancestral CFGPP',
    'SamplerCustomX0AncestralCFGPP': 'Custom X0 Ancestral CFGPP',
    'SamplerEulerStepControlAncestralCFGPP': 'Euler Step Control Ancestral CFGPP',
    'EPSCFGPPScheduler': 'EPS CFGPP Scheduler',
    'SamplerLCMUpscaleWCFGPP': 'LCM Upscale W CFGPP',
    
    'LatentConvolution': 'Latent Convolution',
    'LatentActivation': 'Latent Activation',
    'LatentMath': 'Latent Math',
    'LatentWarpPresets': 'Latent Warp Presets',
    'LatentChannelPresets': 'Latent Channel Presets',
    'LatentValuePresets': 'Latent Value Presets',
    'LatentFrequencyPresets': 'Latent Frequency Presets',
    'LatentNoisePresets': 'Latent Noise Presets',
    'LatentMathFormulaBuilder': 'Latent Math Formula Builder',
    'LatentFFT': 'Latent FFT',
    'LatentGlitch': 'Latent Glitch',
    'LatentTwist': 'Latent Twist',
    'LatentMosaic': 'Latent Mosaic',
    'LatentPixelSort': 'Latent Pixel Sort',
    
    'LatentSelfAttention': 'Latent Self Attention',
    'AttentionToSigmas': 'Attention To Sigmas',
    # Activators    
    'Threshold': 'Threshold',
    'ReLU': 'ReLU',
    'RReLU': 'RReLU',
    'Hardtanh': 'Hardtanh',
    'ReLU6': 'ReLU6',
    'Hardsigmoid': 'Hardsigmoid',
    'Tanh': 'Tanh',
    'SiLU': 'SiLU',
    'Mish': 'Mish',
    'Hardswish': 'Hardswish',
    'ELU': 'ELU',
    'CELU': 'CELU',
    'SELU': 'SELU',
    'GLU': 'GLU',
    'GELU': 'GELU',
    'Hardshrink': 'Hardshrink',
    'LeakyReLU': 'LeakyReLU',
    'LogSigmoid': 'LogSigmoid',
    'Softplus': 'Softplus',
    'Softshrink': 'Softshrink',
    'PReLU': 'PReLU',
    'Softsign': 'Softsign',
    'Tanhshrink': 'Tanhshrink',
    'Softmin': 'Softmin',
    'Softmax': 'Softmax',
    'Softmax2D': 'Softmax2D',
    'LogSoftmax': 'LogSoftmax',
    # Schedulers    
    'SoftmaxScheduler': 'SoftmaxScheduler',
    'HardTanhScheduler': 'HardTanhScheduler',
    # Attenuator
    'AttenuatorNode': 'Attenuator Node',
}    

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
