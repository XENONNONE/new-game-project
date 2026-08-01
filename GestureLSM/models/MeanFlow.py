import logging
from functools import partial
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.config import instantiate_from_config
from models.utils.utils import count_parameters

logger = logging.getLogger(__name__)


def print_memory_usage(location: str, device: torch.device = None):
    """Print current GPU memory usage."""
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if device.type == 'cuda':
        allocated = torch.cuda.memory_allocated(device) / 1024**3  # GB
        reserved = torch.cuda.memory_reserved(device) / 1024**3    # GB
        max_allocated = torch.cuda.max_memory_allocated(device) / 1024**3  # GB
        print(f"[{location}] GPU Memory - Allocated: {allocated:.3f}GB, Reserved: {reserved:.3f}GB, Max: {max_allocated:.3f}GB")
    else:
        print(f"[{location}] Using CPU device")


def clear_gpu_cache():
    """Clear GPU cache to free memory."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print("GPU cache cleared")


def find_attention_modules(module, attention_modules=None):
    """Recursively find all attention modules in a model."""
    if attention_modules is None:
        attention_modules = []
    
    for name, child in module.named_children():
        if hasattr(child, 'set_force_no_fused_attn'):
            attention_modules.append(child)
        find_attention_modules(child, attention_modules)
    
    return attention_modules


def mean_flat(x):
    """
    Take the mean over all non-batch dimensions.
    """
    return torch.mean(x, dim=list(range(1, len(x.size()))))


def reshape_coefs(t):
    """Reshape coefficients for broadcasting."""
    return t.reshape((t.shape[0], 1, 1, 1))


class GestureMF(torch.nn.Module):
    """
    MeanFlow loss calculator for gesture generation, designed to be similar to GestureLSM.
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        
        # Initialize model components
        self.modality_encoder = instantiate_from_config(cfg.model.modality_encoder)
        self.denoiser = instantiate_from_config(cfg.model.denoiser)

        # Model hyperparameters
        self.do_classifier_free_guidance = cfg.model.do_classifier_free_guidance
        self.guidance_scale = cfg.model.guidance_scale
        self.num_inference_steps = cfg.model.n_steps
        
        # meanflow args
        self.weighting = cfg.model.weighting
        self.path_type = cfg.model.path_type
        self.noise_dist = cfg.model.noise_dist
        self.data_proportion = cfg.model.data_proportion
        
        self.cfg_min_t = cfg.model.cfg_min_t
        self.cfg_max_t = cfg.model.cfg_max_t
        
        self.time_mu = cfg.model.time_mu
        self.time_sigma = cfg.model.time_sigma
        
        self.time_min = cfg.model.time_min
        self.time_max = cfg.model.time_max
        
        # CFG parameters
        self.cfg_omega = cfg.model.get("cfg_omega", 0.5)
        self.cfg_kappa = cfg.model.get("cfg_kappa", 0.5)
        self.adaptive_p = cfg.model.get("adaptive_p", 0.5)

        
        self.num_joints = self.denoiser.joint_num
        
        self.seq_len = self.denoiser.seq_len
        self.input_dim = self.denoiser.input_dim
        self.latent_dim = self.denoiser.latent_dim
        
        # Flow matching mode: 'v' for velocity prediction, 'x1' for direct position prediction
        self.flow_mode = cfg.model.get("flow_mode", "v")
        assert self.flow_mode in [
            "v",
            "x1",
        ], f"Flow mode must be 'v' or 'x1', got {self.flow_mode}"
        logger.info(f"Using flow mode: {self.flow_mode}")

        # Set up JVP function for computing derivatives
        self.jvp_fn = torch.func.jvp
    
    def summarize_parameters(self) -> None:
        logger.info(f'Denoiser: {count_parameters(self.denoiser)}M')
        logger.info(f'Encoder: {count_parameters(self.modality_encoder)}M')

    def _disable_fused_attn_for_jvp(self):
        """Temporarily disable fused attention to avoid forward AD issues."""
        # Find all attention modules in the denoiser
        attention_modules = find_attention_modules(self.denoiser)
        
        if attention_modules:
            # Disable fused attention for all found modules
            for attn_module in attention_modules:
                attn_module.set_force_no_fused_attn(True)
            return attention_modules, False
        else:
            # Fallback: check if denoiser itself has the method
            if hasattr(self.denoiser, 'set_force_no_fused_attn'):
                self.denoiser.set_force_no_fused_attn(True)
                return self.denoiser, True
        return None, None
        
    def _restore_fused_attn(self, original_state, is_simple):
        """Restore original fused attention setting."""
        if original_state is None:
            return
        if is_simple:
            # Restore for denoiser itself
            if hasattr(original_state, 'set_force_no_fused_attn'):
                original_state.set_force_no_fused_attn(False)
        else:
            # Restore for each block
            for attn in original_state:
                if hasattr(attn, 'set_force_no_fused_attn'):
                    attn.set_force_no_fused_attn(False)

    def _logit_normal_dist(self, bz, device):
        rnd_normal = torch.randn((bz, 1, 1, 1), device=device)
        return torch.sigmoid(rnd_normal * self.time_sigma + self.time_mu)

    def _uniform_dist(self, bz, device):
        return torch.rand((bz, 1, 1, 1), device=device)
    
    
    def interpolate(self, t):
        """Define interpolation function"""
        if self.path_type == "linear":
            alpha_t = 1 - t
            sigma_t = t
            d_alpha_t = -1
            d_sigma_t =  1
        elif self.path_type == "cosine":
            alpha_t = torch.cos(t * np.pi / 2)
            sigma_t = torch.sin(t * np.pi / 2)
            d_alpha_t = -np.pi / 2 * torch.sin(t * np.pi / 2)
            d_sigma_t =  np.pi / 2 * torch.cos(t * np.pi / 2)
        else:
            raise NotImplementedError()

        return alpha_t, sigma_t, d_alpha_t, d_sigma_t

    def sample_tr(self, bz, device):
        """Sample time parameters t and r."""
        if self.noise_dist == "logit_normal":
            t = self._logit_normal_dist(bz, device)
            r = self._logit_normal_dist(bz, device)
        elif self.noise_dist == "uniform":
            t = self._uniform_dist(bz, device)
            r = self._uniform_dist(bz, device)
        else:
            raise ValueError(f"Unknown noise distribution: {self.noise_dist}")

        t, r = torch.maximum(t, r), torch.minimum(t, r)
        data_size = int(bz * self.data_proportion)
        zero_mask = (torch.arange(bz, device=t.device) < data_size).view(bz, 1, 1, 1)
        r = torch.where(zero_mask, t, r)
        return t, r
    
    def apply_classifier_free_guidance(self, x, timesteps, seed, at_feat, cond_time=None, guidance_scale=1.0):
        """
        Apply classifier-free guidance by running both conditional and unconditional predictions.
        
        Args:
            x: Input tensor
            timesteps: Timestep tensor
            seed: Seed vectors
            at_feat: Audio features
            cond_time: Conditional time tensor
            guidance_scale: Guidance scale (1.0 means no guidance)
            
        Returns:
            Guided output tensor
        """
        if guidance_scale <= 1.0:
            # No guidance needed, run normal forward pass
            return self.denoiser(
                x=x,
                timesteps=timesteps,
                seed=seed,
                at_feat=at_feat,
                cond_time=cond_time,
            )
        
        # Double the batch for classifier free guidance
        x_doubled = torch.cat([x] * 2, dim=0)
        seed_doubled = torch.cat([seed] * 2, dim=0)
        
        # Properly expand timesteps to match doubled batch size
        batch_size = x.shape[0]
        timesteps_doubled = timesteps.expand(batch_size * 2)
        
        if cond_time is not None:
            cond_time_doubled = cond_time.expand(batch_size * 2)
        else:
            cond_time_doubled = None
        
        # Create conditional and unconditional audio features
        batch_size = at_feat.shape[0]
        null_cond_embed = self.denoiser.null_cond_embed.to(at_feat.dtype)
        at_feat_uncond = null_cond_embed.unsqueeze(0).expand(batch_size, -1, -1)
        at_feat_combined = torch.cat([at_feat, at_feat_uncond], dim=0)
        
        # Run both conditional and unconditional predictions
        output = self.denoiser(
            x=x_doubled,
            timesteps=timesteps_doubled,
            seed=seed_doubled,
            at_feat=at_feat_combined,
            cond_time=cond_time_doubled,
        )
        
        # Split predictions and apply guidance
        pred_cond, pred_uncond = output.chunk(2, dim=0)
        guided_output = pred_uncond + guidance_scale * (pred_cond - pred_uncond)
        
        return guided_output
    
    
    def apply_conditional_dropout(self, at_feat, cond_drop_prob=0.1):
        """
        Apply conditional dropout during training to simulate classifier-free guidance.
        
        Args:
            at_feat: Audio features tensor
            cond_drop_prob: Probability of dropping conditions (default 0.1)
            
        Returns:
            Modified audio features with some conditions replaced by null embeddings
        """
        batch_size = at_feat.shape[0]
        
        # Create dropout mask
        keep_mask = torch.rand(batch_size, device=at_feat.device) > cond_drop_prob
        
        # Create null condition embeddings
        null_cond_embed = self.denoiser.null_cond_embed.to(at_feat.dtype)
        
        # Apply dropout: replace dropped conditions with null embeddings
        at_feat_dropped = at_feat.clone()
        at_feat_dropped[~keep_mask] = null_cond_embed.unsqueeze(0).expand((~keep_mask).sum(), -1, -1)
        
        return at_feat_dropped, keep_mask

    
    @torch.no_grad()
    def forward(self, condition_dict: Dict[str, Dict]) -> Dict[str, torch.Tensor]:
        """Forward pass for inference.
        
        Args:
            condition_dict: Dictionary containing input conditions including audio, word tokens,
                          and other features
        
        Returns:
            Dictionary containing generated latents
        """
        
        # Extract input features
        audio = condition_dict['y']['audio_onset']
        word_tokens = condition_dict['y']['word']
        ids = condition_dict['y']['id']
        seed_vectors = condition_dict['y']['seed']
        style_features = condition_dict['y']['style_feature']
        if 'wavlm' in condition_dict['y']:
            wavlm_features = condition_dict['y']['wavlm']
        else:
            wavlm_features = None
        
        return_dict = {}
        return_dict['seed'] = seed_vectors
        
        # Encode input modalities
        audio_features = self.modality_encoder(audio, word_tokens, wavlm_features)
        return_dict['at_feat'] = audio_features

        # Initialize generation
        batch_size = audio_features.shape[0]
        latent_shape = (batch_size, self.input_dim * self.num_joints, 1, self.seq_len)

        # Sampling parameters
        x_t = torch.randn(latent_shape, device=audio_features.device)

        return_dict['init_noise'] = x_t
        
        
        if self.num_inference_steps == 1:
            cond_time = torch.zeros(1, device=audio_features.device)
            timestep = torch.ones(1, device=audio_features.device)
            
            model_output = self.apply_classifier_free_guidance(
                x=x_t,
                timesteps=timestep,
                seed=seed_vectors,
                at_feat=audio_features,
                cond_time=cond_time,
                guidance_scale=self.guidance_scale
            )
            
            # one-step meanflow
            x_t = x_t - model_output
            
        else:
            epsilon = 1e-8
            
            timesteps = torch.linspace(1 - epsilon, 0, self.num_inference_steps + 1).to(audio_features.device)
            
            # Generation loop
            for step in range(len(timesteps) - 1):
                current_t = timesteps[step].unsqueeze(0)
                current_r = timesteps[step + 1].unsqueeze(0)
                
                model_output = self.apply_classifier_free_guidance(
                    x=x_t,
                    timesteps=current_t,
                    cond_time=current_r,
                    seed=seed_vectors,
                    at_feat=audio_features,
                    guidance_scale=self.guidance_scale
                )
                
                # only support v-prediction mode for now
                # Update x_t using the predicted meanflow velocity field
                x_t = x_t - (current_t - current_r) * model_output
                
                    
        return_dict['latents'] = x_t
        return return_dict