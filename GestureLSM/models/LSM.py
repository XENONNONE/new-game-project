import time
import inspect
import logging
from typing import Optional

import scipy.stats as stats
import tqdm
import numpy as np
from omegaconf import DictConfig
from typing import Dict
import math
import torch
import torch.distributions as dist
import torch.nn as nn

import torch
import torch.nn.functional as F
from models.config import instantiate_from_config
from models.utils.utils import count_parameters, extract_into_tensor, sum_flat

logger = logging.getLogger(__name__)

def exponential_pdf(x, a):
    C = a / (np.exp(a) - 1)
    return C * np.exp(a * x)

# Define a custom probability density function
class ExponentialPDF(stats.rv_continuous):
    def _pdf(self, x, a):
        return exponential_pdf(x, a)

def sample_t(exponential_pdf, num_samples, a=2):
    t = exponential_pdf.rvs(size=num_samples, a=a)
    t = torch.from_numpy(t).float()
    t = torch.cat([t, 1 - t], dim=0)
    t = t[torch.randperm(t.shape[0])]
    t = t[:num_samples]

    t_min = 1e-5
    t_max = 1-1e-5

    # Scale t to [t_min, t_max]
    t = t * (t_max - t_min) + t_min
    return t

def sample_beta_distribution(num_samples, alpha=2, beta=0.8, t_min=1e-5, t_max=1-1e-5):
    """
    Samples from a Beta distribution with the specified parameters.
    
    Args:
        num_samples (int): Number of samples to generate.
        alpha (float): Alpha parameter of the Beta distribution (shape1).
        beta (float): Beta parameter of the Beta distribution (shape2).
        t_min (float): Minimum value for scaling the samples (default is near 0).
        t_max (float): Maximum value for scaling the samples (default is near 1).
        
    Returns:
        torch.Tensor: Tensor of sampled values.
    """
    # Define the Beta distribution
    beta_dist = dist.Beta(alpha, beta)
    
    # Sample values from the Beta distribution
    samples = beta_dist.sample((num_samples,))
    
    # Scale the samples to the range [t_min, t_max]
    scaled_samples = samples * (t_max - t_min) + t_min
    
    return scaled_samples

def sample_t_fast(num_samples, a=2, t_min=1e-5, t_max=1-1e-5):
    # Direct inverse sampling for exponential distribution
    C = a / (np.exp(a) - 1)
    
    # Generate uniform samples
    u = torch.rand(num_samples * 2)
    
    # Inverse transform sampling formula for the exponential PDF
    # F^(-1)(u) = (1/a) * ln(1 + u*(exp(a) - 1))
    t = (1/a) * torch.log(1 + u * (np.exp(a) - 1))
    
    # Combine t and 1-t
    t = torch.cat([t, 1 - t])
    
    # Random permutation and slice
    t = t[torch.randperm(t.shape[0])][:num_samples]
    
    # Scale to [t_min, t_max]
    t = t * (t_max - t_min) + t_min
    
    return t

def sample_cosmap(num_samples, t_min=1e-5, t_max=1-1e-5, device='cpu'):
    """
    CosMap sampling.
    Args:
        num_samples: Number of samples to generate
        t_min, t_max: Range limits to avoid numerical issues
    """
    # Generate uniform samples
    u = torch.rand(num_samples, device=device)
    
    # Apply the cosine mapping
    pi_half = torch.pi / 2
    t = 1 - 1 / (torch.tan(pi_half * u) + 1)
    
    # Scale to [t_min, t_max]
    t = t * (t_max - t_min) + t_min
    
    return t

def reshape_coefs(t):
    return t.reshape((t.shape[0], 1, 1, 1))

class GestureLSM(torch.nn.Module):
    def __init__(self, cfg) -> None:
        super().__init__()
        self.cfg = cfg

        # Initialize model components
        self.modality_encoder = instantiate_from_config(cfg.model.modality_encoder)
        self.denoiser = instantiate_from_config(cfg.model.denoiser)

        # Model hyperparameters
        self.do_classifier_free_guidance = cfg.model.do_classifier_free_guidance
        self.guidance_scale = cfg.model.guidance_scale
        self.num_inference_steps = cfg.model.n_steps

        # Loss functions
        self.smooth_l1_loss = torch.nn.SmoothL1Loss(reduction='none')
        
        self.num_joints = self.denoiser.joint_num
        
        self.seq_len = self.denoiser.seq_len
        self.input_dim = self.denoiser.input_dim
        
        # Flow matching mode: 'v' for velocity prediction, 'x1' for direct position prediction
        self.flow_mode = cfg.model.get("flow_mode", "v")
        assert self.flow_mode in [
            "v",
            "x1",
        ], f"Flow mode must be 'v' or 'x1', got {self.flow_mode}"
        logger.info(f"Using flow mode: {self.flow_mode}")
        


    def summarize_parameters(self) -> None:
        logger.info(f'Denoiser: {count_parameters(self.denoiser)}M')
        logger.info(f'Encoder: {count_parameters(self.modality_encoder)}M')
    
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
        at_feat_doubled = torch.cat([at_feat] * 2, dim=0)
        
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
        
        return at_feat_dropped
    
    def apply_force_cfg(self, at_feat, force_cfg):
        """
        Apply forced conditional dropout based on the force_cfg mask.
        
        Args:
            at_feat: Audio features tensor
            force_cfg: Boolean mask indicating which samples should use null conditions
            
        Returns:
            Modified audio features with forced conditions replaced by null embeddings
        """
        batch_size = at_feat.shape[0]
        
        # Create null condition embeddings
        null_cond_embed = self.denoiser.null_cond_embed.to(at_feat.dtype)
        
        # Apply forced dropout: replace forced conditions with null embeddings
        at_feat_forced = at_feat.clone()
        force_cfg_tensor = torch.tensor(force_cfg, device=at_feat.device)
        at_feat_forced[force_cfg_tensor] = null_cond_embed.unsqueeze(0).expand(force_cfg_tensor.sum(), -1, -1)
        
        return at_feat_forced
    
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
        
        epsilon = 1e-8
        delta_t = torch.tensor(1 / self.num_inference_steps).to(audio_features.device)
        timesteps = torch.linspace(epsilon, 1 - epsilon, self.num_inference_steps + 1).to(audio_features.device)
        
        # Generation loop
        for step in range(1, len(timesteps)):
            current_t = timesteps[step - 1].unsqueeze(0)
            current_delta = delta_t.unsqueeze(0)
            
            with torch.no_grad():
                model_output = self.apply_classifier_free_guidance(
                    x=x_t,
                    timesteps=current_t,
                    seed=seed_vectors,
                    at_feat=audio_features,
                    cond_time=current_delta,
                    guidance_scale=self.guidance_scale
                )
               
                if self.flow_mode == "v":
                    # Velocity prediction mode (original)
                    # Update x_t using the predicted velocity field
                    x_t = x_t + (timesteps[step] - timesteps[step - 1]) * model_output
                else:  # 'x1' mode
                    # Direct position prediction mode
                    x_t = x_t + (timesteps[step] - timesteps[step - 1]) * (model_output - return_dict['init_noise'])
                    
        return_dict['latents'] = x_t
        return return_dict
    
    def train_forward(self, condition_dict: Dict[str, Dict], 
                              latents: torch.Tensor, train_consistency=False) -> Dict[str, torch.Tensor]:
        """Compute training losses for both flow matching and consistency.
        
        Args:
            condition_dict: Dictionary containing training conditions
            latents: Target latent vectors
            
        Returns:
            Dictionary containing individual and total losses
        """

        # Extract input features
        audio = condition_dict['y']['audio_onset']
        word_tokens = condition_dict['y']['word']
        instance_ids = condition_dict['y']['id']
        seed_vectors = condition_dict['y']['seed']
        style_features = condition_dict['y']['style_feature']
    
        # Encode input modalities
        audio_features = self.modality_encoder(audio, word_tokens)

        # Initialize noise
        x0_noise = torch.randn_like(latents)

        # Sample timesteps and deltas
        deltas = 1 / torch.tensor([2 ** i for i in range(1, 8)]).to(latents.device)
        delta_probs = torch.ones((deltas.shape[0],)).to(latents.device) / deltas.shape[0]

        batch_size = latents.shape[0]
        flow_batch_size = int(batch_size * 3/4)

        # Apply conditional dropout during training for flow matching loss
        audio_features_flow = self.apply_conditional_dropout(audio_features[:flow_batch_size], cond_drop_prob=0.1)

        # Sample random coefficients
        t = sample_beta_distribution(batch_size, alpha=2, beta=1.2).to(latents.device)
        # t = sample_beta_distribution(batch_size, alpha=2, beta=0.8).to(latents.device)
        d = deltas[delta_probs.multinomial(batch_size, replacement=True)]
        d[:flow_batch_size] = 0

        # Prepare inputs
        t_coef = reshape_coefs(t)
        x_t = t_coef * latents + (1 - t_coef) * x0_noise
        t = t_coef.flatten()
        
        # Flow matching loss
        model_output = self.denoiser(
            x=x_t[:flow_batch_size],
            timesteps=t[:flow_batch_size],
            seed=seed_vectors[:flow_batch_size],
            at_feat=audio_features_flow,
            cond_time=d[:flow_batch_size],
        )
        
        losses = {}
        
        if self.flow_mode == "v":
            # Velocity prediction mode (original)
            flow_target = latents[:flow_batch_size] - x0_noise[:flow_batch_size]
            flow_loss = (
                F.mse_loss(flow_target, model_output) / t[:flow_batch_size]
            ).mean()
        else:  # 'x1' mode
            # Direct position prediction mode
            flow_target = latents[:flow_batch_size]
            flow_loss = (F.mse_loss(flow_target, model_output) / t[:flow_batch_size]).mean()

        losses["flow_loss"] = flow_loss

        # Consistency loss computation
        # Jan 11, perform cfg at the same time, 50% true and 50% false
        force_cfg = np.random.choice(
            [True, False], size=batch_size - flow_batch_size, p=[0.8, 0.2]
        )
        
        # Apply force_cfg externally
        audio_features_consistency = self.apply_force_cfg(audio_features[flow_batch_size:], force_cfg)
        
        with torch.no_grad():
            pred_t = self.denoiser(
                x=x_t[flow_batch_size:],
                timesteps=t[flow_batch_size:],
                seed=seed_vectors[flow_batch_size:],
                at_feat=audio_features_consistency,
                cond_time=d[flow_batch_size:],
            )
            
            d_coef = reshape_coefs(d)
            if self.flow_mode == "v":
                speed_t = pred_t
            else:
                speed_t = speed_t - x0_noise
            x_td = x_t[flow_batch_size:] + d_coef[flow_batch_size:] * speed_t
            
            d = d_coef.flatten()

            pred_td = self.denoiser(
                x=x_td,
                timesteps=t[flow_batch_size:] + d[flow_batch_size:],
                seed=seed_vectors[flow_batch_size:],
                at_feat=audio_features_consistency,
                cond_time=d[flow_batch_size:],
            )
            if self.flow_mode == "v":
                speed_td = pred_td
            else:
                speed_td = speed_t - x0_noise
            
            speed_target = (speed_t + speed_td) / 2

        model_pred = self.denoiser(
            x=x_t[flow_batch_size:],
            timesteps=t[flow_batch_size:],
            seed=seed_vectors[flow_batch_size:],
            at_feat=audio_features_consistency,
            cond_time=2 * d[flow_batch_size:],
        )
        if self.flow_mode == "v":
            speed_pred = model_pred
        else:
            speed_pred = model_pred - x0_noise

        consistency_loss = F.mse_loss(speed_pred, speed_target, reduction="mean")
        losses["consistency_loss"] = consistency_loss

        losses["loss"] = sum(losses.values())
        return losses
    

    def train_reflow(self, latents, audio_features, x0_noise, seed_vectors) -> Dict[str, torch.Tensor]:
        """Compute training losses for both flow matching and consistency.
        
        Args:
            condition_dict: Dictionary containing training conditions
            latents: Target latent vectors
            
        Returns:
            Dictionary containing individual and total losses
        """

        # Sample timesteps and deltas
        deltas = 1 / torch.tensor([2 ** i for i in range(1, 8)]).to(latents.device)
        delta_probs = torch.ones((deltas.shape[0],)).to(latents.device) / deltas.shape[0]

        batch_size = latents.shape[0]
        flow_batch_size = int(batch_size * 3/4)

        # Sample random coefficients
        t = sample_beta_distribution(batch_size, alpha=2, beta=1.2).to(latents.device)
        # t = sample_beta_distribution(batch_size, alpha=2, beta=0.8).to(latents.device)
        d = deltas[delta_probs.multinomial(batch_size, replacement=True)]
        d[:flow_batch_size] = 0

        # Prepare inputs
        t_coef = reshape_coefs(t)
        x_t = t_coef * latents + (1 - t_coef) * x0_noise
        t = t_coef.flatten()
        
        # Flow matching loss
        flow_pred = self.denoiser(
            x=x_t[:flow_batch_size],
            timesteps=t[:flow_batch_size],
            seed=seed_vectors[:flow_batch_size],
            at_feat=audio_features[:flow_batch_size],
            cond_time=d[:flow_batch_size],
        )
        
        flow_target = latents[:flow_batch_size] - x0_noise[:flow_batch_size]
        
        losses = {}
        flow_loss = (F.mse_loss(flow_target, flow_pred) / t).mean()
        losses['flow_loss'] = flow_loss

        # Consistency loss computation
        # Jan 11, perform cfg at the same time, 50% true and 50% false
        force_cfg = np.random.choice([True, False], size=batch_size-flow_batch_size, p=[0.8, 0.2])
        with torch.no_grad():
            speed_t = self.denoiser(
                x=x_t[flow_batch_size:],
                timesteps=t[flow_batch_size:],
                seed=seed_vectors[flow_batch_size:],
                at_feat=audio_features[flow_batch_size:],
                cond_time=d[flow_batch_size:],
            )
            
            d_coef = reshape_coefs(d)
            x_td = x_t[flow_batch_size:] + d_coef[flow_batch_size:] * speed_t
            d = d_coef.flatten()

            speed_td = self.denoiser(
                x=x_td,
                timesteps=t[flow_batch_size:] + d[flow_batch_size:],
                seed=seed_vectors[flow_batch_size:],
                at_feat=audio_features[flow_batch_size:],
                cond_time=d[flow_batch_size:],
            )
            
            speed_target = (speed_t + speed_td) / 2
        
        speed_pred = self.denoiser(
            x=x_t[flow_batch_size:],
            timesteps=t[flow_batch_size:],
            seed=seed_vectors[flow_batch_size:],
            at_feat=audio_features[flow_batch_size:],
            cond_time=2 * d[flow_batch_size:],
        )
        
        consistency_loss = F.mse_loss(speed_pred, speed_target, reduction="mean")
        losses['consistency_loss'] = consistency_loss

        losses['loss'] = sum(losses.values())
        return losses