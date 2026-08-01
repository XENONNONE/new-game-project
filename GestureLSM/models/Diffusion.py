import time
import inspect
import logging
from typing import Optional
import numpy as np
from omegaconf import DictConfig

import torch
import torch.nn.functional as F
from models.config import instantiate_from_config
from models.utils.utils import count_parameters, extract_into_tensor, sum_flat

logger = logging.getLogger(__name__)


class GestureDiffusion(torch.nn.Module):
    def __init__(self, cfg) -> None:
        super().__init__()
        self.cfg = cfg
        self.modality_encoder = instantiate_from_config(cfg.model.modality_encoder)
        self.denoiser = instantiate_from_config(cfg.model.denoiser)
        self.scheduler = instantiate_from_config(cfg.model.scheduler)
        self.alphas = torch.sqrt(self.scheduler.alphas_cumprod)
        self.sigmas = torch.sqrt(1 - self.scheduler.alphas_cumprod)

        self.do_classifier_free_guidance = cfg.model.do_classifier_free_guidance
        self.guidance_scale = cfg.model.guidance_scale
        self.smooth_l1_loss = torch.nn.SmoothL1Loss(reduction='none')

        self.seq_len = self.denoiser.seq_len
        self.input_dim = self.denoiser.input_dim
        self.num_joints = self.denoiser.joint_num

    def summarize_parameters(self) -> None:
        logger.info(f'Denoiser: {count_parameters(self.denoiser)}M')
        logger.info(f'Scheduler: {count_parameters(self.modality_encoder)}M')

    def apply_classifier_free_guidance(self, x, timesteps, seed, at_feat, guidance_scale=1.0):
        """
        Apply classifier-free guidance by running both conditional and unconditional predictions.
        
        Args:
            x: Input tensor
            timesteps: Timestep tensor
            seed: Seed vectors
            at_feat: Audio features
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
                cond_drop_prob=0.0,
                null_cond=False
            )
        
        # Double the batch for classifier free guidance
        x_doubled = torch.cat([x] * 2, dim=0)
        seed_doubled = torch.cat([seed] * 2, dim=0)
        at_feat_doubled = torch.cat([at_feat] * 2, dim=0)
        
        # Properly expand timesteps to match doubled batch size
        batch_size = x.shape[0]
        timesteps_doubled = timesteps.expand(batch_size * 2)
        
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

    def predicted_origin(self, model_output: torch.Tensor, timesteps: torch.Tensor, sample: torch.Tensor) -> tuple:
        self.alphas = self.alphas.to(model_output.device)
        self.sigmas = self.sigmas.to(model_output.device)
        alphas = extract_into_tensor(self.alphas, timesteps, sample.shape)
        sigmas = extract_into_tensor(self.sigmas, timesteps, sample.shape)

        # i will do this
        if self.scheduler.config.prediction_type == "epsilon":
            pred_original_sample = (sample - sigmas * model_output) / alphas
            pred_epsilon = model_output

        elif self.scheduler.config.prediction_type == "sample":
            pred_original_sample = model_output
            pred_epsilon = (sample - alphas * model_output) / sigmas

        elif self.scheduler.config.prediction_type == "v_prediction":
            pred_original_sample = alphas * sample - sigmas * model_output
            pred_epsilon = alphas * model_output + sigmas * sample
        else:
            raise ValueError(f"Invalid prediction_type {self.scheduler.config.prediction_type}.")

        return pred_original_sample, pred_epsilon



    def forward(self, cond_: dict) -> dict:

        audio = cond_['y']['audio_onset']
        word = cond_['y']['word']
        id = cond_['y']['id']
        seed = cond_['y']['seed']
        style_feature = cond_['y']['style_feature']

        audio_feat = self.modality_encoder(audio, word)

        bs = audio_feat.shape[0]
        shape_ = (bs, self.input_dim * self.num_joints, 1, self.seq_len)
        latents = torch.randn(shape_, device=audio_feat.device)

        latents = self._diffusion_reverse(latents, seed, audio_feat, guidance_scale=self.guidance_scale)

        return latents



    def _diffusion_reverse(
            self,
            latents: torch.Tensor,
            seed: torch.Tensor,
            at_feat: torch.Tensor,
            guidance_scale: float = 1,
    ) -> torch.Tensor:

        return_dict = {}
        # scale the initial noise by the standard deviation required by the scheduler, like in Stable Diffusion
        # this is the initial noise need to be returned for rectified training
        latents = latents * self.scheduler.init_noise_sigma


        noise = latents


        return_dict["init_noise"] = latents
        return_dict['at_feat'] = at_feat
        return_dict['seed'] = seed

        # set timesteps
        self.scheduler.set_timesteps(self.cfg.model.scheduler.num_inference_steps)
        timesteps = self.scheduler.timesteps.to(at_feat.device)

        latents = torch.zeros_like(latents)

        latents = self.scheduler.add_noise(latents, noise, timesteps[0])

        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, and between [0, 1]
        extra_step_kwargs = {}
        if "eta" in set(
                inspect.signature(self.scheduler.step).parameters.keys()):
            extra_step_kwargs["eta"] = self.cfg.model.scheduler.eta

        for i, t in enumerate(timesteps):
            latent_model_input = latents
            # actually it does nothing here according to ddim scheduler
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

            # predict the noise residual
            model_output = self.apply_classifier_free_guidance(
                x=latent_model_input,
                timesteps=t,
                seed=seed,
                at_feat=at_feat,
                guidance_scale=guidance_scale)

            latents = self.scheduler.step(model_output, t, latents, **extra_step_kwargs).prev_sample
        return_dict['latents'] = latents
        return return_dict

    def _diffusion_process(self,
            latents: torch.Tensor,
            audio_feat: torch.Tensor,
            id: torch.Tensor,
            seed: torch.Tensor,
            style_feature: torch.Tensor
        ) -> dict:

        # [batch_size, n_frame, latent_dim]
        noise = torch.randn_like(latents)
        bsz = latents.shape[0]


        timesteps = torch.randint(
            0,
            self.scheduler.config.num_train_timesteps,
            (bsz,),
            device=latents.device
        )

        timesteps = timesteps.long()
        noisy_latents = self.scheduler.add_noise(latents.clone(), noise, timesteps)

        model_output = self.denoiser(
            x=noisy_latents,
            timesteps=timesteps,
            seed=seed,
            at_feat=audio_feat,
        )

        latents_pred, noise_pred = self.predicted_origin(model_output, timesteps, noisy_latents)

        n_set = {
            "noise": noise,
            "noise_pred": noise_pred,
            "sample_pred": latents_pred,
            "sample_gt": latents,
            "timesteps": timesteps,
            "model_output": model_output,
        }
        return n_set

    def train_forward(self, cond_: dict, x0: torch.Tensor) -> dict:
        audio = cond_['y']['audio_onset']
        word = cond_['y']['word']
        id = cond_['y']['id']
        seed = cond_['y']['seed']
        style_feature = cond_['y']['style_feature']

        audio_feat = self.modality_encoder(audio, word)
        
        # Apply conditional dropout during training
        audio_feat = self.apply_conditional_dropout(audio_feat, cond_drop_prob=0.1)
        
        n_set = self._diffusion_process(x0, audio_feat, id, seed, style_feature)

        loss_dict = dict()

        # Diffusion loss
        if self.scheduler.config.prediction_type == "epsilon":
            model_pred, target = n_set['noise_pred'], n_set['noise']
        elif self.scheduler.config.prediction_type == "sample":
            model_pred, target = n_set['sample_pred'], n_set['sample_gt']
        elif self.scheduler.config.prediction_type == "v_prediction":
            # For v_prediction, we need to compute the v target
            # v = alpha * noise - sigma * x0
            timesteps = n_set['timesteps']

            self.alphas = self.alphas.to(x0.device)
            self.sigmas = self.sigmas.to(x0.device)
            alphas = extract_into_tensor(self.alphas, timesteps, x0.shape)
            sigmas = extract_into_tensor(self.sigmas, timesteps, x0.shape)

            v_target = alphas * n_set['noise'] - sigmas * n_set['sample_gt']
            model_pred, target = n_set['model_output'], v_target  # The model output is the v prediction
        else:
            raise ValueError(f"Invalid prediction_type {self.scheduler.config.prediction_type}.")


        # mse loss
        diff_loss = F.mse_loss(target, model_pred, reduction="mean")

        loss_dict['diff_loss'] = diff_loss

        total_loss = sum(loss_dict.values())
        loss_dict['loss'] = total_loss
        return loss_dict
