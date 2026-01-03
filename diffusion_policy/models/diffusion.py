# diffusion_policy/models/diffusion.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import DDPMScheduler


@dataclass
class DiffusionConfig:
    num_train_timesteps: int = 100

    # a cosine schedule from "Improved Denoising Diffusion Probabilistic Models" (Nichol & Dhariwal, 2021)""
    beta_schedule: str = "squaredcos_cap_v2"
    prediction_type: str = "epsilon"  # predict noise


class DiffusionPolicy(nn.Module):
    """
    Full policy:
      cond = obs_encoder(obs)
      eps_pred = denoiser(x_noisy, k, cond)
    Uses diffusers DDPMScheduler for noise/add/step.
    """
    def __init__(
        self,
        obs_encoder: nn.Module,
        denoiser: nn.Module,
        action_dim: int = 2,
        horizon: int = 10,
        cfg: DiffusionConfig = DiffusionConfig(),
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        self.obs_encoder = obs_encoder
        self.denoiser = denoiser
        self.action_dim = action_dim
        self.horizon = horizon

        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=cfg.num_train_timesteps,
            beta_schedule=cfg.beta_schedule,
            prediction_type=cfg.prediction_type,
            clip_sample=False,  # we control clipping outside if needed
        )

        if device is not None:
            self.to(device)

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Returns training loss (MSE on epsilon).
        batch:
          obs_image: (B, To, 3, 96, 96)
          obs_agent_pos: (B, To, 2)
          action: (B, H, 2)  normalized [-1,1]
        """
        obs_image = batch["obs_image"]
        obs_agent_pos = batch["obs_agent_pos"]
        action = batch["action"]

        B, H, action_dim = action.shape
        assert H == self.horizon and action_dim == self.action_dim

        # Efficient conditioning: encode obs once
        cond = self.obs_encoder(obs_image, obs_agent_pos)  # (B, cond_dim)

        # Sample diffusion timestep per batch element
        k = torch.randint(
            low=0,
            high=self.noise_scheduler.config.num_train_timesteps,
            size=(B,),
            device=action.device,
            dtype=torch.long,
        )

        # Add noise
        noise = torch.randn_like(action)
        x_noisy = self.noise_scheduler.add_noise(action, noise, k)

        # Predict noise epsilon
        eps_pred = self.denoiser(x_noisy, k, cond)

        loss = F.mse_loss(eps_pred, noise) # eps_pred, noise: (B,H,action_dim)
        return loss

    @torch.no_grad()
    def sample_actions(
        self,
        obs_image: torch.Tensor,
        obs_agent_pos: torch.Tensor,
        num_inference_steps: Optional[int] = None,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        """
        Returns denoised action trajectory (B, H, action_dim) in normalized space.
        """
        device = obs_image.device
        B = obs_image.shape[0]

        # encode once
        cond = self.obs_encoder(obs_image, obs_agent_pos)

        # init noisy trajectory
        x = torch.randn((B, self.horizon, self.action_dim), device=device, generator=generator)

        # setup inference steps (default: full 100)
        if num_inference_steps is None:
            num_inference_steps = self.noise_scheduler.config.num_train_timesteps

        self.noise_scheduler.set_timesteps(num_inference_steps, device=device)

        for k in self.noise_scheduler.timesteps:
            # k is scalar tensor; expand to (B,)
            kk = torch.full((B,), int(k), device=device, dtype=torch.long)
            eps = self.denoiser(x, kk, cond)
            step = self.noise_scheduler.step(eps, k, x, generator=generator)
            x = step.prev_sample

        # optional clamp to training range
        x = x.clamp(-1.0, 1.0)
        return x
