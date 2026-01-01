# eval_pusht.py
from __future__ import annotations

import argparse
from collections import deque
import numpy as np
import torch

import gymnasium as gym

from diffusion_policy.models.encoders import ObsConditionEncoder
from diffusion_policy.models.denoisers import TemporalUNetDenoiser
from diffusion_policy.models.diffusion import DiffusionPolicy, DiffusionConfig
from diffusion_policy.checkpoint import load_checkpoint
from diffusion_policy.normalizer import MinMaxNormalizer


def _extract_obs(obs):
    """
    Try to be robust to different PushT wrappers.
    Returns: (img_thwc uint8/float, agent_pos (2,))
    """
    if isinstance(obs, dict):
        img = obs.get("img", None)
        if img is None:
            img = obs.get("image", None)
        if img is None:
            raise KeyError(f"obs dict has no img/image keys: {list(obs.keys())}")

        # agent pos
        if "agent_pos" in obs:
            agent_pos = obs["agent_pos"]
        elif "state" in obs:
            agent_pos = obs["state"][..., :2]
        else:
            raise KeyError(f"obs dict has no agent_pos/state keys: {list(obs.keys())}")

        return img, np.asarray(agent_pos)
    else:
        raise TypeError(f"Unsupported obs type: {type(obs)}")


def _img_to_torch(img, device):
    """
    img: HWC or CHW, uint8 or float
    returns: (1, 3, 96, 96) float32 in [0,1]
    """
    x = np.asarray(img)
    if x.ndim == 3 and x.shape[-1] == 3:
        x = np.transpose(x, (2, 0, 1))  # CHW
    x = x.astype(np.float32)
    if x.max() > 1.0:
        x = x / 255.0
    t = torch.from_numpy(x).unsqueeze(0).to(device)  # (1,3,H,W)
    return t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--inference_steps", type=int, default=16)
    ap.add_argument("--render", action="store_true")
    args = ap.parse_args()

    device = torch.device(args.device)

    # build model (match training dims)
    obs_enc = ObsConditionEncoder(cond_dim=256, n_obs_steps=2).to(device)
    denoiser = TemporalUNetDenoiser(action_dim=2, cond_dim=256, base_dim=128).to(device)
    policy = DiffusionPolicy(
        obs_encoder=obs_enc,
        denoiser=denoiser,
        action_dim=2,
        horizon=10,
        cfg=DiffusionConfig(num_train_timesteps=100, beta_schedule="squaredcos_cap_v2", prediction_type="epsilon"),
    ).to(device)

    # EMA model is stored in ckpt; we load it into a separate module for eval
    # simplest: load into policy, then (optionally) overwrite with ema weights if present
    ckpt = load_checkpoint(args.ckpt, model=policy, ema_model=None, optimizer=None, map_location="cpu")

    # if EMA weights exist, prefer them
    if "ema_model" in ckpt:
        policy.load_state_dict(ckpt["ema_model"], strict=True)

    normalizer = MinMaxNormalizer.load_state_dict(ckpt["normalizer"])
    policy.eval()

    # env
    env_id = "pusht-v0"
    env = gym.make(env_id, obs_type="pixels")

    returns = []
    for ep in range(args.episodes):
        obs, info = env.reset()
        hist_img = deque(maxlen=2)
        hist_pos = deque(maxlen=2)

        # bootstrap history with first obs repeated
        img, pos = _extract_obs(obs)
        hist_img.append(img)
        hist_pos.append(pos)
        hist_img.append(img)
        hist_pos.append(pos)

        done = False
        total_r = 0.0

        while not done:
            if args.render:
                env.render()

            # build To=2 batch
            imgs = torch.cat([_img_to_torch(hist_img[0], device), _img_to_torch(hist_img[1], device)], dim=0) # (2,3,H,W)
            imgs = imgs.unsqueeze(0)  # (1,2,3,H,W)

            pos_np = np.stack([hist_pos[0], hist_pos[1]], axis=0)  # (2,2)
            pos_norm = normalizer.normalize_np("agent_pos", pos_np).astype(np.float32)
            pos_t = torch.from_numpy(pos_norm).unsqueeze(0).to(device)  # (1,2,2)

            # sample trajectory (normalized)
            traj_norm = policy.sample_actions(imgs, pos_t, num_inference_steps=args.inference_steps)  # (1,10,2)
            traj_norm = traj_norm[0].detach().cpu().numpy()

            # unnormalize to env action space
            traj = normalizer.unnormalize_np("action", traj_norm)  # (10,2)

            # receding horizon: execute first 8 actions
            for a in traj[:8]:
                obs, r, terminated, truncated, info = env.step(a.astype(np.float32))
                total_r += float(r)
                done = bool(terminated or truncated)
                img, pos = _extract_obs(obs)
                hist_img.append(img)
                hist_pos.append(pos)
                if done:
                    break

        returns.append(total_r)
        print(f"ep={ep} return={total_r:.3f}")

    print(f"avg_return={np.mean(returns):.3f} ± {np.std(returns):.3f}")


if __name__ == "__main__":
    main()
