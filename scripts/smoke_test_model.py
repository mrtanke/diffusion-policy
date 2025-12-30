# scripts/smoke_test_model_step2.py
import argparse
import torch
from torch.utils.data import DataLoader

from diffusion_policy.data.pusht_zarr_dataset import PushTImageZarrDataset, PushTWindowSpec
from diffusion_policy.models.encoders import ObsConditionEncoder
from diffusion_policy.models.denoisers import TemporalUNetDenoiser
from diffusion_policy.models.diffusion import DiffusionPolicy, DiffusionConfig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zarr_path", type=str, required=True)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    ds = PushTImageZarrDataset(
        zarr_path=args.zarr_path,
        window=PushTWindowSpec(n_obs_steps=2, horizon=10, n_action_steps=8, pad_before=1, pad_after=7),
        in_memory=False,
    )
    dl = DataLoader(ds, batch_size=8, shuffle=True, num_workers=2, pin_memory=True)
    batch = next(iter(dl))
    batch = {k: v.to(args.device) for k, v in batch.items()}

    obs_enc = ObsConditionEncoder(cond_dim=256, n_obs_steps=2).to(args.device)
    denoiser = TemporalUNetDenoiser(action_dim=2, cond_dim=256).to(args.device)

    policy = DiffusionPolicy(
        obs_encoder=obs_enc,
        denoiser=denoiser,
        action_dim=2,
        horizon=10,
        cfg=DiffusionConfig(num_train_timesteps=100, beta_schedule="squaredcos_cap_v2", prediction_type="epsilon"),
    ).to(args.device)

    loss = policy(batch)
    print("train loss:", float(loss))

    act = policy.sample_actions(batch["obs_image"], batch["obs_agent_pos"], num_inference_steps=10)
    print("sampled action traj:", act.shape, act.min().item(), act.max().item())


if __name__ == "__main__":
    main()
