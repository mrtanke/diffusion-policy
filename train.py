# train.py
from __future__ import annotations

import argparse
import os
from dataclasses import asdict

import torch
from torch.utils.data import DataLoader
from accelerate import Accelerator
from ema_pytorch import EMA

from diffusion_policy.data.pusht_zarr_dataset import PushTImageZarrDataset, PushTWindowSpec
from diffusion_policy.models.encoders import ObsConditionEncoder
from diffusion_policy.models.denoisers import TemporalUNetDenoiser
from diffusion_policy.models.diffusion import DiffusionPolicy, DiffusionConfig
from diffusion_policy.checkpoint import save_checkpoint, load_checkpoint


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zarr_path", type=str, required=True)
    ap.add_argument("--out_dir", type=str, default="outputs/pusht_image")
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-6)
    ap.add_argument("--train_steps", type=int, default=200_000)
    ap.add_argument("--log_every", type=int, default=100)
    ap.add_argument("--save_every", type=int, default=5000)
    ap.add_argument("--resume", type=str, default="")
    ap.add_argument("--mixed_precision", type=str, default="no", choices=["no", "fp16", "bf16"])
    ap.add_argument("--grad_accum", type=int, default=1)

    # diffusion / inference knobs
    ap.add_argument("--num_train_timesteps", type=int, default=100)
    ap.add_argument("--beta_schedule", type=str, default="squaredcos_cap_v2")
    ap.add_argument("--prediction_type", type=str, default="epsilon")

    args = ap.parse_args()

    accelerator = Accelerator(mixed_precision=args.mixed_precision, gradient_accumulation_steps=args.grad_accum)
    device = accelerator.device

    # dataset
    window = PushTWindowSpec(n_obs_steps=2, horizon=10, n_action_steps=8, pad_before=1, pad_after=7)
    ds = PushTImageZarrDataset(zarr_path=args.zarr_path, window=window, in_memory=False)
    dl = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    # model
    obs_enc = ObsConditionEncoder(cond_dim=256, n_obs_steps=2)
    denoiser = TemporalUNetDenoiser(action_dim=2, cond_dim=256, base_dim=128)
    policy = DiffusionPolicy(
        obs_encoder=obs_enc,
        denoiser=denoiser,
        action_dim=2,
        horizon=10,
        cfg=DiffusionConfig(
            num_train_timesteps=args.num_train_timesteps,
            beta_schedule=args.beta_schedule,
            prediction_type=args.prediction_type,
        ),
    )
    
    optimizer = torch.optim.AdamW(policy.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # EMA model
    ema = EMA(policy, beta=0.995, update_after_step=100, update_every=1)

    # accelerate prepare
    policy, optimizer, dl = accelerator.prepare(policy, optimizer, dl)

    # resume if needed
    step = 0
    if args.resume:
        ckpt = load_checkpoint(args.resume, accelerator.unwrap_model(policy), ema.ema_model, optimizer=optimizer)
        step = int(ckpt.get("step", 0))
        accelerator.print(f"[resume] loaded step={step} from {args.resume}")

    os.makedirs(args.out_dir, exist_ok=True)

    policy.train()
    dl_iter = iter(dl)

    while step < args.train_steps:
        try:
            batch = next(dl_iter)
        except StopIteration:
            dl_iter = iter(dl)
            batch = next(dl_iter)

        # move to device (accelerate’s DataLoader already gives tensors; just ensure device)
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}

        with accelerator.accumulate(policy):
            loss = policy(batch)
            accelerator.backward(loss)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            # EMA update on the unwrapped (real) model parameters
            ema.update()

        # logging and saving
        if accelerator.is_main_process and (step % args.log_every == 0):
            accelerator.print(f"step={step} loss={loss.item():.4f}")

        if accelerator.is_main_process and (step % args.save_every == 0) and step > 0:
            ckpt_path = os.path.join(args.out_dir, f"ckpt_step_{step}.pt")
            save_checkpoint(
                ckpt_path,
                model=accelerator.unwrap_model(policy),
                ema_model=ema.ema_model,
                optimizer=optimizer,
                step=step,
                normalizer_state=ds.normalizer.state_dict(),
                extra={"args": vars(args)},
            )
            accelerator.print(f"[save] {ckpt_path}")

        step += 1

    if accelerator.is_main_process:
        ckpt_path = os.path.join(args.out_dir, f"ckpt_final.pt")
        save_checkpoint(
            ckpt_path,
            model=accelerator.unwrap_model(policy),
            ema_model=ema.ema_model,
            optimizer=optimizer,
            step=step,
            normalizer_state=ds.normalizer.state_dict(),
            extra={"args": vars(args)},
        )
        accelerator.print(f"[save final] {ckpt_path}")


if __name__ == "__main__":
    main()
