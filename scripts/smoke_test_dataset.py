# scripts/smoke_test_dataset.py
import argparse
from torch.utils.data import DataLoader

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from diffusion_policy.data.pusht_zarr_dataset import PushTImageZarrDataset, PushTWindowSpec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zarr_path", type=str, required=True)
    args = ap.parse_args()

    ds = PushTImageZarrDataset(
        zarr_path=args.zarr_path,
        window=PushTWindowSpec(n_obs_steps=2, horizon=10, n_action_steps=8, pad_before=1, pad_after=7),
        in_memory=False,
    )
    dl = DataLoader(ds, batch_size=8, shuffle=True, num_workers=2, pin_memory=True)

    batch = next(iter(dl))
    print("obs_image:", batch["obs_image"].shape, batch["obs_image"].dtype)
    print("obs_agent_pos:", batch["obs_agent_pos"].shape, batch["obs_agent_pos"].dtype)
    print("action:", batch["action"].shape, batch["action"].dtype)
    print("image min/max:", batch["obs_image"].min().item(), batch["obs_image"].max().item())
    print("agent_pos min/max:", batch["obs_agent_pos"].min().item(), batch["obs_agent_pos"].max().item())
    print("action min/max:", batch["action"].min().item(), batch["action"].max().item())


if __name__ == "__main__":
    main()
