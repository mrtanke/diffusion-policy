## diffusion-policy
Implementation of **Diffusion Policy**, from [Diffusion Policy: Visuomotor Policy Learning via Action Diffusion](https://arxiv.org/abs/2303.04137)

This repo trains a policy to denoise an action trajectory conditioned on a short observation history(images + agent positions). At inference time, the policy generates a horizon of actions and typically executes only the first few steps in a **receding-horizon** loop.

More implementation details: [Reproducing Diffusion Policy](https://mrtanke.github.io/projects/2026-01-02-reproducing-diffusion-policy/)

### Install

Create a virtual environment (optional) and install from `pyproject.toml`:

```bash
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -e .
```


### Dataset

This repo expects a PushT replay buffer stored as a **zarr** directory:

- `data/pusht/pusht_cchi_v7_replay.zarr`

The dataset loader produces one training sample as:

- `obs_image`: `(To, 3, 96, 96)` in `[0, 1]`
- `obs_agent_pos`: `(To, 2)` normalized to `[-1, 1]`
- `action`: `(H, 2)` normalized to `[-1, 1]`

Window defaults:

- observation steps: `To = 2`
- horizon: `H = 10`
- executed actions per replanning step: `Ta = 8` (receding-horizon)


### Quick sanity checks

Inspect zarr structure:

```bash
python scripts/inspect_zarr.py --zarr_path data/pusht/pusht_cchi_v7_replay.zarr
```

Smoke test dataset shapes:

```bash
python scripts/smoke_test_dataset.py --zarr_path data/pusht/pusht_cchi_v7_replay.zarr
```

Smoke test forward + sampling:

```bash
python scripts/smoke_test_model.py --zarr_path data/pusht/pusht_cchi_v7_replay.zarr
```

### Training

Train with `accelerate`:

```bash
accelerate launch train.py --zarr_path data/pusht/pusht_cchi_v7_replay.zarr --out_dir outputs/pusht_image --batch_size 64 --train_steps 200000 --mixed_precision no
```

### Evaluation (PushT)

Roll out a checkpoint in the `pusht-v0` environment:

```bash
python eval_pusht.py --ckpt outputs/pusht_image/ckpt_final.pt --episodes 10 --inference_steps 16 --render
```


### Acknowledgements

This is an implementation inspired by the Diffusion Policy line of work and the broader diffusion-model ecosystem.
