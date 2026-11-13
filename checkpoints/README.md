# Trained weights

Enough to re-run every benchmark without training anything.

```
tmaze_corridor32/<backbone>/seed<k>/memory.pt   the memory model M
carracing/<backbone>/seed0/memory.pt
tmaze_corridor16/<backbone>/seed0/memory.pt
vision/<dataset>/vae_z32_s0.pt                  the frozen V for that dataset
```

## Optimiser state was stripped — and we checked that this is lossless

Checkpoints are written with Adam state, which is two thirds of the bulk
(297 MB → 94 MB after stripping). `load_checkpoint` only restores an optimiser
when one is passed, so evaluation is unaffected in principle. We confirmed it in
practice by re-running the full benchmark from a stripped checkpoint:

| metric | from stripped checkpoint | shipped `bench.json` |
|---|---|---|
| decision accuracy | 1.000000 | 1.000000 |
| NLL @ memory-critical | 14.641197 | 14.641197 |
| NLL, 1-step | 15.831006 | 15.831006 |
| effective memory horizon | 32 | 32 |

These checkpoints are therefore suitable for verification but **not** for
resuming training.

## Each checkpoint carries its own provenance

```python
import torch
p = torch.load("tmaze_corridor32/supreme/seed0/memory.pt",
               map_location="cpu", weights_only=False)
p["model_class"]   # which class to rebuild
p["config"]        # the full config that produced it
p["epoch"]         # the epoch this (best-validation) checkpoint came from
p["extra"]         # val metrics, parameter table, latent/action dims, timing, device
```

`wmcore.train.train_memory.load_memory(cfg, device)` rebuilds the model from a
config plus the checkpoint's recorded `latent_dim` / `action_dim`, so no
dimensions need to be supplied by hand.

## Why the frozen V is shipped

*V* is trained once per environment and then frozen; both memory cores consume
latents from the *same* checkpoint. Retraining it would not give identical
latents — MPS convolution kernels are not bit-deterministic — and every
downstream number would shift slightly. Shipping *V* makes the latents in
`../data/` regenerable from the frames exactly.
