# Test data

What the models actually consumed, for every dataset that backs a reported
result.

## What is in each directory

Directory names are the **exact** names the code generates
(`dataset_dir()` in `code/wmcore/data/collect.py`), so they are drop-in: pass
`-o data.root=../data` or copy them into `code/runs/data/`.

| file | what it is | why it is here |
|---|---|---|
| `latents.npz` | `mu`, `logvar` — the frozen *V*'s posterior over every frame | **this is the memory model's input**; *M* never sees pixels |
| `action.npy` | `(N, T, A)` float32 actions | the other half of *M*'s input |
| `meta.npz` | reward, terminal, mask, and the ground-truth labels: `cue`, `phase`, `delay_since_cue`, `memory_critical`, `correct_action`, `variant` | read by the benchmark's probes and the decision metric; **never** seen by any model |
| `manifest.json` | environment id, kwargs, policy, seed, shapes | provenance |

## What is omitted, and why that is safe

`obs.npy` — the raw 64×64×3 uint8 frame store — is **not** shipped. It is 2.2 GB
across these four datasets, and it is bit-exactly reproducible.

Rollout *i* is seeded `base_seed + i` independently of which worker process
handles it, so regeneration is deterministic and does not depend on core count.
We verified this rather than asserting it: regenerating a dataset from scratch
and hashing it against the original gives **identical SHA-256** for both
`obs.npy` and `action.npy`.

To check it yourself:

```bash
cd ../code
python wm.py collect -c configs/tmaze_hard.yaml --seed 0
cd runs/data && shasum -a 256 -c ../../../data/checksums.sha256
```

`checksums.sha256` covers `obs.npy` and `action.npy` for every dataset below.
Lines for datasets you have not regenerated will report as missing; that is
expected — check the ones you did.

## Datasets

| label | directory | backs |
|---|---|---|
| corridor 32, seed 0 | `TMaze-v0_corridor_length32_..._n1200_t37_prandom_s0` | headline table, seed 0 |
| corridor 32, seed 1 | `TMaze-v0_corridor_length32_..._n1200_t37_prandom_s1` | headline table, seed 1 |
| corridor 16 | `TMaze-v0_corridor_length16_..._n1500_t21_prandom_s0` | `results/tmaze_corridor16/` |
| CarRacing | `CarRacing-v0_n150_t250_prandom_s0` | `results/carracing/` |
| λ sweep, corridor 6 | `TMaze-v0_corridor_length6_..._n2000_t11_prandom_s0` | `results/lambda_sweep/` |

See `_index.json` for shipped file sizes and the size of the omitted frame store
per dataset.

## A note on what `latents.npz` means for reproducibility

Because *V* is trained once per environment and then **frozen**, and because the
latents are precomputed from it, shipping `latents.npz` ships the exact input
distribution the memory comparison ran on. Two consequences:

1. Anyone can retrain and re-benchmark *M* — the part of the study the paper is
   about — from the shipped latents alone, with no frames and no VAE training.
2. Retraining *V* yourself will **not** give identical latents (MPS convolution
   kernels are not bit-deterministic), which would shift every downstream
   number slightly. That is why the frozen *V* checkpoints are shipped too, in
   `../checkpoints/vision/`.
