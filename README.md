# Supreme — Self-Modifying Memory in a World Model: code, test data and benchmark artefacts

Companion repository for the paper. It exists to answer one reviewer request
directly: **the code and the test data used to build and evaluate the models.**

Everything reported in the paper that has backing artefacts is here, together
with the exact inputs the models consumed, the trained weights, and the raw
benchmark outputs. Section [What is *not* here](#what-is-not-here-read-this)
states plainly which paper claims are **not** covered, and why.

---

## 1. What was tested

The study compares two world models in the sense of Ha & Schmidhuber (2018) —
a vision encoder *V*, a memory *M*, a controller *C* — that differ in **exactly
one component**: the recurrent core inside *M*.

| | Memory core | Source |
|---|---|---|
| Baseline | LSTM(256) → MDN head (the MDN-RNN) | Ha & Schmidhuber 2018, §2.2 |
| **Supreme** | self-modifying Titans → Continuum Memory System (**Hope**) | Behrouz et al. 2025, §7.1, §8.1–8.3 |

*V* (a 4,348,547-parameter ConvVAE), *C* (a linear controller trained by
CMA-ES), the environments, the data pipeline, the mixture-density output head,
the loss, the optimiser and the whole evaluation suite are **the same code** in
both arms. A machine check (`assert_comparable`) aborts any run whose two legs
differ outside the memory backbone.

### Memory cores evaluated

| key | composition | backbone params | role |
|---|---|---|---|
| `lstm` | LSTM(256) | 299,008 | baseline (the MDN-RNN core) |
| `gru` | GRU(256) | 224,256 | gated-RNN control |
| `attention` | causal self-attention | 299,058 | "perfect memory" control |
| **`supreme`** | **Titans → CMS (Hope)** | **300,696** | **model under test** |
| `supreme-cms` | LSTM → CMS | 375,184 | ablation: CMS alone |
| `hope-attention` | attention → CMS | 375,234 | the paper's Hope-Attention variant |

`supreme` is parameter-matched to the baseline within 0.6%.

### Environments

| environment | what it tests | role |
|---|---|---|
| `TMaze-v0` | retention of one cue across a tunable delay | primary |
| `CarRacing-v0` | the reference task of Ha & Schmidhuber (2018) | external control |
| `SequenceRecall-v0` | holding *k* associations at once | implemented, **not run** |

`TMaze-v0` and `SequenceRecall-v0` are pure NumPy (no gymnasium, Box2D or
pygame), render at 64×64 like the paper, and are bit-reproducible. They place
the memory requirement **inside the world-model loss**, not only in the reward:
the frame after the decision reveals the answer, so that one transition is
unpredictable unless the state still holds the cue. The environment tags it
`memory_critical` and the benchmark reports likelihood on exactly those steps.

### Benchmarks run

| # | benchmark | what it measures |
|---|---|---|
| 0 | **Decision accuracy** | at the memory-critical transition, does *M* predict the frame that actually comes next? Chance is exactly 0.5 and hedging scores at chance |
| 1 | Teacher-forced NLL | overall, and restricted to memory-critical steps |
| 2 | Open-loop imagination | error vs. horizon, and its log-slope (drift) |
| 3 | **Retention probes** | linear decodability of the cue from `h_t` vs. delay, with an observation-only control |
| 4 | Control | real-environment return with the CMA-ES controller |
| 5 | Cost | parameters, latency, throughput, peak RSS |
| — | Continual learning | implemented, **not run** (see §6) |

**Instrument validation.** 7 of the 25 tests in `code/tests/` exist to validate
the measurement apparatus rather than the code: the open-loop harness is checked
against an analytic oracle (a perfect model must score zero error at every
horizon; a biased one must compound), decision accuracy is checked with both an
oracle and a *hedger* (which must score chance), the probe is checked to find a
planted signal and to **reject** pure noise, the MDN likelihood is checked
against the closed-form Gaussian, and every backbone must satisfy
`forward(sequence) == step-by-step` to 1e-5.

---

## 2. Verification — three checks run while assembling this repository

Stated up front because they are what the reviewer request is really asking
about: not only *are* the code and data here, but *can the reported numbers be
recovered from them*. Each check is reproducible with the commands in
[REPRODUCE.md](REPRODUCE.md).

### 2.1 Dataset regeneration is bit-exact

The 2.21 GB frame store is not shipped (see §5), on the grounds that it is
reproducible. We tested that claim rather than relying on it: a dataset was
regenerated from scratch and hashed against the original, giving **identical
SHA-256** for both `obs.npy` and `action.npy`.

Rollout *i* is seeded `base_seed + i` independently of which worker process
handles it, so regeneration does not depend on core count either.
`data/checksums.sha256` carries the digests for all five datasets, so a reviewer
can confirm the same thing:

```bash
cd code && python wm.py collect -c configs/tmaze_hard.yaml --seed 0
cd runs/data && shasum -a 256 -c ../../../data/checksums.sha256
```

### 2.2 The shipped checkpoints reproduce the reported numbers

Optimiser state was stripped from every checkpoint (297 MB → 94 MB). To confirm
that is lossless for evaluation rather than merely assume it, the full benchmark
was re-run from a stripped checkpoint and compared against the committed
`bench.json`:

| metric | from stripped checkpoint | shipped `bench.json` |
|---|---|---|
| decision accuracy | 1.000000 | 1.000000 |
| NLL @ memory-critical | 14.641197 | 14.641197 |
| NLL, 1-step | 15.831006 | 15.831006 |
| effective memory horizon | 32 | 32 |

Identical to six decimal places. The checkpoints are therefore good for
verification, though not for resuming training.

### 2.3 The loss-weight sweep was lost, re-run, and reproduced exactly

The artefacts behind the paper's loss-weight (λ) table had been deleted. Rather
than cite a table with no backing data, the sweep was re-run end to end — fresh
*V* training, fresh latent encoding, fresh memory training — and it reproduces
the submitted table on **every cell**:

| λ | decision accuracy | NLL @ memory-critical | NLL all steps | probe at decision |
|---|---|---|---|---|
| 1 | 0.500 | 25.53 | 13.16 | 0.44 |
| 20 | 1.000 | 19.42 | 12.76 | 1.00 |
| 100 | 1.000 | 13.68 | 12.49 | 1.00 |

Artefacts in `results/lambda_sweep/`, data in
`data/TMaze-v0_corridor_length6_.../`, configuration in
`code/configs/tmaze_lambda.yaml`. A missing-data gap became a reproducibility
result.

---

## 3. Headline results

`TMaze-v0`, corridor 32 (a 32-step required memory horizon), 2 seeds,
mean ± s.e.m. Every column shares the same data, the same frozen *V*, the same
head, loss, optimiser and seeds.

| metric | `lstm` | `gru` | **`supreme`** | `supreme-cms` | `hope-attention` |
|---|---|---|---|---|---|
| Decision accuracy (chance 0.5) | 0.475 ± 0.047 | 0.483 ± 0.006 | **1.000 ± 0.000** | 0.483 ± 0.017 | 1.000 ± 0.000 |
| NLL @ memory-critical | 25.55 ± 2.15 | 25.30 ± 2.34 | **15.61 ± 0.97** | 25.50 ± 2.58 | 16.73 ± 0.33 |
| NLL, all steps | 14.08 ± 1.99 | 14.12 ± 1.97 | 13.68 ± 2.16 | 13.90 ± 2.07 | 14.64 ± 2.09 |
| Effective memory horizon | 4 | 20 ± 4 | **32** | 4 | 32 |
| Probe accuracy @ delay 32 | 0.411 ± 0.078 | 0.533 ± 0.067 | **0.978 ± 0.000** | 0.411 ± 0.056 | 0.867 ± 0.022 |
| Real-env return | 0.06 ± 0.19 | −0.13 ± 0.13 | **1.00** (n=1) | — | — |
| Backbone params | 299,008 | 224,256 | 300,696 | 375,184 | 375,234 |
| Training pass (ms) | 2.8 | 7.2 | 827 | 3.8 | 4.2 |

Three things this table says:

1. **Hope solves a horizon neither recurrent core reaches.** Both Supreme seeds
   agree exactly (1.000, and probe 0.978 at delay 32).
2. **The advantage is invisible in average NLL** (13.68 vs 14.08/14.12,
   overlapping error bars). A study reporting only world-model likelihood would
   have found nothing here. This is why the suite leads with decision accuracy
   and the retention probes.
3. **CMS alone contributes nothing.** `supreme-cms` is the baseline LSTM plus
   CMS and 25% *more* parameters, and it lands on the baseline (0.483 vs 0.475,
   horizon 4 vs 4). The advantage is in the sequence mixer — and both mixers
   that work (`supreme`, `hope-attention`) reach 1.000. The honest claim is
   therefore *"the MDN-RNN's recurrent core is the bottleneck, and either a
   self-modifying memory or attention removes it"*, not *"self-modification is
   uniquely necessary"*.

A dissociation worth noting: `hope-attention` retains the cue but its
imagination collapses (open-loop MSE 44.6 at step 16 against Supreme's 0.51,
drift slope +0.126 against −0.070). Attention remembers and predicts badly;
Hope does both — at a fixed-size state rather than one that grows with the
episode.

### CarRacing (external control)

| metric | `lstm` | `gru` | `attention` | `supreme` |
|---|---|---|---|---|
| NLL, all steps | 10.87 | 11.13 | 10.19 | 10.36 |
| MSE, 1-step | 1.118 | 1.085 | 1.107 | **0.873** |
| Open-loop MSE @ 10 | **7.86** | 8.73 | 13.51 | 9.41 |

CarRacing does **not** discriminate between memory cores, as expected — the task
is close to fully observable from a single frame. All four land within a nat of
one another, with no ordering consistent across metrics. It is reported as a
negative control: the discriminating power of the synthetic suite is a property
of the suite, not of the models.

---

## 4. Compute and environment

Every number in this repository was produced on **one laptop**. No cluster, no
accelerator beyond the integrated GPU.

| | |
|---|---|
| Machine | Apple MacBook, **M4** chip |
| CPU | 10 cores (4 performance + 6 efficiency) |
| Memory | 16 GB unified (17.18 × 10⁹ bytes); ≈10 GB usable in practice |
| GPU backend | Apple Metal via PyTorch **MPS** |
| OS | macOS 27.0 (build 26A5388g), `macOS-27.0-arm64-arm-64bit` |
| Python | 3.12.12 (Homebrew) |
| PyTorch | 2.13.0, MPS built and available |
| NumPy | 2.5.2 |
| gymnasium / Box2D | 1.3.0 / 2.3.10 (CarRacing only) |
| cma | 4.4.4 |
| matplotlib / Pillow / PyYAML | 3.11.1 / 12.3.0 / 6.0.3 |
| Precision | float32 throughout (no mixed precision) |

The execution environment is recorded in **every** run's `bench.json` under
`meta.device`, so any artefact can be traced to the machine that produced it.

### Measured cost

Per training step, batch 64, sequence length 37:

| core | training step | inference step | corridor-32 run (150 epochs) |
|---|---|---|---|
| `lstm` | ~10 ms | ~0.5 ms | ~3 min |
| `gru` | ~16 ms | ~0.9 ms | ~5 min |
| `attention` | ~13 ms | ~0.6 ms | ~4 min |
| **`supreme`** | **~2000 ms** | ~2.5 ms | **~90 min** |

Hope's two-orders-of-magnitude training cost is inherent to a per-timestep inner
optimisation, not an artefact of a slow implementation. The chunk-parallel dual
form of Behrouz et al. §8.2 **was** implemented and measured at 1.01×, 0.98× and
1.05× speed-up at L = 37, 69 and 250, then removed: the cost is the backward
pass through the sequential memory recurrence, not kernel launches. Both arms see
the same data for the same number of epochs at matched parameter counts, so
wall-clock must not be read as "spent more compute to learn more".

Approximate wall-clock for the whole set of artefacts here: **≈6 hours**, almost
all of it Supreme's memory training.

### Data volumes

| | |
|---|---|
| Raw frame store generated | **≈2.2 GB** across 5 datasets (64×64×3 uint8, memory-mapped) |
| Frames **shipped** here | none — regenerable bit-exactly, see §2 |
| Latents / actions / labels shipped | **46 MB** (what *M* actually consumed) |
| Trained weights shipped | **94 MB** (optimiser state stripped) |
| Benchmark artefacts shipped | 2.1 MB |

---

## 5. Repository layout

```
code/                      the full implementation (60 Python files)
├── World Model/           MODEL 1: lstm_backbone.py, attention_backbone.py
├── Supreme/               MODEL 2: titans.py, cms.py, hope.py
├── wmcore/                SHARED core, used verbatim by both models
│   ├── envs/              CarRacing adapter + the synthetic memory environments
│   ├── data/              rollout collection, memmapped store, torch datasets
│   ├── vision/            V: the ConvVAE
│   ├── memory/            M: backbone contract + shared MDN head + registry
│   ├── controller/        C: linear controller, CMA-ES / OpenAI-ES
│   ├── train/             the four training stages
│   └── bench/             the benchmark suite + reporting
├── configs/               every experiment configuration (YAML)
├── scripts/               run drivers, the sweep, the LaTeX table generator
├── tests/                 25 correctness tests
└── docs/                  design rationale, benchmark definitions, findings

data/                      the test data the models consumed
├── checksums.sha256       SHA-256 of every regenerable frame store
└── <dataset>/             latents.npz, action.npy, meta.npz, manifest.json

results/                   raw benchmark outputs, one directory per experiment
├── tmaze_corridor32/      the headline comparison (2 seeds)
├── tmaze_corridor16/      the easier operating point (1 seed)
├── carracing/             the external control
└── lambda_sweep/          the loss-weight ablation (w1, w20, w100)

checkpoints/               trained weights, optimiser state stripped
├── tmaze_corridor32/      one memory model per (backbone, seed)
├── carracing/
├── tmaze_corridor16/
└── vision/                the frozen V for each dataset

paper/                     supreme.tex and the auto-generated results_tables.tex
```

### Why the frames are not shipped, and why that is safe

The frame store is 2.21 GB across five datasets and is **bit-exactly
reproducible**. Rollout *i* is
seeded `base_seed + i` independently of worker assignment, so regeneration is
deterministic and worker-count-independent. We verified this rather than assuming
it: regenerating a dataset from scratch and hashing it against the original gives
**identical SHA-256** for both `obs.npy` and `action.npy`.

A stronger piece of evidence arrived while assembling this repository. The
artefacts behind the paper's loss-weight table had been deleted, so the sweep was
re-run from scratch — new *V* training, new latent encoding, new memory
training — and it reproduced the submitted table **exactly**, every cell:

| λ | decision accuracy | NLL @ memory-critical | NLL all steps | probe at decision |
|---|---|---|---|---|
| 1 | 0.500 | 25.53 | 13.16 | 0.44 |
| 20 | 1.000 | 19.42 | 12.76 | 1.00 |
| 100 | 1.000 | 13.68 | 12.49 | 1.00 |

`data/checksums.sha256` lets a reviewer confirm the same thing for the datasets
backing the reported results:

```bash
cd code && python wm.py collect -c configs/tmaze_hard.yaml --seed 0
cd ../data && shasum -a 256 -c checksums.sha256   # paths relative to runs/data
```

What *is* shipped is what the memory model actually consumed: the precomputed
latents (`latents.npz`, the frozen *V*'s posterior means and log-variances), the
actions, and the ground-truth labels (`meta.npz` — cue, phase, delay,
`memory_critical`, correct action) that the benchmark's probes read and the model
never sees.

### The checkpoints are verified, not just shipped

Optimiser state was stripped (297 MB → 94 MB) and confirmed lossless for
evaluation by re-running the benchmark from a stripped checkpoint — see §2.2.
The comparison, repeated here for convenience:

| metric | from stripped checkpoint | shipped `bench.json` |
|---|---|---|
| decision accuracy | 1.000000 | 1.000000 |
| NLL @ memory-critical | 14.641197 | 14.641197 |
| NLL, 1-step | 15.831006 | 15.831006 |
| effective memory horizon | 32 | 32 |

---

## 6. What is *not* here — read this

The gaps are as load-bearing as the results, and a reviewer will find them
anyway.

### Experiments not run

- **`attention` at corridor 32.** The strongest retention control was run on
  CarRacing but **not** at the headline operating point. This is the most
  significant omission: it means the headline table compares Hope against two
  recurrent cores, not against attention. (`hope-attention` *is* at corridor 32
  and reaches 1.000, which partly covers it.)
- **`supreme-titans` (Hope without CMS) was never run on the current code.** So
  "which half of Hope does the work" is only partly answered: `supreme-cms`
  shows CMS alone is inert, but Titans-alone is untested here.
- **`SequenceRecall-v0`** (the capacity axis) — implemented and unit-tested,
  never run.
- **Continual learning** — implemented, with a self-check that reports
  `informative: false` when no task was learned; never run in a valid regime.
- **The corridor sweep** — `code/scripts/sweep_horizon.py` exists but was not run
  to completion. This matters: corridor 32 was chosen *after* observing that
  corridor 16 saturates, so presented alone it is a selected operating point. The
  sweep is what would turn that into a curve whose crossover is the finding.

### Caveats on what *is* here

- **Two seeds, not three.** With a near-bimodal metric, two seeds agreeing
  exactly is encouraging but is not a distribution.
- **Supreme seed 1 trained for 61 of 150 epochs** before being interrupted. Its
  best-validation checkpoint was already past the phase transition and reaches
  the same decision accuracy (1.000) and probe accuracy (0.978) as seed 0, so it
  is reported — but the two seeds do **not** share a training budget, and the
  error bar on Supreme's NLL column mixes two training lengths.
- **Supreme's real-environment return is a single seed**; no error bar.
  `supreme-cms` and `hope-attention` have no controller runs at all.
- **CarRacing ran at half the planned scale** (150 rollouts, 12 memory epochs)
  and without a controller.
- **Corridor-16 Supreme artefacts are deliberately excluded.** A stability fix to
  the Titans recurrence (see below) changed what `supreme` computes. The
  corridor-16 Supreme and `supreme-titans` runs predate it and are therefore
  **not reproducible from this code**, so they are not shipped as results. The
  `lstm`, `gru` and `supreme-cms` columns at corridor 16 are unaffected.

### Paper claims without artefacts in this repository

| paper claim | status |
|---|---|
| Headline table, retention table, CarRacing table | ✅ `results/` |
| Loss-weight (λ) sweep | ✅ `results/lambda_sweep/` — artefacts for the submitted table were lost, so the sweep was **re-run for this release and reproduces the submitted table exactly** (every cell, to the reported precision) |
| "corridor 24: LSTM 12.87 vs GRU 12.88 NLL" | ❌ artefacts lost. The same point is made by the corridor-32 table already here (14.08 vs 14.12 with horizons 4 vs 20); we suggest the paper cite that instead |
| "corridor 32 at a 60-epoch budget" | ⚠️ recoverable from `results/tmaze_corridor32/*/metrics.jsonl` at epoch 60, not a separate run |

### One implementation deviation that matters

Written exactly as Behrouz et al. Eq. 88, the Titans recurrence **diverges** —
measured here, the memory norm reaches 10¹⁰ by step 80 and overflows by step 96.
The cause is a property of self-modification, not a coding error: the generated
target `v̂ = M_ref(v)` is itself a function of the memory and `v` is in turn
`M_ref(x)`, so the write term contains `+η·M_ref·v·kᵀ`, which is *quadratic* in
‖M‖. In the paper the memories are two-layer MLPs inside a normalised stack that
absorbs this; the linear form of their Eq. 93 has nothing to absorb it.

The fix is a projection onto a Frobenius ball after each update. The update rule
is left **exactly** as Eq. 88, and the projection is inactive below the radius —
at the study's operating point it never engages (‖M‖_F ≈ 2.5 against a radius of
20). The engagement rate is logged every epoch as `titans_clip_rate`, so a run
where the model is being *constrained* rather than *guarded* is visible rather
than silent.

Two less invasive-looking alternatives were tried first and are documented in
`code/Supreme/titans.py`, because both are tempting and both are worse.
Normalising `v̂` stabilises the recurrence but destroys the magnitude of the
quantity the update regresses on — measured, decision accuracy at corridor 32
fell from 1.000 to chance while the training curve looked healthy for the first
~25 epochs, so the damage was invisible early.

---

## 7. Reproducing

See [REPRODUCE.md](REPRODUCE.md) for exact commands. The short version:

```bash
cd code
./setup_env.sh && source .venv/bin/activate
make test                                    # 25 correctness tests, ~1 min
python wm.py study -c configs/tmaze_hard.yaml --backbones lstm,gru,supreme
python wm.py report runs
```

Every run writes `config.json`, `metrics.jsonl` and `bench.json`; every table in
the paper is generated from those artefacts by `code/scripts/make_tables.py`, so
no number in the paper is typed by hand.

## References

- D. Ha, J. Schmidhuber. *World Models.* arXiv:1803.10122, 2018.
- A. Behrouz, M. Razaviyayn, P. Zhong, V. Mirrokni. *Nested Learning: The
  Illusion of Deep Learning Architecture.* NeurIPS 2025, arXiv:2512.24695.
- N. Hansen. *The CMA Evolution Strategy: A Comparing Review.* 2006.
- D. Lopez-Paz, M. Ranzato. *Gradient Episodic Memory for Continual Learning.*
  NeurIPS 2017.

The two source papers are **not** redistributed here — download them from arXiv.
