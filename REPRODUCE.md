# Reproducing the results

Three levels of effort, cheapest first. Level 1 verifies the tables without
training anything.

Target machine for all timings: Apple M4, 16 GB unified memory, PyTorch MPS.
Everything runs on CPU too, more slowly.

---

## 0. Set up (≈3 min)

```bash
cd code
./setup_env.sh                 # creates .venv on Python 3.12, installs deps
source .venv/bin/activate
python wm.py info              # device, registered environments, registered memory cores
make test                      # 25 correctness tests (~1 min) -- run before trusting anything
```

`setup_env.sh` pins **Python 3.12** deliberately: PyTorch publishes no wheels
for 3.14, which is the default `python3` on the machine this was built on.

CarRacing needs two extra packages, and they are only needed for that one
environment:

```bash
pip install swig && pip install 'gymnasium[box2d]'
```

---

## 1. Verify the reported tables without training (≈5 min)

Every table in the paper is a pure function of the artefacts in `results/`.

```bash
cd code
python wm.py report ../results/tmaze_corridor32      # headline + retention tables
python wm.py report ../results/carracing
python scripts/make_tables.py --runs ../results/tmaze_corridor32 \
                              --carracing ../results/carracing \
                              --out ../paper/results_tables.tex
```

The regenerated `results_tables.tex` should match the committed one.

### Re-run a benchmark from the shipped weights (no training)

This recomputes every metric for one model from its trained checkpoint, using
the shipped latents. It is the strongest cheap check available.

```bash
cd code
mkdir -p /tmp/verify/supreme/seed0
cp ../checkpoints/tmaze_corridor32/supreme/seed0/memory.pt /tmp/verify/supreme/seed0/

# point the data root at the shipped latents
python wm.py bench -c configs/tmaze_hard.yaml \
    -o out_dir=/tmp/verify -o name=supreme -o memory.backbone=supreme \
    -o seed=0 -o memory.epochs=150 \
    -o data.root=../data -o vision.checkpoint=../checkpoints/vision/<dataset>/vae_z32_s0.pt \
    --no-control

python - <<'EOF'
import json

def get(d, path):
    for k in path.split('.'):
        d = d[k]
    return d

a = json.load(open('/tmp/verify/supreme/seed0/bench.json'))
b = json.load(open('../results/tmaze_corridor32/supreme/seed0/bench.json'))
for path in ('decision.accuracy', 'teacher_forced.nll_memory_critical',
             'teacher_forced.nll_1step', 'probes.effective_memory_horizon'):
    va, vb = get(a, path), get(b, path)
    print(f'{path:40s} {va:14.6f} vs {vb:14.6f}  {"OK" if abs(va - vb) < 1e-9 else "DIFFERS"}')
EOF
```

We ran exactly this and got identical values to six decimal places — see §4 of
the README.

> **Note on `data.root`.** The shipped dataset directories keep the *exact* names
> the code generates (`dataset_dir()` in `wmcore/data/collect.py`), so they are
> drop-in: either pass `-o data.root=../data` or copy them to `code/runs/data/`.
> The frames (`obs.npy`) are absent, which is fine for anything that reads
> latents — benchmarking, memory training, probes. Only VAE training and latent
> *re*-encoding need frames; regenerate those as in §2.

---

## 2. Regenerate the test data (≈10 min, 2.2 GB of disk)

Rollout collection is deterministic: rollout *i* is seeded `base_seed + i`
independently of which worker handles it, so the frame store is bit-exact
regardless of core count.

```bash
cd code
python wm.py collect -c configs/tmaze_hard.yaml --seed 0     # corridor 32, seed 0
python wm.py collect -c configs/tmaze_hard.yaml --seed 1
python wm.py collect -c configs/tmaze.yaml                    # corridor 16
python wm.py collect -c configs/carracing.yaml                # needs Box2D
python wm.py collect -c configs/tmaze_lambda.yaml             # corridor 6, the lambda sweep

cd runs/data && shasum -a 256 -c ../../../data/checksums.sha256
```

Every line should report `OK`. If a line fails, the regeneration is **not**
bit-identical on your machine and anything downstream will differ — tell us
rather than working around it.

---

## 3. Full retraining (≈6 h for everything)

Stage order matters: data → V (frozen) → latents → M → C → benchmark. `wm.py
study` does all of it and skips any stage whose artefacts already exist.

### The headline comparison

```bash
cd code
for SEED in 0 1; do
  python wm.py study -c configs/tmaze_hard.yaml \
      --backbones lstm,gru,attention,supreme \
      --seed $SEED -o out_dir=runs_seeds -o memory.epochs=150
done
python wm.py report runs_seeds
```

Cost per seed: `lstm` ≈3 min, `gru` ≈5 min, `attention` ≈4 min,
**`supreme` ≈90 min**, plus ≈20 min of CMA-ES per core that gets a controller.

### The ablations

```bash
python wm.py study -c configs/tmaze_hard.yaml \
    --backbones supreme-cms,hope-attention,supreme-titans \
    --seed 0 -o out_dir=runs_seeds -o memory.epochs=150 --no-control
```

`supreme-cms` and `hope-attention` are ≈1 min each. `supreme-titans` runs the
same per-timestep inner optimisation as `supreme` and costs ≈55 min per seed —
it is the one ablation we did not run (README §5).

### CarRacing

```bash
python wm.py study -c configs/carracing.yaml \
    --backbones lstm,gru,attention,supreme --no-control -o out_dir=runs_carracing
```

### The loss-weight sweep

```bash
for W in 1 20 100; do
  python wm.py study -c configs/tmaze_lambda.yaml --backbones lstm --no-control \
      -o out_dir=runs_lambda_w$W -o memory.critical_loss_weight=$W
done
```

### The corridor sweep (not run for the paper)

```bash
python scripts/sweep_horizon.py --backbones lstm,gru,supreme \
    --corridors 6,16,32,48 --rollouts 1200 --epochs 150 --out runs_sweep
```

Budget ≈5 h, dominated by `supreme`. This is the experiment that would turn the
choice of corridor 32 from a selected operating point into a curve.

---

## 4. Two configuration constraints that will silently invalidate a run

Both are enforced in code, but worth knowing before changing a config.

**`data.sequence_length` must span cue → decision.** In a delayed-recall episode
the cue appears at *t*=0 and the transition that needs it at *t*=*T*−3. If the
training window is shorter than that span, **no** window contains both: windows
starting at the cue stop before the decision, and windows reaching the decision
start after the cue. The dependency is then absent from the training signal
entirely and every memory core scores at chance for reasons unrelated to memory.
`check_window_spans_the_dependency()` reads the collected data and aborts.

**`memory.critical_loss_weight` changes what the benchmark can see.** One
transition per episode needs memory; at sequence length *L* it carries ≈1/*L* of
the loss, and getting it right saves only the outcome entropy (≈0.69 nats for a
binary cue) against a per-step NLL of ≈25. At weight 1.0 a converged model
predicts the *midpoint* of the two possible successors and every memory core
scores at chance — the benchmark has no headroom. The memory configs ship
weight 20, applied identically to both arms. See `results/lambda_sweep/` for the
measurement and README §5 for the disclosure.

---

## 5. Adding a memory core

Drop a folder next to `World Model/` and `Supreme/`; it is discovered by path.

```python
# code/Your Model/backbone.py   (+ an __init__.py that imports it)
from wmcore.memory.base import MemoryBackbone
from wmcore.memory.registry import register_backbone

class YourBackbone(MemoryBackbone):
    name = "yours"
    def initial_state(self, batch_size, device): ...
    def forward(self, u, state=None): ...   # (B, L, input_dim) -> (B, L, feature_dim)
    def step(self, u, state):        ...    # (B, input_dim)    -> (B, feature_dim)
    def state_features(self, state): ...    # (B, feature_dim), for C and for the probes

register_backbone("yours", YourBackbone)
```

Then add `"yours"` to `ALL_BACKBONES` in `code/tests/test_core.py` and run
`make test`. The load-bearing test is
`test_backbone_forward_matches_step`: a core whose sequence pass and
single-step pass disagree trains normally, benchmarks normally on teacher-forced
metrics, and is silently wrong in every open-loop, dream and control result.

Constraints so the comparison stays valid: `feature_dim` must equal
`memory.hidden_dim`, and the backbone's parameter count should be within ~10% of
the baseline's 299,008.
