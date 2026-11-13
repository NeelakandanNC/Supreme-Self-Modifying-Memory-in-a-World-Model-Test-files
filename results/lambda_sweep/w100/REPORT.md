# Memory-layer comparison -- `TMaze-v0`

Every number below comes from runs whose configs differ **only** in `memory.backbone` (verified by `wmcore.config.assert_comparable`). V, C, the dataset, the MDN head, the optimiser and the seeds are shared.

**Settings.** env `TMaze-v0` {'corridor_length': 6, 'n_cues': 2, 'distractors': 2, 'cue_steps': 2, 'feedback_steps': 2}; train window 11; seeds [0]; device `mps` on arm64, torch 2.13.0.

Note the `memory.critical_loss_weight` used for these runs (see each run's `config.json`, and `docs/findings.md` §1 for why it matters): at weight 1.0 every memory layer scores at chance on the decision metric.

## Headline results

| Metric | lstm | Better |
|---|---|---|
| Decision accuracy (model) | 1 | higher |
| 1-step NLL | 12.49 | lower |
| NLL @ memory-critical | 13.68 | lower |
| NLL @ long eval length | -- | lower |
| Open-loop MSE at step 2 | 0.08664 | lower |
| Open-loop MSE at step 4 | 0.01742 | lower |
| Drift slope | -0.3937 | lower |
| Probe AUC (state) | 1 | higher |
| Probe AUC (obs control) | 0.568 | higher |
| Effective horizon | 6 | higher |
| Return (real env) | -- | higher |
| Decision accuracy (agent) | -- | higher |
| Forgetting on T0 (acc) | -- | lower |
| Backbone params | 2.99e+05 | lower |
| Step latency (ms) | 0.707 | lower |
| Train pass (ms) | 1.493 | lower |

_Seeds per model: lstm=1._


## Open-loop imagination error by horizon

| mse_at_horizon | lstm |
|---|---|
| 1 | 0.07097 |
| 2 | 0.08664 |
| 4 | 0.01742 |


## Cue retention by delay (linear probe accuracy)

`state` probes the recurrent features h_t; `obs` probes the latent z_t alone and is the control -- once the cue leaves the screen it falls to chance, so any gap above it is memory.

| delay | lstm (state) | lstm (obs) |
|---|---|---|
| 0 | 1.000 | 1.000 |
| 1 | 1.000 | 0.507 |
| 2 | 1.000 | 0.413 |
| 4 | 1.000 | 0.480 |
| 6 | 1.000 | 0.440 |


## Cost

Measured on the machine named in each run's `meta.device`.
