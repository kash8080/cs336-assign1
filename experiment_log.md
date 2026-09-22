# Experiment Log — CS336 Assignment 1

A running record of every training run I tried, why I tried it, and what I concluded.
Each experiment has a W&B run; metrics are logged against both **gradient steps** and
**wall-clock time** (`time/wallclock_s`).

- **W&B project:** `<project-name>` — <link to project>
- **Data:** train `<path>`, val `<path>` (vocab size `<N>`)
- **Hardware:** <e.g. Apple M-series MPS / A100 / etc.>

Logging infra lives in [`cs336_basics/train.py`](cs336_basics/train.py). Metrics tracked:
`train/loss`, `train/lr`, `train/grad_norm` (pre-clip), `val/loss`, `val/best_loss`,
`time/wallclock_s`, `throughput/tokens_seen`, `throughput/tokens_per_sec`. Full config
is captured per-run via `wandb.config = vars(args)`.

---

## How to read this log

For each run:
- **Hypothesis** — what I expected and why (before looking at results).
- **Change** — what differs from the previous/baseline run.
- **Command** — exact CLI so the run is reproducible.
- **Result** — final train loss, best val loss, wall-clock, steps, W&B link.
- **Conclusion** — did the hypothesis hold? What I'll try next.

---

## Runs

### exp-000 — baseline
- **Hypothesis:** Small model with batch 16 and context 128. 
- **Change:** max learning rate from 1e-4 to 1e0 to create U shaped graph of loss vs lr
- **Command:**
  ```bash
  python -m cs336_basics.train --train-path training_data_encoded.npy --val-path validation_data_encoded.npy --d-model 512 --num-layers 4 --max-steps 501 --warmup-iters 50 --cosine-cycle-iters 500 --eval-interval 50 --device mps --batch-size 16 --context-length 128 --max-lr 3e1 --wandb-project cs336-b16 --wandb-name lr3e1
  ```
- **Result:** at lr 1e-3 : train loss 3.06 · best val 3.06 · 500 steps · 2 min · [W&B run] (cs336-b16)
- **Conclusion:** _fill in_

<!--
### exp-00X — <short title>
- **Hypothesis:**
- **Change:**
- **Command:**
  ```bash
  ```
- **Result:** train loss `___` · best val `___` · `___` steps · `___` min · [W&B run](…)
- **Conclusion:**
-->

---

## Sweeps summary

Keep the headline comparison here so trends are visible at a glance. Note both
step-efficiency and time-efficiency — a config can win on steps but lose on wall-clock.

### Learning-rate sweep
| run | max_lr | best val | steps to best | wall-clock | notes |
|-----|--------|----------|---------------|------------|-------|
|     |        |          |               |            |       |

### Batch-size sweep
| run | batch_size | best val | tokens seen | tokens/sec | wall-clock | notes |
|-----|-----------|----------|-------------|------------|------------|-------|
|     |           |          |             |            |            |       |

### Other ablations
| run | what changed | best val | wall-clock | notes |
|-----|--------------|----------|------------|-------|
|     |              |          |            |       |

---

## Takeaways
- _Running list of conclusions across the whole assignment: best config found, what
  helped, what didn't, surprising instabilities (watch `train/grad_norm`)._
