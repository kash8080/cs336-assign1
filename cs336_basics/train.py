"""train.py — training entry point for the CS336 Transformer LM."""
import argparse
import os
import time

import numpy as np
import torch
import wandb

from cs336_basics.TransformerLM import TransformerLM
from cs336_basics.AdamW import AdamW
from cs336_basics.DataLoading import DataLoading
from cs336_basics.CrossEntropy import cross_entropy
from cs336_basics.CosineLRScheduling import cosine_lr_scheduling
from cs336_basics.GradientClipping import gradient_clipping
from cs336_basics.Serialization import save_checkpoint, load_checkpoint


def resolve_device(requested: str) -> str:
    """Pick a device. On Apple Silicon, "auto" prefers MPS, then CUDA, then CPU."""
    if requested != "auto":
        return requested
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


@torch.no_grad()
def evaluate(model, loader, val_data, num_batches):
    """Average cross-entropy over a few validation batches."""
    model.eval()
    total = 0.0
    for _ in range(num_batches):
        inputs, targets = loader.load(val_data)
        logits = model(inputs)
        total += cross_entropy(logits, targets).item()
    model.train()
    return total / num_batches


def train(args: argparse.Namespace) -> None:
    """Run the training loop. All hyperparameters come in via `args`."""
    device = resolve_device(args.device)
    print(f"Using device: {device}")

    # --- experiment tracking (only if a project was given; each init() = one new run) ---
    use_wandb = args.wandb_project is not None
    if use_wandb:
        wandb.init(project=args.wandb_project, name=args.wandb_name, config=vars(args))

    # --- data (memory-mapped so we never load the whole corpus into RAM) ---
    train_data = np.load(args.train_path, mmap_mode="r")
    val_data = np.load(args.val_path, mmap_mode="r") if args.val_path else None
    loader = DataLoading(args.batch_size, args.context_length, device)

    # --- model ---
    model = TransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        rope_theta=args.rope_theta,
    ).to(device)

    # --- optimizer ---
    optimizer = AdamW(
        model.parameters(),
        lr=args.max_lr,
        weight_decay=args.weight_decay,
        betas=(args.beta1, args.beta2),
        eps=args.eps,
    )

    # --- optionally resume ---
    start_step = 0
    if args.resume_from:
        start_step = load_checkpoint(args.resume_from, model, optimizer, map_location=device)
        print(f"Resumed from {args.resume_from} at step {start_step}")

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    model.train()
    t0 = time.time()
    tokens_per_step = args.batch_size * args.context_length
    best_val_loss = float("inf")

    for step in range(start_step, args.max_steps):
        step_t0 = time.time()

        # 1. LR schedule -> write into the optimizer's param groups
        lr = cosine_lr_scheduling(
            args.max_lr, args.min_lr, args.warmup_iters, args.cosine_cycle_iters, step
        )
        for group in optimizer.param_groups:
            group["lr"] = lr

        # 2. Sample a batch
        inputs, targets = loader.load(train_data)

        # 3. Forward + loss
        logits = model(inputs)
        loss = cross_entropy(logits, targets)

        # 4. Backward
        optimizer.zero_grad()
        loss.backward()

        # 5. Clip, then step
        grad_norm = gradient_clipping(args.max_grad_norm, model.parameters())
        optimizer.step()

        # mac is heating up. so slow down training to reduce heating. 
        # longer training is fine as it's just learning
        # remove for actual training
        # for cuda, we can set a max temp so this is not needed TODO
        if device == "mps":
            torch.mps.synchronize()   # ensure GPU work truly finished
            step_dt = time.time() - step_t0
            time.sleep(step_dt * 0.25)    # ~25% cooldown -> ~25% longer wall-clock

        # --- logging ---
        if step % args.log_interval == 0:
            dt = time.time() - t0
            loss_val = loss.item()
            tokens_seen = (step + 1) * tokens_per_step
            tokens_per_sec = tokens_seen / dt if dt > 0 else 0.0
            print(f"step {step:>6} | loss {loss_val:.4f} | lr {lr:.3e} | "
                  f"gnorm {grad_norm:.2f} | {tokens_per_sec/1e3:.1f}k tok/s | {dt:.1f}s")
            if use_wandb:
                wandb.log({
                    "train/loss": loss_val,
                    "train/lr": lr,
                    "train/grad_norm": grad_norm,   # pre-clip; watch for spikes/divergence
                    "time/wallclock_s": dt,          # set as x-axis to plot loss vs. time
                    "throughput/tokens_seen": tokens_seen,
                    "throughput/tokens_per_sec": tokens_per_sec,
                }, step=step)

        # --- validation ---
        if val_data is not None and step > 0 and step % args.eval_interval == 0:
            val_loss = evaluate(model, loader, val_data, args.eval_batches)
            best_val_loss = min(best_val_loss, val_loss)
            print(f"step {step:>6} | val_loss {val_loss:.4f} | best {best_val_loss:.4f}")
            if use_wandb:
                wandb.log({
                    "val/loss": val_loss,
                    "val/best_loss": best_val_loss,
                    "time/wallclock_s": time.time() - t0,
                }, step=step)
                wandb.run.summary["best_val_loss"] = best_val_loss

        # --- checkpointing ---
        if step > 0 and step % args.checkpoint_interval == 0:
            path = os.path.join(args.checkpoint_dir, f"ckpt_{step}.pt")
            save_checkpoint(model, optimizer, step, path)
            print(f"saved checkpoint -> {path}")

    # --- final checkpoint ---
    final_path = os.path.join(args.checkpoint_dir, "ckpt_final.pt")
    save_checkpoint(model, optimizer, args.max_steps, final_path)
    print(f"saved final checkpoint -> {final_path}")

    if use_wandb:
        wandb.finish()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train a Transformer language model.")

    # --- model architecture ---
    p.add_argument("--vocab-size", type=int, default=10_000)
    p.add_argument("--context-length", type=int, default=256)
    p.add_argument("--d-model", type=int, default=512)
    p.add_argument("--num-layers", type=int, default=4)
    p.add_argument("--num-heads", type=int, default=16)
    p.add_argument("--d-ff", type=int, default=1344) # ~8/3 of d_model and a multiple of 64
    p.add_argument("--rope-theta", type=float, default=10_000.0)

    # --- optimizer / AdamW ---
    p.add_argument("--max-lr", type=float, default=1e-3)
    p.add_argument("--min-lr", type=float, default=1e-5)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--beta1", type=float, default=0.9)
    p.add_argument("--beta2", type=float, default=0.95)
    p.add_argument("--eps", type=float, default=1e-8)
    p.add_argument("--max-grad-norm", type=float, default=1.0)

    # --- LR schedule ---
    p.add_argument("--warmup-iters", type=int, default=200)
    p.add_argument("--cosine-cycle-iters", type=int, default=5_000)

    # --- training ---
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--max-steps", type=int, default=5_000)
    p.add_argument("--device", type=str, default="auto",
                   help='"auto" (MPS on Apple Silicon, else CUDA, else CPU), or "mps"/"cuda"/"cpu"')

    # --- data / IO ---
    p.add_argument("--train-path", type=str, required=True)
    p.add_argument("--val-path", type=str, default=None)
    p.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    p.add_argument("--resume-from", type=str, default=None)

    # --- logging / checkpointing cadence ---
    p.add_argument("--log-interval", type=int, default=10)
    p.add_argument("--eval-interval", type=int, default=500)
    p.add_argument("--eval-batches", type=int, default=20)
    p.add_argument("--checkpoint-interval", type=int, default=1000)

    # --- experiment tracking (W&B) ---
    p.add_argument("--wandb-project", type=str, default=None,
                   help="W&B project name. If omitted, W&B logging is disabled.")
    p.add_argument("--wandb-name", type=str, default=None,
                   help="Optional run name; W&B auto-generates one if omitted.")

    return p.parse_args()


def main() -> None:
    args = parse_args()
    train(args)


if __name__ == "__main__":
    main()
