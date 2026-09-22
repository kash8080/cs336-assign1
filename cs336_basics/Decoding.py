"""Autoregressive decoding (sampling) from a trained TransformerLM.

Supports temperature scaling and top-p (nucleus) sampling, stops on the
`<|endoftext|>` token or after a maximum number of new tokens.

Gotchas that shaped this file (details at each call site):
  - The checkpoint stores WEIGHTS ONLY, not architecture. You must rebuild
    TransformerLM with the exact hyperparameters used at training time, or
    `load_state_dict` fails on shape mismatch. Hence all the --*-size / --d-*
    flags in the CLI, whose defaults mirror train.py.
  - `context_length` is passed in explicitly rather than read off the model:
    the block/RoPE modules don't expose it cleanly, so introspection would be
    brittle. It must match training.
  - The tokenizer must be constructed with special_tokens=[END_OF_TEXT] or the
    EOT string tokenizes as ordinary bytes and generation never stops on it.
"""
import argparse

import torch
from torch import Tensor
from jaxtyping import Float

from cs336_basics.TransformerLM import TransformerLM
from cs336_basics.Softmax import softmax
from cs336_basics.tokenizer import Tokenizer
from cs336_basics.Serialization import load_model

END_OF_TEXT = "<|endoftext|>"


def top_p_filter(probs: Float[Tensor, " vocab"], top_p: float) -> Float[Tensor, " vocab"]:
    """Zero out the tail of `probs` outside the smallest nucleus with cumulative mass >= top_p,
    then renormalize. `probs` is a 1-D probability distribution (already softmaxed).

    Nucleus (top-p) sampling: instead of a fixed number of candidates (top-k), keep the
    smallest set of highest-probability tokens whose cumulative mass first reaches top_p.
    The set size adapts to how peaked the distribution is.
    """
    # top_p >= 1.0 keeps the whole distribution, so there is nothing to filter. This is also
    # the fast path for the CLI default (--top-p 1.0 == plain temperature sampling).
    if top_p >= 1.0:
        return probs

    sorted_probs, sorted_idx = torch.sort(probs, descending=True)
    cumsum = torch.cumsum(sorted_probs, dim=-1)

    # `cumsum - sorted_probs` is the mass strictly BEFORE each token. A token is in the nucleus
    # iff that preceding mass is still under the threshold, i.e. it is the token that first
    # pushes the running total to >= top_p (inclusive) or any token before it. Using the
    # exclusive-prefix form avoids an off-by-one where a hard cutoff would drop the very token
    # that crosses the threshold.
    keep = cumsum - sorted_probs < top_p
    keep[0] = True  # guard: if the top token alone already exceeds top_p, still keep it

    # Scatter the kept probabilities back to their ORIGINAL vocab positions (sorted_idx maps
    # sorted rank -> vocab id); everything else stays zero.
    filtered = torch.zeros_like(probs)
    filtered[sorted_idx[keep]] = sorted_probs[keep]
    # Renormalize so the survivors form a valid distribution for torch.multinomial.
    return filtered / filtered.sum()


@torch.no_grad()
def generate(
    model: TransformerLM,
    tokenizer: Tokenizer,
    prompt: str,
    context_length: int,
    max_new_tokens: int = 256,
    temperature: float = 1.0,
    top_p: float = 1.0,
    device: str = "cpu",
) -> str:
    """Sample a completion for `prompt` and return the newly generated text (prompt excluded).

    `context_length` must match the value the model was trained with; the running
    sequence is cropped to its last `context_length` tokens before each forward pass.
    """
    # eval() disables dropout/etc.; @torch.no_grad() (on the decorator) skips autograd so we
    # neither build a graph nor leak memory across the generation loop.
    model.eval()

    # Resolve the EOT id THROUGH the tokenizer's special-token path. This only returns the
    # single sentinel id if the tokenizer was built with special_tokens=[END_OF_TEXT];
    # otherwise the string would be byte-encoded into several ordinary ids and we'd never
    # match it below, so generation would only ever stop at max_new_tokens.
    eot_id = tokenizer.encode(END_OF_TEXT)[0]

    ids = tokenizer.encode(prompt)
    x = torch.tensor(ids, dtype=torch.long, device=device)[None, :]  # [1, seq]; batch dim of 1

    # We accumulate ONLY the newly sampled ids and decode those at the end, so the returned
    # string is the completion without the prompt echoed back.
    generated: list[int] = []
    for _ in range(max_new_tokens):
        # Crop to the last context_length tokens before each forward pass: RoPE positions are
        # only defined up to context_length, so feeding a longer sequence would index past the
        # precomputed rotation tables (or silently extrapolate). Slicing is cheap since we
        # re-run the full forward each step (no KV cache in this assignment).
        logits = model(x[:, -context_length:])          # [1, seq, vocab]
        # Only the LAST position predicts the next token; earlier positions are the model's
        # (re)predictions of tokens we already have.
        next_logits = logits[0, -1]                      # [vocab]

        if temperature == 0.0:
            # Special-case temperature 0 = greedy/argmax. Dividing by 0 below would give inf/nan,
            # so we branch here instead. This makes output deterministic (top_p is irrelevant).
            next_id = int(torch.argmax(next_logits))
        else:
            # Temperature scaling: divide LOGITS by temperature before softmax. <1 sharpens the
            # distribution (more greedy), >1 flattens it (more random). Must happen pre-softmax.
            probs = softmax(next_logits / temperature, dim=-1)
            # Then restrict to the nucleus and renormalize.
            probs = top_p_filter(probs, top_p)
            # Sample one id from the resulting categorical distribution.
            next_id = int(torch.multinomial(probs, num_samples=1))

        # Stop as soon as the model emits EOT, and do NOT append it to the output.
        if next_id == eot_id:
            break

        generated.append(next_id)
        # Feed the sampled token back in for the next step (autoregression). Building the tensor
        # on `device` avoids a host<->device copy each iteration.
        x = torch.cat([x, torch.tensor([[next_id]], device=device)], dim=1)

    return tokenizer.decode(generated)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Sample completions from a trained TransformerLM.")

    # Prompt / sampling controls
    p.add_argument("--prompt", type=str, required=True)
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top-p", type=float, default=1.0)

    # Checkpoint + tokenizer
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--vocab-path", type=str, required=True)
    p.add_argument("--merges-path", type=str, required=True)

    # Model hyperparameters. These MUST match whatever the checkpoint was trained with:
    # the checkpoint holds only weights, so a mismatch makes load_state_dict fail on shape
    # (or, worse, silently build a wrong-sized model). Defaults mirror train.py's defaults.
    p.add_argument("--vocab-size", type=int, default=10_000)
    p.add_argument("--context-length", type=int, default=256)
    p.add_argument("--d-model", type=int, default=512)
    p.add_argument("--num-layers", type=int, default=4)
    p.add_argument("--num-heads", type=int, default=16)
    p.add_argument("--d-ff", type=int, default=1344)
    p.add_argument("--rope-theta", type=float, default=10_000.0)

    p.add_argument("--device", type=str, default="auto",
                   help='"auto", "cpu", "cuda", or "mps".')
    return p.parse_args()


def resolve_device(requested: str) -> str:
    if requested != "auto":
        return requested
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)

    # special_tokens=[END_OF_TEXT] is required so the EOT string maps to its single sentinel id
    # (see generate()); without it the stop condition can never fire.
    tokenizer = Tokenizer.from_files(
        args.vocab_path, args.merges_path, special_tokens=[END_OF_TEXT]
    )

    model = TransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        rope_theta=args.rope_theta,
    ).to(device)
    # load_model (in Serialization.py) restores weights ONLY -- no optimizer needed for inference,
    # unlike load_checkpoint. map_location=device lets a GPU/MPS-trained checkpoint load on CPU.
    load_model(args.checkpoint, model, map_location=device)

    completion = generate(
        model,
        tokenizer,
        prompt=args.prompt,
        context_length=args.context_length,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        device=device,
    )

    print(args.prompt, end="")
    print(completion)


if __name__ == "__main__":
    main()
