"""Generate a deterministic synthetic dataset for training comparison.

Pattern: next-token prediction on a repeating modulo sequence.
`labels[i] = (input_ids[i] + 1) % effective_vocab`

This is learnable by a small GPT model (copy the input pattern forward).
"""
import torch
import os


def generate_synthetic(num_steps, B, S, effective_vocab, seed=42):
    """Generate [num_steps, B, S] tensors with a learnable pattern."""
    torch.manual_seed(seed)
    full = torch.randint(0, effective_vocab, (num_steps, B, S))
    input_ids = full.clone()
    labels = (full + 1) % effective_vocab
    return input_ids, labels


def generate_and_save(
    save_path,
    num_steps=100,
    B=4,
    S=512,
    effective_vocab=1024,
    model_vocab=50304,
    seed=42,
):
    """Generate and save to .pt file."""
    input_ids, labels = generate_synthetic(num_steps, B, S, effective_vocab, seed)
    data = {
        "input_ids": input_ids,  # [num_steps, B, S]
        "labels": labels,  # [num_steps, B, S]
        "effective_vocab": effective_vocab,
        "model_vocab": model_vocab,
        "B": B,
        "S": S,
        "num_steps": num_steps,
        "seed": seed,
    }
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    torch.save(data, save_path)
    print(f"Saved synthetic data to {save_path}")
    print(f"  shape: [{num_steps}, {B}, {S}], pattern: (x+1)%{effective_vocab}")
    return data


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "experiments/synthetic_100steps.pt"
    generate_and_save(path)
