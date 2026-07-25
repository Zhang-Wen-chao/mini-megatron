"""Create identity dataset (labels = input_ids) from the synthetic dataset."""
import torch
d = torch.load("experiments/synthetic_data.pt")
d["labels"] = d["input_ids"].clone()
torch.save(d, "experiments/identity_data.pt")
print("OK, saved identity_data.pt")
