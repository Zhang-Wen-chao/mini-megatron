import torch
import os


def save_checkpoint(model, optimizer, scheduler, step, path):
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    state = {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "step": step,
    }
    torch.save(state, path)


def load_checkpoint(model, optimizer, scheduler, path, device):
    state = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(state["model_state"])
    optimizer.load_state_dict(state["optimizer_state"])
    scheduler.load_state_dict(state["scheduler_state"])
    return state["step"]
