import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    scaler: "torch.amp.GradScaler | None" = None,
    grad_clip_max_norm: float = 1.0,
) -> dict:
    """`scaler` is only needed for mixed precision — pass a
    `torch.amp.GradScaler` to train in fp16/bf16 autocast, or leave it
    `None` for plain fp32. VRAM is tight on a 4GB card (RTX 3050), so AMP
    matters more here than it would on a larger GPU.

    Gradients are clipped to `grad_clip_max_norm` before every optimizer
    step — a cheap guard against occasional loss spikes destabilizing
    training; pass `None` to disable.
    """
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    use_amp = scaler is not None

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()

        with torch.autocast(device_type=device.type, enabled=use_amp):
            outputs = model(images)
            loss = criterion(outputs, labels)

        if use_amp:
            scaler.scale(loss).backward()
            if grad_clip_max_norm is not None:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_max_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if grad_clip_max_norm is not None:
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_max_norm)
            optimizer.step()

        total_loss += loss.item() * images.size(0)
        correct += (outputs.argmax(dim=1) == labels).sum().item()
        total += images.size(0)

    return {"loss": total_loss / total, "accuracy": correct / total}


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    use_amp: bool = False,
) -> dict:
    """`y_true`/`y_pred` are returned alongside the aggregate loss/accuracy so
    callers can compute macro-F1, confusion matrices, and other per-class
    metrics without a second pass over the data.
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    y_true: list[int] = []
    y_pred: list[int] = []

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        with torch.autocast(device_type=device.type, enabled=use_amp):
            outputs = model(images)
            loss = criterion(outputs, labels)

        preds = outputs.argmax(dim=1)
        total_loss += loss.item() * images.size(0)
        correct += (preds == labels).sum().item()
        total += images.size(0)
        y_true.extend(labels.cpu().tolist())
        y_pred.extend(preds.cpu().tolist())

    return {
        "loss": total_loss / total,
        "accuracy": correct / total,
        "y_true": y_true,
        "y_pred": y_pred,
    }
