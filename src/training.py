"""
Boucles d'entraînement, évaluation et recherche d'hyperparamètres.
Partie 2 — Hedi MATHLOUTHI
"""

import random
import time
from typing import Any, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm


def set_seed(seed: int = 42) -> None:
    """Reproductibilité des expériences."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        return torch.device("cuda")
    return torch.device("cpu")


def build_optimizer(
    model: nn.Module,
    name: str = "adam",
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
) -> torch.optim.Optimizer:
    """
    Construit l'optimiseur.
    Adam : convergence rapide, adaptatif (recommandé pour CNN).
    SGD+momentum : baseline classique, souvent plus stable en généralisation.
    """
    params = model.parameters()
    name = name.lower()
    if name == "adam":
        return Adam(params, lr=lr, weight_decay=weight_decay)
    if name == "sgd":
        return SGD(params, lr=lr, momentum=0.9, weight_decay=weight_decay, nesterov=True)
    raise ValueError(f"Optimiseur inconnu : {name}")


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    max_grad_norm: float = 1.0,
    show_batch_progress: bool = False,
) -> Tuple[float, float]:
    """Une époque d'entraînement avec gradient clipping."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    batch_iter = tqdm(loader, desc="Train", leave=False) if show_batch_progress else loader
    for X_batch, y_batch in batch_iter:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()

        if max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

        optimizer.step()

        total_loss += loss.item() * y_batch.size(0)
        correct += (outputs.argmax(1) == y_batch).sum().item()
        total += y_batch.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float, np.ndarray, np.ndarray]:
    """Évaluation sur un jeu de données."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    all_preds: List[int] = []
    all_labels: List[int] = []

    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)

        total_loss += loss.item() * y_batch.size(0)
        preds = outputs.argmax(1)
        correct += (preds == y_batch).sum().item()
        total += y_batch.size(0)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(y_batch.cpu().numpy())

    return (
        total_loss / total,
        correct / total,
        np.array(all_labels),
        np.array(all_preds),
    )


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    epochs: int = 15,
    lr: float = 1e-3,
    optimizer_name: str = "adam",
    weight_decay: float = 1e-4,
    class_weights: Optional[torch.Tensor] = None,
    device: Optional[torch.device] = None,
    use_scheduler: bool = True,
    max_grad_norm: float = 1.0,
    verbose: bool = True,
    early_stop_patience: int = 8,
    show_batch_progress: bool = False,
) -> dict[str, Any]:
    """
    Entraîne un modèle et retourne l'historique + meilleur état.
    """
    device = device or get_device()
    model = model.to(device)

    criterion = nn.CrossEntropyLoss(
        weight=class_weights.to(device) if class_weights is not None else None
    )
    optimizer = build_optimizer(model, optimizer_name, lr, weight_decay)
    scheduler = (
        ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)
        if use_scheduler
        else None
    )

    history: dict[str, list] = {
        "train_loss": [],
        "train_acc": [],
        "test_loss": [],
        "test_acc": [],
    }
    best_acc = 0.0
    best_state = None
    epochs_without_improve = 0
    start = time.time()

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            max_grad_norm,
            show_batch_progress=show_batch_progress,
        )
        test_loss, test_acc, _, _ = evaluate(
            model, test_loader, criterion, device
        )

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["test_loss"].append(test_loss)
        history["test_acc"].append(test_acc)

        if scheduler:
            scheduler.step(test_acc)

        if test_acc > best_acc:
            best_acc = test_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            epochs_without_improve = 0
        else:
            epochs_without_improve += 1

        if early_stop_patience > 0 and epochs_without_improve >= early_stop_patience:
            if verbose:
                print(f"Early stopping (pas d'amélioration depuis {early_stop_patience} epochs)")
            break

        if verbose:
            print(
                f"Epoch {epoch:02d}/{epochs} | "
                f"train_loss={train_loss:.4f} acc={train_acc*100:.2f}% | "
                f"test_acc={test_acc*100:.2f}%"
            )

    if best_state:
        model.load_state_dict(best_state)

    duration = time.time() - start
    _, final_acc, y_true, y_pred = evaluate(
        model, test_loader, criterion, device
    )

    return {
        "model": model,
        "history": history,
        "best_test_acc": best_acc,
        "final_test_acc": final_acc,
        "y_true": y_true,
        "y_pred": y_pred,
        "train_time_sec": duration,
        "device": str(device),
    }


@torch.no_grad()
def measure_inference_time(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    n_warmup: int = 5,
) -> float:
    """Temps d'inférence moyen par batch (secondes)."""
    model.eval()
    batches = list(loader)
    if not batches:
        return 0.0

    for _ in range(n_warmup):
        X, _ = batches[0]
        model(X.to(device))

    start = time.perf_counter()
    for X, _ in batches:
        model(X.to(device))
    elapsed = time.perf_counter() - start
    return elapsed / len(batches)
