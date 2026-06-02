"""
Dataset PyTorch — Partie 2 (sans modifier data.pkl de la Partie 1).
"""

import os
import pickle
import random
from typing import Literal, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms.functional import adjust_brightness, rotate

# Fichier dédié Partie 2 (dataset complet) — Partie 1 garde data.pkl
DL_FULL_PKL = "data/processed/data_dl_full.pkl"


class FER2013TensorDataset(Dataset):
    """Dataset FER2013 à partir de tableaux numpy (aplati ou image)."""

    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        mode: Literal["flat", "image"] = "image",
        augment: bool = False,
    ) -> None:
        self.X = X.astype(np.float32)
        self.y = y.astype(np.int64)
        self.mode = mode
        self.augment = augment

    def __len__(self) -> int:
        return len(self.y)

    def _apply_augment(self, x: torch.Tensor) -> torch.Tensor:
        """Augmentation légère (alignée sur l'EDA Partie 1)."""
        if random.random() < 0.5:
            x = torch.flip(x, dims=[2])
        if random.random() < 0.5:
            angle = random.uniform(-15, 15)
            x = rotate(x, angle)
        if random.random() < 0.5:
            factor = random.uniform(0.85, 1.15)
            x = adjust_brightness(x, factor)
        return x

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.X[idx]
        if self.mode == "image":
            x = torch.from_numpy(x.reshape(48, 48)).unsqueeze(0)
            if self.augment:
                x = self._apply_augment(x)
        else:
            x = torch.from_numpy(x)
        y = torch.tensor(self.y[idx], dtype=torch.long)
        return x, y


def build_processed_pkl(
    raw_path: str = "data/raw",
    pkl_path: str = "data/processed/data.pkl",
    max_per_class_train: int = 500,
    max_per_class_test: int = 150,
) -> dict:
    """Construit un .pkl depuis data/raw."""
    from PIL import Image

    from utils import EMOTIONS

    def _load_split(data_path: str, split: str, max_per_class: int) -> tuple:
        X, y = [], []
        for label, emotion in enumerate(EMOTIONS):
            folder = os.path.join(data_path, split, emotion)
            if not os.path.isdir(folder):
                raise FileNotFoundError(f"Dossier manquant : {folder}")
            files = sorted(os.listdir(folder))
            if max_per_class < len(files):
                files = files[:max_per_class]
            for fname in files:
                img = np.array(
                    Image.open(os.path.join(folder, fname)).convert("L")
                ).flatten()
                X.append(img)
                y.append(label)
        return np.array(X, dtype=np.float32), np.array(y, dtype=np.int64)

    print("Chargement train...")
    X_train, y_train = _load_split(raw_path, "train", max_per_class_train)
    print("Chargement test...")
    X_test, y_test = _load_split(raw_path, "test", max_per_class_test)

    X_train = X_train / 255.0
    X_test = X_test / 255.0

    data = {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "emotions": EMOTIONS,
    }

    os.makedirs(os.path.dirname(pkl_path) or ".", exist_ok=True)
    with open(pkl_path, "wb") as f:
        pickle.dump(data, f)

    print(f"Sauvegardé : {pkl_path}")
    print(f"  X_train : {X_train.shape}  |  X_test : {X_test.shape}")
    return data


def load_processed_data(
    pkl_path: str = "data/processed/data.pkl",
    auto_build: bool = True,
    raw_path: Optional[str] = None,
) -> dict:
    """Charge data.pkl (Partie 1 ou usage générique)."""
    if not os.path.isfile(pkl_path) and auto_build:
        if raw_path is None:
            raw_path = os.path.normpath(
                os.path.join(os.path.dirname(pkl_path), "..", "raw")
            )
        if os.path.isdir(os.path.join(raw_path, "train")):
            print(f"{pkl_path} introuvable — génération depuis {raw_path}...")
            build_processed_pkl(raw_path=raw_path, pkl_path=pkl_path)
        else:
            raise FileNotFoundError(
                f"{pkl_path} introuvable. Placez le dataset dans {raw_path}/"
            )
    with open(pkl_path, "rb") as f:
        return pickle.load(f)


def ensure_dl_full_data(
    pkl_path: str = DL_FULL_PKL,
    raw_path: Optional[str] = None,
    min_train_samples: int = 20000,
) -> dict:
    """
    Dataset complet pour la Partie 2 (55–62 % visés).
    N'écrase pas data.pkl utilisé par la Partie 1.
    """
    if raw_path is None:
        raw_path = os.path.normpath(
            os.path.join(os.path.dirname(pkl_path), "..", "raw")
        )

    rebuild = True
    if os.path.isfile(pkl_path):
        with open(pkl_path, "rb") as f:
            existing = pickle.load(f)
        if len(existing["y_train"]) >= min_train_samples:
            rebuild = False
            data = existing

    if rebuild:
        print("Partie 2 : construction du dataset complet (data_dl_full.pkl)...")
        data = build_processed_pkl(
            raw_path=raw_path,
            pkl_path=pkl_path,
            max_per_class_train=999_999,
            max_per_class_test=999_999,
        )
    else:
        print(f"Partie 2 : chargement {pkl_path} ({len(data['y_train'])} train)")

    return data


def create_dataloaders(
    data: dict,
    batch_size: int = 64,
    mode: Literal["flat", "image"] = "image",
    num_workers: int = 0,
    augment_train: bool = True,
) -> Tuple[DataLoader, DataLoader]:
    """Crée les DataLoaders train et test."""
    use_cuda = torch.cuda.is_available()
    train_ds = FER2013TensorDataset(
        data["X_train"],
        data["y_train"],
        mode=mode,
        augment=augment_train and mode == "image",
    )
    test_ds = FER2013TensorDataset(
        data["X_test"], data["y_test"], mode=mode, augment=False
    )

    loader_kw = {
        "num_workers": num_workers,
        "pin_memory": use_cuda,
    }
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, **loader_kw
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, **loader_kw
    )
    return train_loader, test_loader


def subsample_for_hp_search(
    data: dict,
    train_fraction: float = 0.35,
    test_fraction: float = 0.5,
    seed: int = 42,
) -> dict:
    """
    Sous-échantillon pour la recherche d'hyperparamètres (plus rapide).
    L'entraînement final utilise le dataset complet.
    """
    rng = np.random.default_rng(seed)
    n_train = int(len(data["y_train"]) * train_fraction)
    n_test = int(len(data["y_test"]) * test_fraction)
    idx_tr = rng.choice(len(data["y_train"]), size=n_train, replace=False)
    idx_te = rng.choice(len(data["y_test"]), size=n_test, replace=False)
    return {
        "X_train": data["X_train"][idx_tr],
        "y_train": data["y_train"][idx_tr],
        "X_test": data["X_test"][idx_te],
        "y_test": data["y_test"][idx_te],
        "emotions": data["emotions"],
    }


def get_class_weights(y_train: np.ndarray, num_classes: int = 7) -> torch.Tensor:
    """Poids inverses de fréquence pour gérer le déséquilibre des classes."""
    counts = np.bincount(y_train, minlength=num_classes).astype(np.float32)
    counts = np.maximum(counts, 1.0)
    weights = counts.sum() / (num_classes * counts)
    return torch.tensor(weights, dtype=torch.float32)
