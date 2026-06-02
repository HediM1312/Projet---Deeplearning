"""
Architectures Deep Learning pour la détection d'émotions FER2013.
Partie 2 — Hedi MATHLOUTHI
"""

from typing import List

import torch
import torch.nn as nn


class EmotionMLP(nn.Module):
    """
    Perceptron multicouche : baseline DL sur pixels aplatis (2304 features).
    Équivalent conceptuel à la baseline ML sklearn, mais avec couches non-linéaires.
    """

    def __init__(
        self,
        input_size: int = 2304,
        hidden_sizes: tuple = (512, 256),
        num_classes: int = 7,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        layers: List[nn.Module] = []
        prev = input_size

        for hidden in hidden_sizes:
            layers.extend([
                nn.Linear(prev, hidden),
                nn.BatchNorm1d(hidden),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            ])
            prev = hidden

        layers.append(nn.Linear(prev, num_classes))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        return self.network(x)


class EmotionCNN(nn.Module):
    """
    CNN pour images 48x48 : exploite la spatialité (yeux, bouche, sourcils).
    BatchNorm + ReLU pour limiter vanishing/exploding gradients.
    """

    def __init__(self, num_classes: int = 7, dropout: float = 0.5) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        # 48 -> 24 -> 12 -> 6
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 6 * 6, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.classifier(x)


def count_parameters(model: nn.Module) -> int:
    """Nombre de paramètres entraînables."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def describe_architecture(model: nn.Module, name: str) -> None:
    """Affiche un résumé de l'architecture."""
    print(f"\n{'=' * 50}")
    print(f"Architecture : {name}")
    print(f"{'=' * 50}")
    print(model)
    print(f"\nParamètres entraînables : {count_parameters(model):,}")
