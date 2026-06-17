"""
Transfer Learning pour la détection d'émotions FER2013.
Partie 3 — Meissa MARA

Justification technique du Transfer Learning :
- FER2013 = ~35k images 48x48 niveaux de gris, dataset relativement petit
- Le CNN from-scratch (Partie 2) est limité par la quantité de données annotées
- ResNet-18 pré-entraîné sur ImageNet (1.2M images) a déjà appris :
    * Détecteurs de bords et textures (couches basses)
    * Représentations spatiales abstraites (couches hautes)
    * Ces features sont transférables à la reconnaissance faciale
- Stratégie deux phases :
    1. Feature extraction : backbone gelé, seule la tête est entraînée
    2. Fine-tuning : dégel progressif des derniers blocs résiduels
"""

from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tv_models
from torchvision.models import ResNet18_Weights


class FERResNet(nn.Module):
    """
    ResNet-18 fine-tuné pour la classification d'émotions FER2013 (7 classes).

    Adaptations par rapport au ResNet-18 standard :
    - Input : images 48x48 niveaux de gris (1 canal)
    - Upsampling bilinéaire 48 → 112 avant passage dans le backbone
    - Première conv modifiée : 3 canaux → 1 canal (poids moyennés)
    - Tête de classification : 512 → Dropout → Linear(7)
    """

    def __init__(
        self,
        num_classes: int = 7,
        dropout: float = 0.4,
        pretrained: bool = True,
    ) -> None:
        super().__init__()
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        backbone = tv_models.resnet18(weights=weights)

        # Adapter la première convolution 3-canaux → 1 canal
        # On moyenne les poids des 3 canaux pour conserver les features ImageNet
        orig_conv = backbone.conv1
        new_conv = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        new_conv.weight.data = orig_conv.weight.data.mean(dim=1, keepdim=True)
        backbone.conv1 = new_conv

        # Remplacer la tête de classification (1000 classes → 7)
        in_features = backbone.fc.in_features
        backbone.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, num_classes),
        )

        self.backbone = backbone

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Upsampling 48x48 → 112x112 pour coller au receptive field ResNet
        x = F.interpolate(x, size=(112, 112), mode="bilinear", align_corners=False)
        return self.backbone(x)

    def freeze_backbone(self) -> None:
        """
        Phase 1 — Feature extraction :
        Gèle tout le backbone, seule la tête fc est entraînée.
        Utile quand le dataset cible est petit ou similaire à ImageNet.
        """
        for name, param in self.backbone.named_parameters():
            if "fc" not in name:
                param.requires_grad_(False)

    def unfreeze_last_layers(self, n_blocks: int = 2) -> None:
        """
        Phase 2 — Fine-tuning partiel :
        Dégèle les n derniers blocs résiduels + la tête.
        Permet d'adapter les features de haut niveau à FER2013.
        """
        all_layers = ["layer1", "layer2", "layer3", "layer4"]
        layers_to_unfreeze = all_layers[-n_blocks:]
        for name, param in self.backbone.named_parameters():
            if any(name.startswith(ln) for ln in layers_to_unfreeze) or "fc" in name:
                param.requires_grad_(True)

    def unfreeze_all(self) -> None:
        """Dégèle tous les paramètres pour un fine-tuning complet."""
        for param in self.backbone.parameters():
            param.requires_grad_(True)

    def count_trainable_params(self) -> int:
        """Renvoie le nombre de paramètres entraînables."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def count_total_params(self) -> int:
        """Renvoie le nombre total de paramètres."""
        return sum(p.numel() for p in self.parameters())


def load_fer_resnet(
    checkpoint_path: str,
    device: torch.device,
    num_classes: int = 7,
) -> FERResNet:
    """
    Charge un FERResNet depuis un checkpoint PyTorch sauvegardé.

    Args:
        checkpoint_path: Chemin vers le fichier .pt
        device: Device cible (cpu / cuda)
        num_classes: Nombre de classes (7 pour FER2013)

    Returns:
        Modèle chargé en mode évaluation.
    """
    model = FERResNet(num_classes=num_classes, pretrained=False)
    state = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def save_fer_resnet(model: FERResNet, checkpoint_path: str) -> None:
    """Sauvegarde l'état du modèle."""
    Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), checkpoint_path)
