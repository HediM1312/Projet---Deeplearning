# Projet Deep Learning — Détection d'Émotions (FER2013)

## Équipe — Rois de Jérusalem

| Nom | Rôle |
|-----|------|
| Hassan HOUSSEIN HOUMED | Partie 1 : Data Science & Baseline ML |
| Hedi MATHLOUTHI | Partie 2 : Deep Learning Fondamental |
| Meissa MARA | Partie 3 : Deep Learning Avancé & Ingénierie |

## Description

Projet de Deep Learning sur la détection d'émotions faciales
à partir du dataset FER2013 (35 887 images, 7 émotions).

## Structure du projet

 Projet-Deeplearning/
    ├── README.md
    ├── requirements.txt
    ├── .gitignore
    ├── data/
    │   ├── raw/          # Dataset FER2013
    │   └── processed/    # Données préparées
    ├── notebooks/
    │   ├── 01_EDA.ipynb              # Exploration des données
    │   ├── 02_baseline_ml.ipynb      # Baseline ML
    │   ├── 03_dl_architecture.ipynb  # Jalon 5 — Architectures DL
    │   ├── 04_dl_optimization.ipynb  # Jalon 6 — Optimisation
    │   └── 05_dl_comparison.ipynb    # Jalon 7 — ML vs DL
    └── src/
        ├── data_loader.py
        ├── dataset.py
        ├── models.py
        ├── training.py
        └── utils.py

## Partie 1 — Data Science & Baseline ML — Hassan HOUSSEIN HOUMED

### Dataset
- FER2013 : 35 887 images en niveaux de gris 48x48 pixels
- 7 émotions : Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral
- Ratio train/test : 4.0 (28 709 train, 7 178 test)
- Déséquilibre détecté : Disgust 1.52% vs Happy 25.1%

### Étapes réalisées
- Téléchargement automatique via API Kaggle
- Analyse exploratoire complète (EDA)
- Détection des anomalies (aucune trouvée)
- Augmentation de données pour les classes minoritaires
- Réduction dimensionnelle PCA (100 composantes, 90.2% variance)
- Baseline ML avec régularisation Ridge, Lasso, ElasticNet

### Résultats Baseline ML

| Modèle | Accuracy | Temps |
|--------|----------|-------|
| Ridge (L2) | 30.27% | 0.1s |
| Lasso (L1) | 30.46% | 6.4s |
| ElasticNet (L1+L2) | 30.46% | 7.0s |

### Conclusion
Le ML classique atteint 30% d'accuracy sur FER2013.
Un underfitting est détecté : le modèle est trop simple
pour capturer les relations spatiales entre pixels.
Le CNN sera nécessaire pour dépasser cette baseline.

## Partie 2 — Deep Learning Fondamental — Hedi MATHLOUTHI

### Choix d'architecture (Jalon 5)
- **RNN** : rejeté (images statiques, pas de séquence temporelle)
- **MLP** : référence DL sur pixels aplatis (comparable à la baseline ML)
- **CNN** : modèle principal — convolutions + pooling pour capturer la spatialité faciale
- Anti vanishing/exploding : ReLU, BatchNorm, gradient clipping, Dropout, class weights

### Optimisation (Jalon 6)
- Grille d'hyperparamètres : learning rate, batch size, dropout, optimiseur (Adam vs SGD)
- Scheduler `ReduceLROnPlateau` sur l'accuracy de validation
- Modèle final sauvegardé : `data/processed/cnn_best.pt`

### Comparaison (Jalon 7)
- Benchmark ML (Partie 1) vs MLP DL vs CNN optimisé
- Analyse : performance, limites, temps d'entraînement et d'inférence
- Figures : `notebooks/figures/comparaison_ml_vs_dl.png`

### Modules Python (`src/`)
- `models.py` : `EmotionMLP`, `EmotionCNN`
- `dataset.py` : `FER2013TensorDataset`, DataLoaders
- `training.py` : entraînement, évaluation, recherche HP

### Données Partie 2 (sans modifier la Partie 1)
- Fichier dédié : `data/processed/data_dl_full.pkl` (dataset **complet** depuis `data/raw/`)
- La Partie 1 conserve son `data/processed/data.pkl` (sous-échantillon)
- Création auto au premier `ensure_dl_full_data()` dans les notebooks 03–05

### Exécution
    jupyter lab
    # Ordre : 03_dl_architecture → 04_dl_optimization → 05_dl_comparison
    # Prérequis : data/raw/ (images FER2013) + PyTorch CUDA recommandé

## Partie 3 — Deep Learning Avancé & Ingénierie — Meissa MARA

### Jalon 8 — Technologie avancée : Transfer Learning (ResNet-18)

**Justification du besoin technique**

| Contrainte | Problème | Solution Transfer Learning |
|---|---|---|
| Dataset limité | ~35k images 48×48, bruité | Backbone pré-entraîné sur 1.2M images ImageNet |
| CNN from-scratch (Partie 2) | Accuracy ~55–60%, risque d'overfitting | Features ImageNet transférables aux visages |
| Déséquilibre de classes | Disgust 1.5% vs Happy 25% | Class weights + augmentation conservée |

**Architecture FERResNet** (`src/transfer_model.py`) :
- ResNet-18 pré-entraîné (ImageNet) comme backbone
- `conv1` adapté : 3 canaux → 1 canal (poids moyennés, sans perdre les features)
- Upsampling bilinéaire 48×48 → 112×112 avant le backbone
- Tête de classification : `Dropout(0.4) → Linear(512, 7)`

**Stratégie d'entraînement deux phases** :
1. **Phase 1 — Feature Extraction** (10 époques) : backbone gelé, seule la tête est entraînée
2. **Phase 2 — Fine-tuning** (15 époques) : `layer3` + `layer4` dégelés, lr réduit + CosineAnnealingLR

### Résultats Transfer Learning

| Modèle | Accuracy test | Paramètres | Technique |
|--------|:---:|:---:|---|
| Ridge / Lasso (Partie 1) | ~30% | — | ML classique |
| EmotionMLP (Partie 2) | ~36% | 1.2M | DL baseline |
| EmotionCNN (Partie 2) | ~57% | 2.3M | CNN from-scratch |
| **FERResNet (Partie 3)** | **~63–68%** | **11.2M** | **Transfer Learning** |

### Jalon 9 — Dashboard Streamlit (`app/streamlit_app.py`)

- Upload d'une image de visage (JPG, PNG, WEBP…)
- Prédiction de l'émotion avec barre de confiance colorée par classe
- Sidebar informatif (architecture, classes, statut du checkpoint)
- Interface responsive deux colonnes (image | résultat)

**Lancement :**
```bash
streamlit run app/streamlit_app.py
```

### Nouveaux fichiers (Partie 3)

| Fichier | Rôle |
|---------|------|
| `src/transfer_model.py` | Architecture FERResNet + utilitaires save/load |
| `notebooks/06_dl_avance.ipynb` | Entraînement Transfer Learning + évaluation |
| `app/streamlit_app.py` | Dashboard interactif de démonstration |

### Transparence IA

Les outils d'IA suivants ont été utilisés dans cette partie :
- **GitHub Copilot (Claude Sonnet 4.6)** : génération et structuration du code (`transfer_model.py`, `streamlit_app.py`, `06_dl_avance.ipynb`)
- Toute l'architecture, la justification technique, la stratégie d'entraînement deux phases et les choix de design du dashboard ont été conçus et validés manuellement
- Limites rencontrées : le fine-tuning sur CPU est lent (~15 min/époque) ; l'utilisation de GPU CUDA est fortement recommandée

## Lancer le projet

### Installation des dépendances
    pip install -r requirements.txt

### Téléchargement du dataset
Configurer le fichier .env avec vos identifiants Kaggle :

    KAGGLE_USERNAME=votre_username
    KAGGLE_KEY=votre_token

### Lancer les notebooks
    jupyter lab
