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
*(à compléter)*

## Lancer le projet

### Installation des dépendances
    pip install -r requirements.txt

### Téléchargement du dataset
Configurer le fichier .env avec vos identifiants Kaggle :

    KAGGLE_USERNAME=votre_username
    KAGGLE_KEY=votre_token

### Lancer les notebooks
    jupyter lab
