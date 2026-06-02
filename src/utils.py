import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

# Labels des émotions FER2013
EMOTIONS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

def plot_emotion_distribution(labels: list, title: str = "Distribution des émotions") -> None:
    """
    Affiche la distribution des émotions dans le dataset.
    """
    plt.figure(figsize=(10, 5))
    sns.countplot(x=labels, order=range(len(EMOTIONS)))
    plt.xticks(range(len(EMOTIONS)), EMOTIONS, rotation=45)
    plt.title(title)
    plt.xlabel("Émotion")
    plt.ylabel("Nombre d'images")
    plt.tight_layout()
    plt.show()


def plot_confusion_matrix(y_true, y_pred, title: str = "Matrice de Confusion") -> None:
    """
    Affiche la matrice de confusion.
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', 
                xticklabels=EMOTIONS, 
                yticklabels=EMOTIONS,
                cmap='Blues')
    plt.title(title)
    plt.ylabel('Vrai label')
    plt.xlabel('Prédit')
    plt.tight_layout()
    plt.show()


def print_classification_report(y_true, y_pred) -> None:
    """
    Affiche le rapport de classification.
    """
    print(classification_report(y_true, y_pred, target_names=EMOTIONS))


def plot_training_history(history: dict, title: str = "Courbes d'entraînement") -> None:
    """Affiche loss et accuracy train/test."""
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(epochs, history["train_loss"], label="Train")
    axes[0].plot(epochs, history["test_loss"], label="Test")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs, [a * 100 for a in history["train_acc"]], label="Train")
    axes[1].plot(epochs, [a * 100 for a in history["test_acc"]], label="Test")
    axes[1].set_title("Accuracy (%)")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.suptitle(title)
    plt.tight_layout()


def save_figure(fig_name: str, folder: str = "figures") -> None:
    """
    Sauvegarde une figure matplotlib.
    """
    os.makedirs(folder, exist_ok=True)
    plt.savefig(os.path.join(folder, fig_name), bbox_inches='tight', dpi=150)
    print(f"Figure sauvegardée : {folder}/{fig_name}")