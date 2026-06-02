import os
from dotenv import load_dotenv

# Chargement des variables d'environnement
load_dotenv()

def download_fer2013(dest_folder: str = "data/raw") -> None:
    """
    Télécharge le dataset FER2013 depuis Kaggle.
    """
    import kaggle
    
    os.makedirs(dest_folder, exist_ok=True)
    
    print("Téléchargement du dataset FER2013...")
    kaggle.api.authenticate()
    kaggle.api.dataset_download_files(
        "msambare/fer2013",
        path=dest_folder,
        unzip=True
    )
    print(f"Dataset téléchargé dans : {dest_folder}")


def get_data_path() -> str:
    """
    Retourne le chemin vers les données.
    """
    return os.path.join("data", "raw")