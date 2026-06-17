"""
Dashboard Streamlit — Détection d'Émotions Faciales FER2013
Partie 3 — Meissa MARA

Lancement :
    streamlit run app/streamlit_app.py
"""

import sys
from pathlib import Path

import numpy as np
import streamlit as st
import torch
from PIL import Image

# ── Résolution des chemins ────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from transfer_model import FERResNet, load_fer_resnet  # noqa: E402
from training import get_device  # noqa: E402

# ── Constantes ────────────────────────────────────────────────────────────────
CHECKPOINT = ROOT / "data" / "processed" / "fer_resnet_best.pt"

EMOTIONS = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]

EMOTION_EMOJIS = {
    "Angry":    "😠",
    "Disgust":  "🤢",
    "Fear":     "😨",
    "Happy":    "😄",
    "Sad":      "😢",
    "Surprise": "😲",
    "Neutral":  "😐",
}

EMOTION_COLORS = {
    "Angry":    "#e74c3c",
    "Disgust":  "#8e44ad",
    "Fear":     "#2c3e50",
    "Happy":    "#f1c40f",
    "Sad":      "#3498db",
    "Surprise": "#e67e22",
    "Neutral":  "#95a5a6",
}


# ── Chargement du modèle (mis en cache par Streamlit) ─────────────────────────
@st.cache_resource
def load_model() -> FERResNet:
    """Charge le FERResNet depuis le checkpoint ou initialise à vide en fallback."""
    device = get_device()
    if CHECKPOINT.exists():
        return load_fer_resnet(str(CHECKPOINT), device)
    # Fallback : modèle non entraîné (pour démo sans checkpoint)
    st.warning(
        "⚠️ Checkpoint non trouvé. "
        "Lancez d'abord `notebooks/06_dl_avance.ipynb` pour entraîner le modèle."
    )
    model = FERResNet(pretrained=False)
    model.to(device)
    model.eval()
    return model


# ── Prétraitement de l'image ───────────────────────────────────────────────────
def preprocess(pil_img: Image.Image) -> torch.Tensor:
    """
    Convertit une image PIL arbitraire en tenseur FER2013.
    Pipeline : RGBA/RGB → grayscale → resize 48×48 → normalisation [0,1] → (1,1,48,48)
    """
    img = pil_img.convert("L").resize((48, 48), Image.LANCZOS)
    arr = np.array(img, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)  # (1, 1, 48, 48)


# ── Inférence ─────────────────────────────────────────────────────────────────
@torch.no_grad()
def predict(model: FERResNet, tensor: torch.Tensor) -> tuple[str, np.ndarray]:
    """Renvoie (classe prédite, tableau de probabilités)."""
    device = next(model.parameters()).device
    tensor = tensor.to(device)
    logits = model(tensor)
    probs = torch.softmax(logits, dim=1).squeeze().cpu().numpy()
    return EMOTIONS[int(probs.argmax())], probs


# ── Interface Streamlit ───────────────────────────────────────────────────────
st.set_page_config(
    page_title="Détection d'Émotions",
    page_icon="🎭",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Sidebar ---
with st.sidebar:
    st.title("🎭 FER2013 Dashboard")
    st.markdown("**Projet Deep Learning Avancé**")
    st.markdown("Partie 3 — *Meissa MARA*")
    st.divider()

    st.subheader("ℹ️ Modèle")
    st.markdown(
        """
        - **Architecture** : ResNet-18 fine-tuné  
        - **Dataset** : FER2013 (35 887 images)  
        - **Classes** : 7 émotions  
        - **Technique** : Transfer Learning (ImageNet → Facial)
        """
    )
    st.divider()

    st.subheader("📋 Émotions détectées")
    for emo, emoji in EMOTION_EMOJIS.items():
        st.markdown(f"- {emoji} &nbsp; **{emo}**", unsafe_allow_html=True)

    st.divider()
    checkpoint_status = "✅ Chargé" if CHECKPOINT.exists() else "⚠️ Non trouvé"
    st.caption(f"Checkpoint : {checkpoint_status}")

# --- Corps principal ---
st.title("🎭 Détection d'Émotions Faciales")
st.caption("Modèle ResNet-18 fine-tuné sur FER2013 · Partie 3 Deep Learning Avancé")
st.markdown("---")

model = load_model()

col_upload, col_result = st.columns([1, 1], gap="large")

with col_upload:
    st.subheader("📤 Image d'entrée")
    uploaded = st.file_uploader(
        "Chargez une photo de visage (JPG, PNG, WEBP…)",
        type=["jpg", "jpeg", "png", "bmp", "webp"],
        help="L'image sera convertie en niveaux de gris 48×48 pour correspondre au format FER2013.",
    )

    if uploaded is not None:
        pil_img = Image.open(uploaded)
        st.image(pil_img, caption="Image chargée", use_column_width=True)
        st.caption(f"Taille originale : {pil_img.size[0]}×{pil_img.size[1]} px")
    else:
        st.info("👆 Chargez une image pour lancer la détection d'émotion.")

with col_result:
    st.subheader("🔍 Prédiction")

    if uploaded is not None:
        with st.spinner("Analyse en cours…"):
            tensor = preprocess(pil_img)
            emotion, probs = predict(model, tensor)

        color = EMOTION_COLORS[emotion]
        emoji = EMOTION_EMOJIS[emotion]

        # Résultat principal
        st.markdown(
            f"""
            <div style="
                background: {color}22;
                border: 2px solid {color};
                border-radius: 12px;
                padding: 20px;
                text-align: center;
                margin-bottom: 20px;
            ">
                <span style="font-size:3rem">{emoji}</span>
                <h2 style="color:{color}; margin:8px 0 0 0">{emotion}</h2>
                <p style="color:#666; margin:4px 0 0 0">
                    Confiance : <strong>{probs.max()*100:.1f}%</strong>
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Barres de confiance
        st.markdown("**Distribution des probabilités :**")
        sorted_indices = np.argsort(probs)[::-1]
        for i in sorted_indices:
            emo = EMOTIONS[i]
            prob = probs[i]
            bar_color = EMOTION_COLORS[emo]
            width_px = int(prob * 280)

            st.markdown(
                f"""
                <div style="display:flex; align-items:center; gap:10px; margin-bottom:6px">
                    <span style="min-width:30px; font-size:1.1rem">{EMOTION_EMOJIS[emo]}</span>
                    <span style="min-width:75px; font-size:0.9rem">{emo}</span>
                    <div style="
                        background: {bar_color};
                        width: {width_px}px;
                        height: 18px;
                        border-radius: 4px;
                        min-width: 2px;
                    "></div>
                    <span style="font-size:0.9rem; color:#555">{prob*100:.1f}%</span>
                </div>
                """,
                unsafe_allow_html=True,
            )
    else:
        st.markdown(
            """
            <div style="
                background: #f8f9fa;
                border: 1px dashed #ccc;
                border-radius: 12px;
                padding: 40px;
                text-align: center;
                color: #aaa;
            ">
                <p style="font-size:2rem">🤔</p>
                <p>En attente d'une image…</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

# --- Section d'information supplémentaire ---
st.markdown("---")
with st.expander("📚 Informations techniques sur le modèle"):
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Architecture", "ResNet-18")
        st.metric("Paramètres", "~11.2M")
    with col2:
        st.metric("Dataset", "FER2013")
        st.metric("Nb classes", "7 émotions")
    with col3:
        st.metric("Technique", "Transfer Learning")
        st.metric("Input size", "48×48 → 112×112")

    st.markdown(
        """
        **Fonctionnement du Transfer Learning :**  
        1. **Phase 1** (Feature Extraction) : Le backbone ResNet-18 est gelé.  
           Seule la tête de classification est entraînée sur FER2013.  
        2. **Phase 2** (Fine-tuning) : Les derniers blocs résiduels (`layer3`, `layer4`) sont dégelés.  
           Le réseau s'adapte aux spécificités des émotions faciales.
        """
    )
