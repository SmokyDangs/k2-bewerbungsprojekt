import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image

# --- SEITEN KONFIGURATION ---
st.set_page_config(page_title="Crack Seg AI", layout="wide")

st.title("🏗️ KI-Rissanalyse (Segmentation)")
st.write("Lade ein Foto von einer Wand oder Straße hoch, um Risse automatisch zu segmentieren.")

# --- MODELL LADEN ---
# Hier den Pfad zu deiner heruntergeladenen 'best.pt' angeben
@st.cache_resource
def load_model():
    # Falls die Datei lokal im selben Ordner liegt:
    model_path = "best.pt" 
    return YOLO(model_path)

try:
    model = load_model()
    st.success("Modell erfolgreich geladen!")
except Exception as e:
    st.error(f"Modell 'best.pt' nicht gefunden. Bitte lade die Datei in den Ordner hoch. Fehler: {e}")
    st.stop()

# --- SIDEBAR & EINSTELLUNGEN ---
st.sidebar.header("Einstellungen")
conf_threshold = st.sidebar.slider("Konfidenz-Schwellenwert", 0.0, 1.0, 0.25, 0.05)

# --- DATEI UPLOAD ---
uploaded_file = st.file_uploader("Bild auswählen...", type=["jpg", "jpeg", "png", "webp"])

if uploaded_file is not None:
    # Bild konvertieren
    image = Image.open(uploaded_file)
    img_array = np.array(image)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Originalbild")
        st.image(image, use_container_width=True)

    with col2:
        st.subheader("Analyse-Ergebnis")
        
        # Vorhersage treffen
        results = model.predict(source=img_array, conf=conf_threshold)
        
        # Das Ergebnisbild mit Masken zeichnen
        res_plotted = results[0].plot()
        
        # Von BGR (OpenCV) zu RGB (Streamlit) konvertieren
        res_rgb = cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB)
        st.image(res_rgb, use_container_width=True)

    # --- STATISTIKEN ---
    st.divider()
    if results[0].masks is not None:
        num_cracks = len(results[0].masks)
        st.metric("Gefundene Riss-Segmente", num_cracks)
        
        # Optional: Berechnung der Fläche (Pixel-basiert)
        total_area = 0
        for mask in results[0].masks.data:
            total_area += mask.sum().item()
        
        st.info(f"Geschätzte rissige Fläche: {int(total_area)} Pixel")
    else:
        st.warning("Keine Risse erkannt.")

else:
    st.info("Bitte lade ein Bild hoch, um die Analyse zu starten.")
