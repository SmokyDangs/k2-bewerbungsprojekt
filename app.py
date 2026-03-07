import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image
import io

# --- SEITEN KONFIGURATION ---
st.set_page_config(
    page_title="ProCrack AI Analyzer",
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS für ein moderneres Interface
st.markdown("""
    <style>
    .main {
        background-color: #f5f7f9;
    }
    .stMetric {
        background-color: #ffffff;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    </style>
    """, unsafe_allow_html=True)

# --- MODELL LADEN ---
@st.cache_resource
def load_model():
    model_path = "best.pt"  # Stelle sicher, dass die Datei im selben Ordner liegt
    return YOLO(model_path)

# --- HELPER FUNKTIONEN ---
def get_image_download_link(img_array, filename="analyse_ergebnis.png"):
    img = Image.fromarray(img_array)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()

# --- HAUPTPROGRAMM ---
def main():
    st.title("🏗️ ProCrack AI: Professionelle Rissanalyse")
    st.info("Dieses System nutzt YOLO-Segmentation zur präzisen Erkennung von Bauschäden.")

    # Sidebar für präzise Steuerung
    st.sidebar.image("https://ultralytics.com/static/images/logo-social-post.png", width=200)
    st.sidebar.header("Analyse-Parameter")
    
    conf_threshold = st.sidebar.slider("Konfidenz (KI-Sicherheit)", 0.0, 1.0, 0.25, 0.05)
    mask_opacity = st.sidebar.slider("Masken-Transparenz", 0.0, 1.0, 0.5, 0.1)
    
    st.sidebar.divider()
    st.sidebar.markdown("### System-Status")
    try:
        model = load_model()
        st.sidebar.success("✅ Modell aktiv")
    except Exception as e:
        st.sidebar.error("❌ Modell fehlt")
        st.error("Bitte lade die 'best.pt' Datei in das Verzeichnis hoch.")
        st.stop()

    # Datei Upload Bereich
    uploaded_file = st.file_uploader("Bild zur Analyse hochladen...", type=["jpg", "jpeg", "png", "webp"])

    if uploaded_file:
        # Bild laden
        image = Image.open(uploaded_file).convert("RGB")
        img_array = np.array(image)

        with st.spinner('KI analysiert Bildstruktur...'):
            # Inferenz
            results = model.predict(source=img_array, conf=conf_threshold)
            result = results[0]

        # Layout Spalten
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Originalaufnahme")
            st.image(image, use_container_width=True)

        with col2:
            st.subheader("KI-Segmentierung")
            # Ergebnis plotten mit Transparenz-Einstellungen
            res_plotted = result.plot(conf=True, labels=True, alpha=mask_opacity)
            res_rgb = cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB)
            st.image(res_rgb, use_container_width=True)
            
            # Download Button
            btn_data = get_image_download_link(res_rgb)
            st.download_button(
                label="💾 Ergebnisbild speichern",
                data=btn_data,
                file_name=f"riss_analyse_{uploaded_file.name}",
                mime="image/png"
            )

        # --- DETAILLIERTE STATISTIKEN ---
        st.divider()
        st.subheader("Analyse-Bericht")
        
        if result.masks is not None:
            m1, m2, m3 = st.columns(3)
            
            num_cracks = len(result.masks)
            # Fläche berechnen
            total_pixels = img_array.shape[0] * img_array.shape[1]
            crack_pixels = sum([mask.sum().item() for mask in result.masks.data])
            percentage = (crack_pixels / total_pixels) * 100

            m1.metric("Anzahl Risse", f"{num_cracks} Segmente")
            m2.metric("Betroffene Fläche", f"{percentage:.2f} %")
            m3.metric("Bildauflösung", f"{img_array.shape[1]}x{img_array.shape[0]}")

            if percentage > 5:
                st.warning("⚠️ Kritischer Schwellenwert überschritten: Großflächige Rissbildung erkannt.")
            else:
                st.success("✅ Oberflächenzustand innerhalb der Toleranzgrenzen.")
        else:
            st.balloons()
            st.success("Keine Risse detektiert! Die Oberfläche scheint intakt zu sein.")

    else:
        # Platzhalter wenn kein Bild geladen ist
        st.info("Warten auf Bildeingabe...")
        # Optional: Ein Beispielbild anzeigen
        
if __name__ == "__main__":
    main()
