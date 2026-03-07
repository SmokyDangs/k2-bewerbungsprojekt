import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image
import io

# --- SEITEN KONFIGURATION ---
st.set_page_config(
    page_title="ProCrack AI Analyzer v2",
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Modernes Design via CSS
st.markdown("""
    <style>
    .main { background-color: #f0f2f6; }
    .stMetric { background-color: #ffffff; padding: 15px; border-radius: 10px; border: 1px solid #e0e0e0; }
    .stAlert { border-radius: 10px; }
    </style>
    """, unsafe_allow_html=True)

# --- MODELL LADEN ---
@st.cache_resource
def load_model():
    # Lädt das Modell. Stelle sicher, dass 'best.pt' im gleichen Verzeichnis liegt.
    model_path = "best.pt" 
    return YOLO(model_path)

# --- HELPER FUNKTIONEN ---
def get_image_download_link(img_array):
    img = Image.fromarray(img_array)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()

# --- HAUPTPROGRAMM ---
def main():
    st.title("🏗️ ProCrack AI: Intelligente Schadensanalyse")
    st.markdown("---")

    # Sidebar
    st.sidebar.header("🛠️ Analyse-Optionen")
    conf_threshold = st.sidebar.slider("KI-Konfidenz (Empfindlichkeit)", 0.0, 1.0, 0.25, 0.05)
    mask_opacity = st.sidebar.slider("Masken-Sichtbarkeit", 0.0, 1.0, 0.4, 0.1)
    
    show_labels = st.sidebar.checkbox("Labels anzeigen", value=True)
    show_boxes = st.sidebar.checkbox("Boxen anzeigen", value=False)

    try:
        model = load_model()
        st.sidebar.success("✅ KI-Modell bereit")
    except Exception as e:
        st.sidebar.error("❌ Modell 'best.pt' fehlt!")
        st.error("Bitte laden Sie die trainierte Modelldatei 'best.pt' in das Projektverzeichnis hoch.")
        st.stop()

    # Datei-Upload
    uploaded_file = st.file_uploader("Bild zur Rissanalyse auswählen...", type=["jpg", "jpeg", "png", "webp"])

    if uploaded_file:
        # Bildverarbeitung
        image = Image.open(uploaded_file).convert("RGB")
        img_array = np.array(image)

        with st.spinner('Analyse läuft...'):
            results = model.predict(source=img_array, conf=conf_threshold)
            result = results[0]

        # Anzeige Spalten
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Original")
            st.image(image, use_container_width=True)

        with col2:
            st.subheader("KI-Erkennung")
            
            # Fehlerfreies Plotten (Behebung des TypeError)
            try:
                # 'mask_alpha' ist der korrekte Parameter für Segmentation
                res_plotted = result.plot(
                    conf=show_labels, 
                    labels=show_labels, 
                    boxes=show_boxes,
                    mask_alpha=mask_opacity
                )
            except TypeError:
                # Sicherheits-Fallback bei Versionskonflikten
                res_plotted = result.plot()

            res_rgb = cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB)
            st.image(res_rgb, use_container_width=True)
            
            # Download Button
            btn_data = get_image_download_link(res_rgb)
            st.download_button(
                label="💾 Analyse als Bild speichern",
                data=btn_data,
                file_name=f"analysis_{uploaded_file.name}",
                mime="image/png"
            )

        # --- AUSWERTUNG ---
        st.markdown("---")
        st.subheader("📊 Analyse-Statistiken")
        
        if result.masks is not None:
            num_cracks = len(result.masks)
            
            # Flächenberechnung
            total_pixels = img_array.shape[0] * img_array.shape[1]
            # Summiere alle Pixel der binären Masken
            crack_pixels = sum([mask.data.sum().item() for mask in result.masks])
            percentage = (crack_pixels / total_pixels) * 100

            # Metriken anzeigen
            m1, m2, m3 = st.columns(3)
            m1.metric("Anzahl Detektionen", f"{num_cracks}")
            m2.metric("Betroffene Fläche", f"{percentage:.3f} %")
            m3.metric("Status", "Auffällig" if percentage > 0.5 else "Unauffällig")

            # Warnhinweis basierend auf Fläche
            if percentage > 1.0:
                st.warning("⚠️ Achtung: Signifikante Rissbildung erkannt. Statische Prüfung empfohlen.")
            else:
                st.info("ℹ️ Kleinere Rissbildungen detektiert. Beobachtung empfohlen.")
            
            # Daten-Tabelle (Optionaler Klapptext)
            with st.expander("Rohdaten anzeigen"):
                st.write(f"Gesamtpixel im Bild: {total_pixels}")
                st.write(f"Segmentierte Riss-Pixel: {int(crack_pixels)}")
        else:
            st.success("✅ Keine Risse gefunden. Die Oberfläche scheint strukturell gesund zu sein.")
            st.balloons()

    else:
        st.info("Bitte laden Sie ein Foto hoch (z.B. von Betonwänden, Asphalt oder Mauerwerk).")

if __name__ == "__main__":
    main()
