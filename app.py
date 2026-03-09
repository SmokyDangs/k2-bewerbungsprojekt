import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2

# Seiteneinstellungen
st.set_page_config(page_title="YOLO12 Image Detector", layout="wide")

def load_model(model_path):
    """Lädt das YOLO Modell sicher."""
    try:
        model = YOLO(model_path)
        return model
    except Exception as e:
        st.error(f"Fehler beim Laden des Modells: {e}")
        return None

def main():
    st.title("🚀 YOLO12 Objekt-Erkennung")
    st.write("Lade ein Bild hoch, um Objekte mit dem neuesten YOLO12 Modell zu erkennen.")

    # Sidebar für Einstellungen
    st.sidebar.header("Konfiguration")
    model_type = st.sidebar.selectbox(
        "Wähle das Modell-Gewicht",
        ["best.pt"]
    )
    conf_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.25)

    # Modell laden
    model = load_model(model_type)

    # Datei-Upload
    uploaded_file = st.file_uploader("Bild auswählen...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Bild öffnen
        image = Image.open(uploaded_file)
        
        # Layout: Vorher vs. Nachher
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Original")
            st.image(image, use_container_width=True)

        # Verarbeitung starten
        if model:
            with st.spinner('Objekte werden erkannt...'):
                # Vorhersage (Inferenz)
                results = model.predict(image, conf=conf_threshold)
                
                # Resultat-Bild generieren (als Numpy Array)
                res_plotted = results[0].plot()
                
                # Konvertierung von BGR (OpenCV) zu RGB (PIL/Streamlit)
                res_image = cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB)

            with col2:
                st.subheader("Ergebnis")
                st.image(res_image, use_container_width=True)

            # Statistiken anzeigen
            st.divider()
            st.subheader("Gefundene Objekte")
            
            # Extrahiere Namen der Klassen und deren Anzahl
            detected_classes = results[0].boxes.cls.tolist()
            names = model.names
            
            if len(detected_classes) > 0:
                counts = {}
                for obj in detected_classes:
                    name = names[int(obj)]
                    counts[name] = counts.get(name, 0) + 1
                
                # Tabellarische Anzeige
                st.write(counts)
            else:
                st.info("Keine Objekte mit der gewählten Confidence gefunden.")

if __name__ == "__main__":
    main()
