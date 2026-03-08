import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image, ImageOps
import io
import pandas as pd

# --- OPTIONALER IMPORT FÜR BILDVERGLEICH ---
# Ermöglicht einen interaktiven Vorher-Nachher-Slider (erfordert: streamlit-image-comparison)
try:
    from streamlit_image_comparison import image_comparison
    HAS_COMPARISON = True
except ImportError:
    HAS_COMPARISON = False

# --- SEITEN KONFIGURATION ---
# Definiert Titel und Layout der Web-Applikation
st.set_page_config(
    page_title="CrackExpert AI Pro",
    page_icon="🛡️",
    layout="wide"
)

# --- CUSTOM CSS (UI/UX Optimierung) ---
# Hier wird das Standard-Design von Streamlit mit CSS angepasst, 
# um ein professionelles "Industrial Dashboard" Look-and-Feel zu erzeugen.
st.markdown("""
    <style>
    .main { background-color: #f0f2f6; }
    div[data-testid="stMetricValue"] { font-size: 1.8rem; font-weight: 700; color: #1f77b4; }
    .stMetric {
        background-color: white;
        border-radius: 12px;
        padding: 20px !important;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        border-top: 4px solid #1f77b4;
    }
    .stDownloadButton > button {
        width: 100%;
        border-radius: 8px;
        background-color: #1f77b4;
        color: white;
        font-weight: 600;
    }
    </style>
    """, unsafe_allow_html=True)

# --- MODELL-LADEFUNKTION ---
# @st.cache_resource verhindert, dass das schwere KI-Modell bei jeder Interaktion neu geladen wird (Performance-Boost)
@st.cache_resource
def load_model():
    return YOLO("best.pt") # Lädt dein trainiertes YOLOv8/v9/v10 Modell

# --- BILD-VORVERARBEITUNG ---
def optimize_image(image):
    # Korrigiert die Bildausrichtung basierend auf EXIF-Daten (wichtig für Handyfotos)
    img = ImageOps.exif_transpose(image)
    # Verbessert den Kontrast, um Risse für die KI deutlicher hervorzuheben
    return ImageOps.autocontrast(img, cutoff=0.5)

# --- HAUPTPROGRAMM ---
def main():
    st.title("🛡️ CrackExpert AI Professional")
    st.caption("Präzisions-Segmentation für infrastrukturelle Schadensanalyse")
    st.markdown("---")

    # --- SIDEBAR (Steuerungselemente) ---
    st.sidebar.header("⚙️ Analyse-Setup")
    # Regler für die KI-Konfidenz: Höherer Wert = weniger Fehlalarme, niedrigerer Wert = empfindlicher
    conf_threshold = st.sidebar.slider("KI-Sensitivität", 0.0, 1.0, 0.15, 0.05)
    use_autocontrast = st.sidebar.checkbox("Bild-Optimierung (Auto-Kontrast)", value=True)
    
    # Sicherstellung, dass das Modell vorhanden ist
    try:
        model = load_model()
        st.sidebar.success("✅ Modell: YOLO-Segmentierung aktiv")
    except Exception:
        st.sidebar.error("❌ 'best.pt' nicht im Ordner!")
        st.stop()

    # --- DATEI-UPLOAD ---
    uploaded_file = st.file_uploader("Bild zur Analyse hochladen...", type=["jpg", "png", "webp"])

    if uploaded_file:
        # Bild laden und in RGB konvertieren
        raw_image = Image.open(uploaded_file).convert("RGB")
        processed_image = optimize_image(raw_image) if use_autocontrast else raw_image
        img_array = np.array(processed_image)

        # --- KI-INFERENZ (Die eigentliche Erkennung) ---
        with st.spinner('KI berechnet Schadenssegmente...'):
            results = model.predict(source=img_array, conf=conf_threshold)
            result = results[0] # Wir nehmen das erste Ergebnis (Single Image)

        # --- LAYOUT AUFTEILUNG ---
        col_left, col_right = st.columns([3, 2], gap="large")

        # Visualisierung vorbereiten: Bounding Boxes und Masken auf das Bild zeichnen
        try:
            res_plotted_bgr = result.plot(conf=True, labels=True, mask_alpha=0.5)
            # OpenCV nutzt BGR, Streamlit/PIL brauchen RGB -> Umwandlung nötig
            res_rgb = cv2.cvtColor(res_plotted_bgr, cv2.COLOR_BGR2RGB)
            res_pil = Image.fromarray(res_rgb)
        except Exception:
            res_plotted_bgr = result.plot()
            res_rgb = cv2.cvtColor(res_plotted_bgr, cv2.COLOR_BGR2RGB)
            res_pil = Image.fromarray(res_rgb)

        # LINKER BEREICH: Visuelle Darstellung
        with col_left:
            st.subheader("🔍 Analyse-Visualisierung")
            if HAS_COMPARISON:
                # Interaktiver Slider: Original vs. KI-Ergebnis
                image_comparison(
                    img1=raw_image,
                    img2=res_pil,
                    label1="Original",
                    label2="KI-Segmentation",
                    starting_position=50
                )
            else:
                st.image(res_pil, caption="KI-Ergebnis inklusive Masken", use_container_width=True)

        # RECHTER BEREICH: Datenanalyse & Metriken
        with col_right:
            st.subheader("📊 Metriken & Bericht")
            
            # Überprüfung, ob Masken (Segmentation) gefunden wurden
            if result.masks is not None:
                num_cracks = len(result.masks) # Anzahl der separaten Risse
                total_pixels = img_array.shape[0] * img_array.shape[1]
                # Summiert alle Pixel der erkannten Masken auf
                crack_pixels = sum([m.data.sum().item() for m in result.masks])
                # Berechnung der Schadensfläche in Prozent
                percentage = (crack_pixels / total_pixels) * 100

                # Anzeige der Ergebnisse in ansprechenden Karten
                st.metric("Detektierte Risse", f"{num_cracks}")
                st.metric("Schadensfläche", f"{percentage:.4f} %")
                
                # Einfache Risikobewertung (Schwellenwert 0.5%)
                status = "🚨 KRITISCH" if percentage > 0.5 else "✅ STABIL"
                st.metric("Zustandsbewertung", status)

                st.markdown("---")
                st.subheader("📂 Ergebnis-Sicherung")
                
                # Pandas DataFrame für den Tabellen-Export erstellen
                df = pd.DataFrame({
                    "Metrik": ["Datei", "Anzahl Risse", "Fläche %", "Pixel-Anzahl"],
                    "Wert": [uploaded_file.name, num_cracks, f"{percentage:.5f}", int(crack_pixels)]
                })

                # --- EXPORT BUTTONS ---
                c1, c2 = st.columns(2)
                with c1:
                    csv = df.to_csv(index=False).encode('utf-8')
                    st.download_button("📊 Daten (CSV)", csv, f"crack_data_{uploaded_file.name}.csv", "text/csv")
                
                with c2:
                    # Bild mit Masken für den Download in Byte-Stream umwandeln
                    img_byte_arr = io.BytesIO()
                    res_pil.save(img_byte_arr, format='PNG')
                    st.download_button("🖼️ Bild + Masken", img_byte_arr.getvalue(), f"crack_mask_{uploaded_file.name}.png", "image/png")
                
                # Experten-Ansicht: Rohe JSON-Daten der KI anzeigen
                with st.expander("📝 Technisches JSON-Protokoll"):
                    try:
                        st.json(result.to_json())
                    except:
                        st.write("JSON-Export nicht unterstützt.")
            else:
                st.success("Keine Risse detektiert. Die Struktur scheint intakt.")
                st.balloons()
    else:
        st.info("Bereit für Analyse. Bitte laden Sie ein Bild hoch.")

# Einstiegspunkt des Skripts
if __name__ == "__main__":
    main()
