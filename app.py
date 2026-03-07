import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image, ImageOps
import io
import pandas as pd

# Optionaler Import für Slider-Vergleich
try:
    from streamlit_image_comparison import image_comparison
    HAS_COMPARISON = True
except ImportError:
    HAS_COMPARISON = False

# --- SEITEN KONFIGURATION ---
st.set_page_config(
    page_title="CrackExpert AI Pro",
    page_icon="🛡️",
    layout="wide"
)

# --- VERBESSERTES CSS ---
st.markdown("""
    <style>
    /* Hintergrund und Font */
    .main { background-color: #f0f2f6; }
    
    /* Metric Cards Styling */
    div[data-testid="stMetricValue"] { font-size: 1.8rem; font-weight: 700; color: #1f77b4; }
    .stMetric {
        background-color: white;
        border-radius: 12px;
        padding: 20px !important;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        border-top: 4px solid #1f77b4;
    }
    
    /* Buttons Styling */
    .stDownloadButton > button {
        width: 100%;
        border-radius: 8px;
        background-color: #1f77b4;
        color: white;
        border: none;
        transition: all 0.3s ease;
        font-weight: 600;
    }
    .stDownloadButton > button:hover {
        background-color: #155a8a;
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    
    /* Sidebar Styling */
    .css-1d391kg { background-color: #1e2630; }
    .sidebar-text { color: white; font-weight: 500; }
    
    /* Header Styling */
    h1 { color: #1e2630; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; }
    </style>
    """, unsafe_allow_html=True)

@st.cache_resource
def load_model():
    return YOLO("best.pt")

def optimize_image(image):
    img = ImageOps.exif_transpose(image)
    return ImageOps.autocontrast(img, cutoff=0.5)

# --- HAUPTPROGRAMM ---
def main():
    st.title("🛡️ CrackExpert AI Professional")
    st.caption("Präzisions-Segmentation für infrastrukturelle Schadensanalyse")
    st.markdown("---")

    # Sidebar
    st.sidebar.header("⚙️ Analyse-Setup")
    conf_threshold = st.sidebar.slider("KI-Sensitivität", 0.0, 1.0, 0.15, 0.05)
    use_autocontrast = st.sidebar.checkbox("Bild-Optimierung (Auto-Kontrast)", value=True)
    
    # Modell-Status in Sidebar
    try:
        model = load_model()
        st.sidebar.success("✅ Modell: YOLO-Segmentierung aktiv")
    except Exception:
        st.sidebar.error("❌ 'best.pt' nicht im Ordner!")
        st.stop()

    uploaded_file = st.file_uploader("Bild zur Analyse hochladen...", type=["jpg", "png", "webp"])

    if uploaded_file:
        raw_image = Image.open(uploaded_file).convert("RGB")
        processed_image = optimize_image(raw_image) if use_autocontrast else raw_image
        img_array = np.array(processed_image)

        with st.spinner('KI berechnet Schadenssegmente...'):
            results = model.predict(source=img_array, conf=conf_threshold)
            result = results[0]

        # --- HAUPTFENSTER LAYOUT ---
        col_left, col_right = st.columns([3, 2], gap="large")

        # 1. Bild-Verarbeitung (Zentral für beide Spalten)
        try:
            # Hier stellen wir sicher, dass die Masken für den Export gerendert werden
            # mask_alpha=0.5 sorgt für Transparenz im Export-Bild
            res_plotted_bgr = result.plot(conf=True, labels=True, mask_alpha=0.5)
            res_rgb = cv2.cvtColor(res_plotted_bgr, cv2.COLOR_BGR2RGB)
            res_pil = Image.fromarray(res_rgb)
        except Exception:
            # Fallback falls Plotting-Parameter Probleme machen
            res_plotted_bgr = result.plot()
            res_rgb = cv2.cvtColor(res_plotted_bgr, cv2.COLOR_BGR2RGB)
            res_pil = Image.fromarray(res_rgb)

        with col_left:
            st.subheader("🔍 Analyse-Visualisierung")
            if HAS_COMPARISON:
                image_comparison(
                    img1=raw_image,
                    img2=res_pil,
                    label1="Original",
                    label2="KI-Segmentation",
                    starting_position=50
                )
            else:
                st.image(res_pil, caption="KI-Ergebnis inklusive Masken", use_container_width=True)

        with col_right:
            st.subheader("📊 Metriken & Bericht")
            
            if result.masks is not None:
                num_cracks = len(result.masks)
                total_pixels = img_array.shape[0] * img_array.shape[1]
                crack_pixels = sum([m.data.sum().item() for m in result.masks])
                percentage = (crack_pixels / total_pixels) * 100

                # Anzeige der Karten
                st.metric("Detektierte Risse", f"{num_cracks}")
                st.metric("Schadensfläche", f"{percentage:.4f} %")
                
                status = "🚨 KRITISCH" if percentage > 0.5 else "✅ STABIL"
                st.metric("Zustandsbewertung", status)

                st.markdown("---")
                st.subheader("📂 Ergebnis-Sicherung")
                
                # Daten-Tabelle
                df = pd.DataFrame({
                    "Metrik": ["Datei", "Anzahl Risse", "Fläche %", "Pixel-Anzahl"],
                    "Wert": [uploaded_file.name, num_cracks, f"{percentage:.5f}", int(crack_pixels)]
                })

                # Export Bereich
                c1, c2 = st.columns(2)
                with c1:
                    csv = df.to_csv(index=False).encode('utf-8')
                    st.download_button("📊 Daten (CSV)", csv, f"crack_data_{uploaded_file.name}.csv", "text/csv")
                
                with c2:
                    # Der Export enthält nun die 'res_pil', welche die Masken eingezeichnet hat
                    img_byte_arr = io.BytesIO()
                    res_pil.save(img_byte_arr, format='PNG')
                    st.download_button("🖼️ Bild + Masken", img_byte_arr.getvalue(), f"crack_mask_{uploaded_file.name}.png", "image/png")
                
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

if __name__ == "__main__":
    main()
