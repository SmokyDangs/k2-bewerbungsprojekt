import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image, ImageOps
import io
import pandas as pd

# Optionaler Import mit Fallback-Logik für den Slider-Vergleich
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

# Custom CSS für optimiertes Layout
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stMetric { background-color: white; border-left: 5px solid #007bff; padding: 15px; border-radius: 5px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
    [data-testid="stVerticalBlock"] > div:has(div.stMetric) { margin-bottom: 10px; }
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

    # Sidebar (Kompakt gehalten)
    st.sidebar.header("⚙️ Konfiguration")
    conf_threshold = st.sidebar.slider("KI-Sensitivität", 0.0, 1.0, 0.25, 0.05)
    use_autocontrast = st.sidebar.checkbox("Bild-Optimierung (Auto-Kontrast)", value=True)
    
    try:
        model = load_model()
        st.sidebar.success("✅ Modell aktiv")
    except Exception:
        st.sidebar.error("❌ 'best.pt' fehlt")
        st.stop()

    uploaded_file = st.file_uploader("Bild zur Analyse hochladen...", type=["jpg", "png", "webp"])

    if uploaded_file:
        # Bildverarbeitung
        raw_image = Image.open(uploaded_file).convert("RGB")
        processed_image = optimize_image(raw_image) if use_autocontrast else raw_image
        img_array = np.array(processed_image)

        with st.spinner('Analysiere Oberflächenstruktur...'):
            results = model.predict(source=img_array, conf=conf_threshold)
            result = results[0]

        # --- HAUPTFENSTER LAYOUT ---
        # Spalte 1: Bildvergleich | Spalte 2: Auswertung & Export
        col_main_left, col_main_right = st.columns([3, 2], gap="large")

        with col_main_left:
            st.subheader("🔍 Visueller Vergleich")
            
            # Plotten (feste Transparenz von 0.5 für optimale Sichtbarkeit)
            try:
                res_plotted = result.plot(conf=True, labels=True, mask_alpha=0.5)
            except TypeError:
                res_plotted = result.plot()
            
            res_rgb = cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB)
            
            if HAS_COMPARISON:
                image_comparison(
                    img1=raw_image,
                    img2=Image.fromarray(res_rgb),
                    label1="Original",
                    label2="KI-Analyse",
                    starting_position=50
                )
            else:
                st.image(res_rgb, caption="KI-Ergebnis (Segmentation)", use_container_width=True)

        with col_main_right:
            st.subheader("📊 Auswertung")
            
            if result.masks is not None:
                num_cracks = len(result.masks)
                total_pixels = img_array.shape[0] * img_array.shape[1]
                crack_pixels = sum([m.data.sum().item() for m in result.masks])
                percentage = (crack_pixels / total_pixels) * 100

                # Metriken untereinander
                st.metric("Detektierte Risse", f"{num_cracks} Segmente")
                st.metric("Schadensfläche", f"{percentage:.4f} %")
                
                status_color = "🔴 Kritisch" if percentage > 0.5 else "🟢 Tolerierbar"
                st.metric("Statusbewertung", status_color)

                st.markdown("---")
                st.subheader("📂 Export & Protokoll")
                
                # CSV Daten vorbereiten
                df = pd.DataFrame({
                    "Parameter": ["Datei", "Segmente", "Pixel Gesamt", "Riss-Pixel", "Fläche %"],
                    "Wert": [uploaded_file.name, num_cracks, total_pixels, int(crack_pixels), f"{percentage:.5f}"]
                })

                c_down1, c_down2 = st.columns(2)
                with c_down1:
                    csv = df.to_csv(index=False).encode('utf-8')
                    st.download_button("📊 CSV Export", csv, "messprotokoll.csv", "text/csv", use_container_width=True)
                
                with c_down2:
                    img_byte_arr = io.BytesIO()
                    Image.fromarray(res_rgb).save(img_byte_arr, format='PNG')
                    st.download_button("🖼️ Bild Export", img_byte_arr.getvalue(), "analyse.png", "image/png", use_container_width=True)
                
                # JSON Log ganz unten in der rechten Spalte
                with st.expander("🛠️ Technisches Log (JSON)"):
                    try:
                        st.json(result.to_json())
                    except:
                        st.write("JSON-Daten nicht verfügbar.")
            else:
                st.success("✅ Keine strukturellen Risse gefunden.")
                st.balloons()

    else:
        st.info("Bitte laden Sie ein Foto hoch, um die Analyse zu starten.")

if __name__ == "__main__":
    main()
