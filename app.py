import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image, ImageOps
import io
import pandas as pd

# Optionaler Import mit Fallback-Logik
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

# Custom CSS für professionelles Look & Feel
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stMetric { background-color: white; border-left: 5px solid #007bff; padding: 15px; border-radius: 5px; }
    </style>
    """, unsafe_allow_html=True)

@st.cache_resource
def load_model():
    # Lädt das trainierte Modell
    return YOLO("best.pt")

def optimize_image(image):
    # Verbessert Kontrast für bessere KI-Erkennung
    img = ImageOps.exif_transpose(image)
    return ImageOps.autocontrast(img, cutoff=0.5)

# --- HAUPTPROGRAMM ---
def main():
    st.title("🛡️ CrackExpert AI Professional")
    st.caption("Präzisions-Segmentation für infrastrukturelle Schadensanalyse")

    # Sidebar
    st.sidebar.header("⚙️ Analyse-Konfiguration")
    conf_threshold = st.sidebar.slider("KI-Sensitivität", 0.0, 1.0, 0.25, 0.05)
    mask_opacity = st.sidebar.slider("Masken-Transparenz", 0.0, 1.0, 0.4, 0.1)
    
    st.sidebar.subheader("Bild-Optimierung")
    use_autocontrast = st.sidebar.checkbox("Auto-Kontrast (bei Schatten)", value=True)
    
    try:
        model = load_model()
        st.sidebar.success("✅ Modell geladen")
    except Exception as e:
        st.sidebar.error("❌ Modell 'best.pt' fehlt")
        st.stop()

    uploaded_file = st.file_uploader("Bild hochladen...", type=["jpg", "png", "webp"])

    if uploaded_file:
        # Bild laden & Vorbereiten
        raw_image = Image.open(uploaded_file).convert("RGB")
        processed_image = optimize_image(raw_image) if use_autocontrast else raw_image
        img_array = np.array(processed_image)

        # Inferenz
        with st.spinner('Führe strukturelle Analyse durch...'):
            results = model.predict(source=img_array, conf=conf_threshold)
            result = results[0]

        # --- VISUALISIERUNG ---
        st.subheader("🔍 Visuelle Analyse")
        
        # Plotten des Ergebnisses mit Fehlerbehandlung für den Parameter-Namen
        try:
            res_plotted = result.plot(conf=True, labels=True, mask_alpha=mask_opacity)
        except TypeError:
            res_plotted = result.plot()
        
        res_rgb = cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB)
        
        # Interaktiver Vergleich (nur wenn Modul installiert, sonst Side-by-Side)
        if HAS_COMPARISON:
            image_comparison(
                img1=raw_image,
                img2=Image.fromarray(res_rgb),
                label1="Original",
                label2="KI-Analyse",
                starting_position=50
            )
        else:
            col_img1, col_img2 = st.columns(2)
            col_img1.image(raw_image, caption="Original", use_container_width=True)
            col_img2.image(res_rgb, caption="KI-Analyse", use_container_width=True)

        # --- DATEN & EXPORT ---
        st.markdown("---")
        col_stats, col_export = st.columns([2, 1])

        if result.masks is not None:
            num_cracks = len(result.masks)
            total_pixels = img_array.shape[0] * img_array.shape[1]
            crack_pixels = sum([m.data.sum().item() for m in result.masks])
            percentage = (crack_pixels / total_pixels) * 100

            with col_stats:
                st.subheader("📊 Messdaten")
                m1, m2, m3 = st.columns(3)
                m1.metric("Anzahl Risse", num_cracks)
                m2.metric("Schadensfläche", f"{percentage:.4f} %")
                m3.metric("Gefahrenstufe", "HOCH" if percentage > 0.5 else "GERING")

                report_data = {
                    "Parameter": ["Dateiname", "Anzahl Segmente", "Pixel Gesamt", "Riss-Pixel", "Flächenanteil %"],
                    "Wert": [uploaded_file.name, num_cracks, total_pixels, int(crack_pixels), f"{percentage:.5f}"]
                }
                df = pd.DataFrame(report_data)

            with col_export:
                st.subheader("📂 Export")
                csv = df.to_csv(index=False).encode('utf-8')
                st.download_button("📊 CSV Protokoll", csv, "messdaten.csv", "text/csv")
                
                img_byte_arr = io.BytesIO()
                Image.fromarray(res_rgb).save(img_byte_arr, format='PNG')
                st.download_button("🖼️ Bild speichern", img_byte_arr.getvalue(), "analyse.png", "image/png")
        else:
            st.success("Keine Risse gefunden!")

        # --- EXPERTEN-LOG (KORRIGIERT) ---
        with st.expander("📝 Technisches Protokoll (Raw Data)"):
            # FIX: .to_json() statt .tojson()
            try:
                st.json(result.to_json())
            except Exception as e:
                st.warning("JSON-Export für dieses Modell-Ergebnis nicht verfügbar.")

if __name__ == "__main__":
    main()
