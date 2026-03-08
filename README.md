# <p align="center">🏗️ K2 Damage Detection AI</p>

<p align="center">
  <a href="https://k2-bewerbungsprojekt.streamlit.app/">
    <img src="https://img.shields.io/badge/LIVE_DEMO-APPLIKATION_STARTEN-1f77b4?style=for-the-badge&logo=streamlit&logoColor=white" alt="Live Demo">
  </a>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Deployment-Streamlit_Cloud-FF4B4B?style=flat-square&logo=streamlit" alt="Streamlit">
  <img src="https://img.shields.io/badge/Model-YOLO_Segmentation-00FFFF?style=flat-square" alt="YOLO">
  <img src="https://img.shields.io/badge/OS-Linux_Mint_Compatible-lightgrey?style=flat-square&logo=linux-mint" alt="Linux Mint">
  <img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python" alt="Python">
</p>

---

## 📖 Projekt-Beschreibung
Dieses Repository enthält eine einsatzbereite Web-Applikation zur **KI-gestützten Objekterkennung und Segmentation**. Das Projekt wurde als Proof-of-Concept entwickelt, um die effiziente Integration von Deep-Learning-Modellen in industrielle Workflows zu demonstrieren.

Im Fokus steht die automatisierte Analyse von Infrastrukturschäden (z. B. Risse in Beton- oder Asphaltflächen), um Inspektionsprozesse zu beschleunigen und menschliche Fehlerquellen zu minimieren.

---

## 🚀 Key Features

* **Echtzeit-Segmentation:** Hochoptimierte Inferenzzeiten für CPU- und GPU-Umgebungen mittels YOLOv8/v10.
* **Custom Precision Model:** Integration eines spezifisch trainierten Modells (`best.pt`) zur Erkennung feinster Oberflächenanomalien.
* **Industrial UI:** Schlankes Dashboard für den einfachen Upload und die sofortige visuelle Auswertung von Bilddaten.
* **Metrik-Extraktion:** Automatisierte Berechnung von Schadensflächen und statistische Aufbereitung der Befunde.
* **Modularer Aufbau:** Clean-Code-Architektur in `app.py` für maximale Skalierbarkeit und einfache Wartung.

---

## 🛠️ Tech Stack

* **Core:** [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) (State-of-the-Art Computer Vision)
* **Interface:** Streamlit (Reactive Web Framework)
* **Processing:** OpenCV, Pillow, NumPy
* **Analysis:** Pandas für den Daten-Export (CSV/JSON)

---

## 📦 Installation & Setup (Lokal unter Linux Mint/Ubuntu)

### 1. Repository klonen
```bash
git clone [https://github.com/SmokyDangs/k2-bewerbungsprojekt.git](https://github.com/SmokyDangs/k2-bewerbungsprojekt.git)
cd k2-bewerbungsprojekt
