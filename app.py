
from __future__ import annotations
import json
from pathlib import Path
import cv2
import numpy as np
import streamlit as st
from PIL import Image
import tensorflow as tf

BASE = Path(__file__).resolve().parent
MODEL_PATH = BASE / "models/cap_classifier.keras"
META_PATH = BASE / "models/metadata.json"
IMG_SIZE = (160, 160)
DEFAULT_CLASSES = ["without_cap", "with_cap"]

st.set_page_config(page_title="Cap-Aware Face Recognition", page_icon="🧢", layout="wide")

@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        return None
    return tf.keras.models.load_model(MODEL_PATH)

@st.cache_resource
def load_detector():
    cascade = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    detector = cv2.CascadeClassifier(cascade)
    return None if detector.empty() else detector

@st.cache_data
def load_classes():
    if META_PATH.exists():
        try:
            data = json.loads(META_PATH.read_text(encoding="utf-8"))
            if data.get("classes"):
                return data["classes"]
        except (OSError, json.JSONDecodeError):
            pass
    return DEFAULT_CLASSES

def classify_face(model, face_bgr, classes):
    rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, IMG_SIZE, interpolation=cv2.INTER_AREA)
    arr = np.asarray(resized, dtype=np.float32)[None, ...]
    probs = model.predict(arr, verbose=0)[0]
    idx = int(np.argmax(probs))
    confidence = float(probs[idx])
    label = classes[idx] if idx < len(classes) else "unknown"
    return label, confidence

def annotate(frame, model, detector, classes, threshold=0.60):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
    results = []
    for x, y, w, h in faces:
        face = frame[y:y+h, x:x+w]
        if face.size == 0:
            continue
        try:
            label, confidence = classify_face(model, face, classes)
        except Exception:
            label, confidence = "prediction_error", 0.0
        display = label if confidence >= threshold else "uncertain"
        results.append((display, confidence, int(x), int(y), int(w), int(h)))
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        text = f"{display} ({confidence:.0%})"
        cv2.putText(frame, text, (x, max(25, y-8)), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0,255,0), 2)
    return frame, results

def main():
    st.title("🧢 Cap-Aware Face Recognition System")
    st.caption("Real-time face detection + cap/no-cap classification using TensorFlow/Keras.")

    model = load_model()
    detector = load_detector()
    classes = load_classes()

    if model is None:
        st.error("Model not found. Train it first with: python train.py")
        st.stop()
    if detector is None:
        st.error("OpenCV face detector could not be loaded.")
        st.stop()

    threshold = st.slider("Confidence threshold", 0.50, 0.95, 0.60, 0.05)
    mode = st.radio("Input", ["Upload image", "Webcam"], horizontal=True)

    if mode == "Upload image":
        uploaded = st.file_uploader("Upload JPG, JPEG, PNG or WEBP", type=["jpg","jpeg","png","webp"])
        if uploaded:
            try:
                image = Image.open(uploaded).convert("RGB")
                frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
                output, results = annotate(frame, model, detector, classes, threshold)
                st.image(cv2.cvtColor(output, cv2.COLOR_BGR2RGB), use_container_width=True)
                st.write(f"Faces detected: {len(results)}")
                for label, conf, *_ in results:
                    st.write(f"- **{label}** — {conf:.1%}")
            except Exception as exc:
                st.error("The uploaded image could not be processed.")
                st.exception(exc)

    else:
        st.info("Webcam mode uses Streamlit's camera input. Capture a frame, then analyze it.")
        camera = st.camera_input("Take a photo")
        if camera:
            try:
                image = Image.open(camera).convert("RGB")
                frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
                output, results = annotate(frame, model, detector, classes, threshold)
                st.image(cv2.cvtColor(output, cv2.COLOR_BGR2RGB), use_container_width=True)
                st.write(f"Faces detected: {len(results)}")
                for label, conf, *_ in results:
                    st.write(f"- **{label}** — {conf:.1%}")
            except Exception as exc:
                st.error("Camera frame could not be processed.")
                st.exception(exc)

if __name__ == "__main__":
    main()
