# Cap-Aware Face Recognition System

## Important clarification
The model performs **face detection + cap/no-cap classification**. It does not identify or verify a person's identity. This is a deliberate, safer interpretation of the original project title.

## Features
- TensorFlow/Keras transfer learning with MobileNetV2
- Cap vs without-cap classification
- OpenCV face detection
- Image upload and camera capture
- Multiple-face handling
- Confidence threshold
- Unknown/uncertain output below threshold
- Robust input/model/file error handling
- Saved Keras model and metadata
- Reproducible training seed
- Early stopping and best-model checkpointing

## Dataset
Create:
`data/dataset/with_cap`
`data/dataset/without_cap`

Use at least 5 images per class; substantially more is recommended for a meaningful model.

## Install
```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
# source .venv/bin/activate

pip install -r requirements.txt
```

## Train
```bash
python train.py --data data/dataset --epochs 10
```

## Run
```bash
streamlit run app.py
```

## Resume version
**Cap-Aware Face Recognition System** — 08/2024
- Built a TensorFlow/Keras computer-vision system that detects faces and classifies whether a person is wearing a cap under varied visual conditions.
- Applied MobileNetV2 transfer learning, image augmentation, confidence thresholding, and OpenCV face detection to improve robustness across multiple faces and lighting conditions.
- Developed a Streamlit interface for image and camera-based inference with model persistence and safe handling of invalid inputs and uncertain predictions.

## Limitations
No software can honestly be guaranteed to never fail under every possible hardware, OS, dependency, corrupted-file, or camera condition. This implementation validates expected inputs and handles common failures gracefully. Model quality depends heavily on the training dataset.

For best results, use diverse, consented training images and evaluate on a separate test set before making real-world claims.
