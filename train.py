
"""
Train a cap/no-cap image classifier.

Dataset structure:
data/dataset/
    with_cap/
        image1.jpg
        ...
    without_cap/
        image1.jpg
        ...

The script uses transfer learning with MobileNetV2 and saves the complete
Keras model plus class names. It validates the dataset before training.
"""
from __future__ import annotations
import argparse, json, logging
from pathlib import Path
import tensorflow as tf
from tensorflow.keras import layers, models

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
LOG = logging.getLogger("cap-trainer")

IMG_SIZE = (160, 160)
SEED = 42
EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

def count_images(folder: Path) -> int:
    return sum(1 for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in EXTS)

def validate_dataset(data_dir: Path) -> tuple[Path, Path]:
    classes = ["without_cap", "with_cap"]
    paths = [data_dir / c for c in classes]
    missing = [str(p) for p in paths if not p.is_dir()]
    if missing:
        raise FileNotFoundError(
            "Missing class folders: " + ", ".join(missing) +
            ". Create data/dataset/with_cap and data/dataset/without_cap."
        )
    counts = {c: count_images(data_dir / c) for c in classes}
    if min(counts.values()) < 5:
        raise ValueError(f"Each class needs at least 5 images. Found: {counts}")
    LOG.info("Dataset counts: %s", counts)
    return paths[0], paths[1]

def build_model(num_classes: int = 2) -> tf.keras.Model:
    base = tf.keras.applications.MobileNetV2(
        input_shape=(*IMG_SIZE, 3), include_top=False, weights="imagenet"
    )
    base.trainable = False
    inputs = layers.Input(shape=(*IMG_SIZE, 3))
    x = layers.RandomFlip("horizontal")(inputs)
    x = layers.RandomRotation(0.05)(x)
    x = layers.RandomZoom(0.10)(x)
    x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.25)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    model = models.Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data/dataset"))
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--output", type=Path, default=Path("models/cap_classifier.keras"))
    args = parser.parse_args()

    if args.epochs < 1 or args.epochs > 100:
        raise ValueError("--epochs must be between 1 and 100.")
    if args.batch_size < 1 or args.batch_size > 256:
        raise ValueError("--batch-size must be between 1 and 256.")

    validate_dataset(args.data)

    train_ds = tf.keras.utils.image_dataset_from_directory(
        args.data, labels="inferred", label_mode="int",
        class_names=["without_cap", "with_cap"],
        image_size=IMG_SIZE, batch_size=args.batch_size,
        validation_split=0.20, subset="training", seed=SEED,
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        args.data, labels="inferred", label_mode="int",
        class_names=["without_cap", "with_cap"],
        image_size=IMG_SIZE, batch_size=args.batch_size,
        validation_split=0.20, subset="validation", seed=SEED,
    )

    autotune = tf.data.AUTOTUNE
    train_ds = train_ds.prefetch(autotune)
    val_ds = val_ds.prefetch(autotune)

    model = build_model()
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint(args.output, monitor="val_accuracy", save_best_only=True),
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    history = model.fit(train_ds, validation_data=val_ds, epochs=args.epochs, callbacks=callbacks)

    # Save final best model and metadata.
    model.save(args.output)
    meta = {
        "classes": ["without_cap", "with_cap"],
        "image_size": list(IMG_SIZE),
        "seed": SEED,
        "epochs_requested": args.epochs,
        "final_train_accuracy": float(history.history["accuracy"][-1]),
        "final_val_accuracy": float(history.history["val_accuracy"][-1]),
    }
    (args.output.parent / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    LOG.info("Saved model: %s", args.output)

if __name__ == "__main__":
    main()
