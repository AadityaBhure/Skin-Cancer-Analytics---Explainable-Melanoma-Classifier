"""Train the frozen EfficientNetB0 baseline; shared dataset/model helpers."""
import json
import os
from pathlib import Path
import random

ROOT = Path(__file__).resolve().parents[1]
# Keep downloads and runtime caches within the project.
os.environ.setdefault("KERAS_HOME", str(ROOT / "models/.keras"))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "outputs/.matplotlib"))
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "8")
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "2")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

SEED = 42
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 16
CLASS_NAMES = ["non_melanoma", "melanoma"]
MODEL_PATH = ROOT / "models/skin_cancer_efficientnetb0.keras"
METRICS = ROOT / "outputs/metrics"
FIGURES = ROOT / "outputs/figures"


def setup():
    random.seed(SEED)
    np.random.seed(SEED)
    tf.keras.utils.set_random_seed(SEED)
    for path in [MODEL_PATH.parent, METRICS, FIGURES, ROOT / "outputs/predictions"]:
        path.mkdir(parents=True, exist_ok=True)
    print("Using GPU." if tf.config.list_physical_devices("GPU") else "Using CPU (no GPU is accessible to this TensorFlow runtime).", flush=True)


def load_class_names():
    path = METRICS / "class_names.json"
    if not path.is_file():
        raise SystemExit("Missing class_names.json. Run src/train.py first.")
    names = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(names, list) or len(names) != 2 or set(names) != set(CLASS_NAMES):
        raise SystemExit("Stored class order is invalid; retrain with src/train.py.")
    return names


def load_model():
    if not MODEL_PATH.is_file():
        raise SystemExit("Missing trained model. Run python src/train.py first.")
    return tf.keras.models.load_model(MODEL_PATH, compile=False)


def dataset(split, class_names):
    path = ROOT / "data/processed" / split
    if not path.is_dir() or any(not (path / name).is_dir() for name in class_names):
        raise SystemExit(f"Missing prepared {split} data. Run python src/prepare_data.py first.")
    return tf.keras.utils.image_dataset_from_directory(
        path, class_names=class_names, label_mode="binary", image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE, shuffle=split == "train", seed=SEED)


def melanoma_probabilities(probabilities, class_names):
    values = np.asarray(probabilities).reshape(-1)
    return values if class_names.index("melanoma") == 1 else 1.0 - values


def main():
    setup()
    train = dataset("train", CLASS_NAMES)
    val = dataset("val", CLASS_NAMES)
    names = list(train.class_names)
    if names != list(val.class_names):
        raise SystemExit("Training and validation class orders differ.")
    augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal", seed=SEED),
        tf.keras.layers.RandomRotation(0.05, seed=SEED + 1),
        tf.keras.layers.RandomZoom(0.10, seed=SEED + 2),
    ], name="training_augmentation")
    train = train.map(lambda images, labels: (augmentation(images, training=True), labels), num_parallel_calls=1).prefetch(1)
    val = val.prefetch(1)
    # EfficientNet contains its own pixel rescaling: inputs must remain in [0, 255].
    # A flat functional graph also keeps convolutional tensors accessible to Grad-CAM.
    backbone = tf.keras.applications.EfficientNetB0(include_top=False, weights="imagenet", input_shape=(*IMAGE_SIZE, 3))
    backbone.trainable = False
    features = tf.keras.layers.GlobalAveragePooling2D(name="global_pool")(backbone.output)
    features = tf.keras.layers.Dropout(0.30, name="classifier_dropout")(features)
    output = tf.keras.layers.Dense(1, activation="sigmoid", name="binary_probability")(features)
    model = tf.keras.Model(backbone.input, output, name="skin_cancer_efficientnetb0")
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
                  loss="binary_crossentropy", metrics=["accuracy"])
    (METRICS / "class_names.json").write_text(json.dumps(names, indent=2), encoding="utf-8")
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=2, restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint(str(MODEL_PATH), monitor="val_loss", save_best_only=True),
    ]
    history = model.fit(train, validation_data=val, epochs=8, callbacks=callbacks, verbose=2, shuffle=False)
    values = pd.DataFrame(history.history)
    values.index = np.arange(1, len(values) + 1)
    values.index.name = "epoch"
    values.to_csv(METRICS / "training_history.csv")
    run = {"tensorflow_version": tf.__version__, "seed": SEED, "epochs_completed": len(values),
           "best_epoch": int(np.argmin(values.val_loss)) + 1, "class_names": names,
           "image_size": list(IMAGE_SIZE), "batch_size": BATCH_SIZE, "backbone_frozen": True,
           "trainable_parameters": int(sum(np.prod(weight.shape) for weight in model.trainable_weights))}
    (METRICS / "training_run.json").write_text(json.dumps(run, indent=2), encoding="utf-8")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, metric in zip(axes, ["accuracy", "loss"]):
        ax.plot(values.index, values[metric], marker="o", label="Train")
        ax.plot(values.index, values[f"val_{metric}"], marker="o", label="Validation")
        ax.set(xlabel="Epoch", ylabel=metric.capitalize(), title=f"Training and validation {metric}")
        ax.set_xticks(values.index)
        ax.legend()
        ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(FIGURES / "training_curves.png", dpi=160)
    plt.close(fig)
    print(f"Saved best validation-loss model: {MODEL_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
