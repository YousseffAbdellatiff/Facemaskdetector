# evaluate.py

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix
from data import get_data_generators

IMG_SIZE = 224
BATCH_SIZE = 32
MODEL_PATHS = {
    "Custom CNN": "cnn_model.h5",
    "MobileNetV2": "mobilenet_model.h5"
}

def evaluate_model(model_name, model_path):
    # 1️. Load the trained model
    model = load_model(model_path)
    print(f"\n=== Evaluating model: {model_name} ===")

    # 2️. Load validation data
    _, val_data = get_data_generators(
        data_path="dataset",
        img_size=IMG_SIZE,
        batch_size=BATCH_SIZE
    )

    # 3️. Run predictions
    y_prob = model.predict(val_data)
    y_pred = (y_prob > 0.5).astype(int).reshape(-1)
    y_true = val_data.classes

    # 4️. Compute metrics: precision, recall, F1-score
    report = classification_report(
        y_true, y_pred,
        target_names=list(val_data.class_indices.keys()),
        digits=4
    )
    print("Classification Report:\n", report)

    # 5️. Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    print("Confusion Matrix:\n", cm)

    #  Plot confusion matrix as heatmap
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=val_data.class_indices.keys(),
                yticklabels=val_data.class_indices.keys())
    plt.title(f"{model_name} — Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

    return model

if __name__ == "__main__":
    # Evaluate and visualize each model
    history_objects = {}
    for name, path in MODEL_PATHS.items():
        model = evaluate_model(name, path)
        # Optional: store history if available for plotting
        if hasattr(model, "history"):
            history_objects[name] = model.history.history

    # 6️. Visualize validation accuracy over epochs
    plt.figure(figsize=(8,5))
    for name, hist in history_objects.items():
        plt.plot(hist['val_accuracy'], label=f"{name} val_acc")
    plt.title("Validation Accuracy: Custom CNN vs MobileNetV2")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.show()

