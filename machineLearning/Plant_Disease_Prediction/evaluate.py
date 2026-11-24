"""Evaluate the plant disease model:
Precision, Recall, Accuracy, F1 (per class + macro) and save results to JSON + CSV.
Usage:  python evaluate.py [model_path]
"""
import json
import re
import sys
import time
import os

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score, precision_score,
                             recall_score)

MODEL_PATH = sys.argv[1] if len(sys.argv) > 1 else "trained_plant_disease_model.keras"
DATA_DIR = "New Plant Diseases Dataset(Augmented)"

# --------------------------------------------------
# 1. Validation data
# --------------------------------------------------
valid_set = tf.keras.utils.image_dataset_from_directory(
    f"{DATA_DIR}/valid",
    labels="inferred",
    label_mode="categorical",
    color_mode="rgb",
    batch_size=32,
    image_size=(128, 128),
    shuffle=False,          # IMPORTANT: keep order for y_true/y_pred alignment
)
class_names = valid_set.class_names

# --------------------------------------------------
# 2. Predict
# --------------------------------------------------
print("Loading model:", MODEL_PATH)
model = tf.keras.models.load_model(MODEL_PATH)

t0 = time.time()
y_true, y_pred = [], []
for images, labels in valid_set:
    preds = model.predict(images, verbose=0)
    y_pred.extend(np.argmax(preds, axis=1))
    y_true.extend(np.argmax(labels.numpy(), axis=1))
predict_time = time.time() - t0
print(f"Inference on {len(y_true)} images took {predict_time:.1f}s")

# --------------------------------------------------
# 3. Metrics
# --------------------------------------------------
acc = accuracy_score(y_true, y_pred)
prec = precision_score(y_true, y_pred, average="macro", zero_division=0)
rec = recall_score(y_true, y_pred, average="macro", zero_division=0)
f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

print("\n========== OVERALL ==========")
print(f"Accuracy : {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall   : {rec:.4f}")
print(f"F1 Score : {f1:.4f}")

# Per-class report -> DataFrame
report = classification_report(y_true, y_pred, target_names=class_names,
                               output_dict=True, zero_division=0)
df = pd.DataFrame(report).transpose()
print("\n========== PER CLASS (head) ==========")
print(df.head(40).to_string())

# Confusion matrix -> CSV
cm = confusion_matrix(y_true, y_pred)
pd.DataFrame(cm, index=class_names, columns=class_names).to_csv("confusion_matrix.csv")

# --------------------------------------------------
# 4. Training time (from train_log.txt epochs)
# --------------------------------------------------
train_time = None
if os.path.exists("train_log.txt"):
    times = re.findall(r"(\d+)s.*?loss", "")  # placeholder
    secs = [int(m) for m in re.findall(r"^Epoch \d+/10\n?", "") ]
    # grab "NNNNs" epoch durations from log lines like "1885s 856ms/step"
    durations = re.findall(r"\b(\d{3,5})s\b\s+\d+ms/step", open("train_log.txt", encoding="utf-8", errors="ignore").read())
    if durations:
        train_time = sum(int(d) for d in durations)

# --------------------------------------------------
# 5. Save results
# --------------------------------------------------
results = {
    "model_path": MODEL_PATH,
    "evaluated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    "n_validation_images": len(y_true),
    "accuracy": round(acc, 4),
    "precision_macro": round(prec, 4),
    "recall_macro": round(rec, 4),
    "f1_macro": round(f1, 4),
    "training_time_seconds": train_time,
    "training_time_readable": (
        f"{train_time//3600:.0f}h {(train_time%3600)//60:.0f}m" if train_time else None
    ),
    "inference_time_seconds": round(predict_time, 1),
}
with open("evaluation_results.json", "w") as f:
    json.dump(results, f, indent=2)
df.to_csv("classification_report.csv")
print("\nSaved: evaluation_results.json, classification_report.csv, confusion_matrix.csv")
