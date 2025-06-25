import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split

# === CONFIG ===
img_size = (224, 224)
batch_size = 32
epochs = 15
dataset_dir = "dataset"

# === MAP ความเข้มข้นให้เป็นค่า 0.0 - 1.0 ===
def get_label_from_folder(folder_name):
    if folder_name.startswith("not_solution"):
        return 0.0
    elif "(lowest)" in folder_name:
        return 0.1
    elif "(lower)" in folder_name:
        return 0.2
    elif "(low)" in folder_name:
        return 0.3
    elif "(midlow)" in folder_name:
        return 0.4
    elif "(mid)" in folder_name:
        return 0.5
    elif "(midhigh)" in folder_name:
        return 0.6
    elif "(high)" in folder_name:
        return 0.7
    elif "(higher)" in folder_name:
        return 0.8
    elif "(highest)" in folder_name:
        return 0.9
    else:
        return 0.0  # fallback

# === LOAD DATA ===
image_paths = []
labels = []

for folder in os.listdir(dataset_dir):
    folder_path = os.path.join(dataset_dir, folder)
    if not os.path.isdir(folder_path):
        continue

    label = get_label_from_folder(folder)
    for fname in os.listdir(folder_path):
        if fname.lower().endswith((".jpg", ".jpeg", ".png")):
            image_paths.append(os.path.join(folder_path, fname))
            labels.append(label)

print(f"Found {len(image_paths)} images.")

# === LOAD & PROCESS IMAGES ===
def load_image(path):
    image = tf.io.read_file(path)
    image = tf.image.decode_image(image, channels=3)
    image.set_shape((None, None, 3))  # เพิ่มเพื่อให้ shape ชัดเจน
    image = tf.image.resize(image, img_size)
    image = tf.cast(image, tf.float32) / 255.0
    return image.numpy()  # กลับเป็น numpy array เพื่อไม่ให้ X เป็น tensor

X = np.array([load_image(path) for path in image_paths])
y = np.array(labels, dtype=np.float32)

# === SPLIT DATA ===
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# === BUILD MODEL ===
base_model = MobileNetV2(include_top=False, input_shape=img_size + (3,), weights="imagenet")
base_model.trainable = False

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(64, activation="relu"),
    layers.Dense(1, activation="sigmoid")
])

model.compile(optimizer="adam", loss=tf.keras.losses.MeanSquaredError(), metrics=["mae"])

# === TRAIN ===
history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=epochs, batch_size=batch_size)

# === SAVE ===
model.save("solution_model.h5")
print("Model saved as solution_model.h5")
