import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models

# CONFIG
img_size = (224, 224)
batch_size = 32
data_dir = "dataset"  # ← เปลี่ยนให้ตรงกับโฟลเดอร์ของคุณ

# === LABEL MAP ===
def label_to_value(label):
    if label == "not_solution":
        return -1.0
    level_map = {"low": 0.3, "mid": 0.6, "high": 0.9}
    for level in level_map:
        if level in label:
            return level_map[level]
    return 0.0

# === CUSTOM GENERATOR สำหรับ Regression ===
class RegressionDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, generator):
        self.generator = generator
        self.filenames = generator.filenames
        self.label_map = [label_to_value(name.split('/')[0]) for name in self.filenames]

    def __len__(self):
        return int(np.ceil(len(self.filenames) / self.generator.batch_size))

    def __getitem__(self, index):
        images, _ = self.generator[index]
        start = index * self.generator.batch_size
        end = start + len(images)
        labels = np.array(self.label_map[start:end])
        return images, labels

# === DATA LOAD ===
datagen = ImageDataGenerator(validation_split=0.2, rescale=1./255)

train_gen_raw = datagen.flow_from_directory(
    data_dir,
    target_size=img_size,
    batch_size=batch_size,
    subset='training',
    class_mode='sparse',
    shuffle=True
)

val_gen_raw = datagen.flow_from_directory(
    data_dir,
    target_size=img_size,
    batch_size=batch_size,
    subset='validation',
    class_mode='sparse',
    shuffle=False
)

train_gen = RegressionDataGenerator(train_gen_raw)
val_gen = RegressionDataGenerator(val_gen_raw)

# === MODEL ===
base_model = tf.keras.applications.MobileNetV2(input_shape=img_size + (3,), include_top=False, weights='imagenet')
base_model.trainable = False

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation='relu'),
    layers.Dense(64, activation='relu'),
    layers.Dense(1, activation='linear')
])

model.compile(optimizer='adam', loss=tf.keras.losses.MeanSquaredError(), metrics=[tf.keras.metrics.MeanAbsoluteError()])

# === TRAIN ===
model.fit(train_gen, validation_data=val_gen, epochs=15)

# === SAVE ===
model.save("solution_model.h5")