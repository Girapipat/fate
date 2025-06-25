import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
import os

# ========== CONFIG ==========
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
DATA_DIR = r"C:\Users\seapr\Pictures\งาน\ai-solution-classifier\dataset"
MODEL_PATH = "solution_model.h5"
TFLITE_PATH = "solution_model.tflite"
EPOCHS = 10
# ============================

# Load data
datagen = ImageDataGenerator(
    validation_split=0.2,
    rescale=1.0/255
)

train_generator = datagen.flow_from_directory(
    DATA_DIR,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="training"
)

val_generator = datagen.flow_from_directory(
    DATA_DIR,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="validation"
)

# Build model (ใช้ MobileNetV2)
base_model = tf.keras.applications.MobileNetV2(input_shape=IMAGE_SIZE + (3,), include_top=False, weights='imagenet')
base_model.trainable = False

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation='relu'),
    layers.Dense(train_generator.num_classes, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train
model.fit(train_generator, validation_data=val_generator, epochs=EPOCHS)

# Save .h5
model.save(MODEL_PATH)

# Convert to .tflite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

with open(TFLITE_PATH, 'wb') as f:
    f.write(tflite_model)

print(f"\n✅ Training done. Saved as:\n- {MODEL_PATH}\n- {TFLITE_PATH}")
