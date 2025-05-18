import tensorflow as tf
from tensorflow import keras

# โหลดโมเดล .h5
model = keras.models.load_model("solution_model.h5")

# แปลงเป็น TFLite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# บันทึกไฟล์ .tflite
with open("solution_model.tflite", "wb") as f:
    f.write(tflite_model)
