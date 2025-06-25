import tensorflow as tf

model = tf.keras.models.load_model("solution_model.h5")
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

with open("solution_model.tflite", "wb") as f:
    f.write(tflite_model)

print("✅ แปลงโมเดลเรียบร้อยแล้ว!")
