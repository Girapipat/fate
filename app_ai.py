from flask import Flask, request, render_template, jsonify
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import io

app = Flask(__name__)

# โหลดโมเดล
model = tf.keras.models.load_model("solution_model.h5")

# รายชื่อคลาส
class_names = [
    "not_solution",
    "solution_soap(high)",
    "solution_soap(low)",
    "solution_soap(mid)",
    "solution_syrup_black(high)",
    "solution_syrup_black(low)",
    "solution_syrup_black(mid)",
    "solution_syrup_orange(high)",
    "solution_syrup_orange(low)",
    "solution_syrup_orange(mid)",
    "solution_syrup_red(high)",
    "solution_syrup_red(low)",
    "solution_syrup_red(mid)",
    "solution_vinegar(high)",
    "solution_vinegar(low)",
    "solution_vinegar(mid)",
]

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload():
    try:
        file = request.files["file"]
        if not file:
            return jsonify({"error": "ไม่พบไฟล์"}), 400

        # แปลงไฟล์เป็นภาพ
        image_bytes = file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        image = image.resize((224, 224))
        image = img_to_array(image)
        image = np.expand_dims(image, axis=0)
        image = image / 255.0

        # ทำนาย
        predictions = model.predict(image)[0]
        predicted_index = np.argmax(predictions)
        predicted_class = class_names[predicted_index]

        if predicted_class == "not_solution":
            return jsonify({"result": "ไม่ใช่สารละลาย"})

        # ประมาณค่าความเข้มข้นจาก softmax confidence
        confidence = predictions[predicted_index]
        intensity = confidence * 255

        return jsonify({
            "result": "สารละลาย",
            "intensity": float(intensity)
        })

    except Exception as e:
        return jsonify({"error": f"เกิดข้อผิดพลาด: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(debug=True)
