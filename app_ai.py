from flask import Flask, request, jsonify, render_template
import numpy as np
import tensorflow as tf
from PIL import Image
import os

app = Flask(__name__)

# โหลดโมเดล .h5
model = tf.keras.models.load_model("solution_model.h5", compile=False)

# ขนาดภาพที่โมเดลต้องการ
IMG_SIZE = (224, 224)

# รายชื่อคลาส (ไม่จำเป็นต้องใช้ถ้าเป็น regression model)
# CLASS_NAMES = [...]

UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return jsonify({'error': 'ไม่มีไฟล์ที่ส่งมา'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'ยังไม่ได้เลือกไฟล์'}), 400

    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)

    # โหลดและแปลงภาพ
    img = Image.open(filepath).convert('RGB')
    img = img.resize(IMG_SIZE)
    img_array = np.array(img, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # ทำนายด้วยโมเดล .h5
    predicted_score = model.predict(img_array)[0][0]  # ค่าระหว่าง 0.0 - 0.9
    intensity = predicted_score * 255.0

    # ตัดสินใจว่าเป็นสารละลายหรือไม่
    is_solution = predicted_score > 0.05  # สมมุติเกณฑ์ เช่น มากกว่า 0.05 ถือว่าเป็นสารละลาย

    if not is_solution:
        return jsonify({'is_solution': False})

    return jsonify({
        'is_solution': True,
        'intensity': float(intensity)
    })

if __name__ == '__main__':
    app.run(debug=True)
