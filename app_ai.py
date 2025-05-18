from flask import Flask, request, jsonify, render_template
from PIL import Image
import numpy as np
import tensorflow as tf
import os

app = Flask(__name__)

# โหลดโมเดล TFLite
interpreter = tf.lite.Interpreter(model_path="solution_model.tflite")
interpreter.allocate_tensors()

# เตรียม input/output ของโมเดล
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return jsonify({'error': 'ไม่พบไฟล์ในคำขอ'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'ไม่ได้เลือกไฟล์'}), 400

    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)

    # ประมวลผลภาพ
    try:
        img = Image.open(filepath).convert('RGB')
        img = img.resize((224, 224))  # ต้องตรงกับ input ของโมเดล
        img_array = np.array(img, dtype=np.float32) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # รันโมเดล
        interpreter.set_tensor(input_details[0]['index'], img_array)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])

        # สมมุติ output เป็นค่าความเข้มข้น (หรือ [0] = confidence ว่าเป็นสารละลาย)
        intensity = float(output_data[0][0])

        # ถ้าโมเดลบอกว่าไม่ใช่สารละลาย ให้แสดงข้อความพิเศษ (เช่นใช้ threshold สมมุติ)
        if intensity < 0:
            return jsonify({'error': 'ไม่ใช่สารละลาย'})

        return jsonify({'intensity': intensity})

    except Exception as e:
        return jsonify({'error': f'เกิดข้อผิดพลาด: {str(e)}'})

if __name__ == '__main__':
    app.run(debug=True)
