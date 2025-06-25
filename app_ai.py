from flask import Flask, request, jsonify, render_template
import numpy as np
import tflite_runtime.interpreter as tflite
from PIL import Image
import os

app = Flask(__name__)

interpreter = tflite.Interpreter(model_path="solution_model.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
input_shape = input_details[0]['shape'][1:3]

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

    # โหลดและเตรียมภาพ
    img = Image.open(filepath).convert('RGB')
    img = img.resize(input_shape)
    img_array = np.array(img, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # รันโมเดล
    interpreter.set_tensor(input_details[0]['index'], img_array)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])

    predicted_score = output_data[0][0]  # ผลลัพธ์ 0.0 - 0.9
    intensity = predicted_score * 255.0

    if predicted_score < 0.05:
        return jsonify({'is_solution': False})

    return jsonify({
        'is_solution': True,
        'intensity': float(intensity)
    })

if __name__ == '__main__':
    app.run(debug=True)
