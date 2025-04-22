from flask import Flask, request, render_template, redirect, url_for, send_file
from datetime import datetime
from ultralytics import YOLO
import os
import cv2
from PIL import Image
import io

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max-limit

# Create uploads directory if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load YOLO model
model = YOLO('best_11n.pt')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return 'No file uploaded', 400
    
    file = request.files['file']
    if file.filename == '':
        return 'No file selected', 400

    # Save the uploaded file
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], 'temp_image.jpg')
    file.save(filepath)
    
    if request.form.get('action') == 'predict':
        # Run YOLO prediction
        results = model(filepath)
        img = results[0].plot()
        
        # Save the result
        cv2.imwrite(filepath, img)
    
    return send_file(filepath, mimetype='image/jpeg')

@app.route('/reset', methods=['POST'])
def reset():
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], 'temp_image.jpg')
    if os.path.exists(filepath):
        os.remove(filepath)
    return 'Reset complete'

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
