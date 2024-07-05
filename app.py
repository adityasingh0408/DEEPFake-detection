from flask import Flask, render_template, request, redirect, url_for, Response
from deepfake_detection import detect_deepfake
import cv2
import time
import tensorflow as tf
from keras.models import load_model
import numpy as np
model1 = load_model(r'C:\projects\deepfake_detection_app\deepfake\25_3.h5')

app = Flask(__name__)

# Global variables
is_detection_running = False
qr_code_count = 0
count = 0

def detect_barcodes(camera):
    global is_detection_running
    model_images = load_model(r'C:\projects\deepfake_detection_app\deepfake\25_3.h5')

    while is_detection_running:
        success, frame = camera.read()
        if not success:
            break
        
        resized_img = cv2.resize(frame, (224, 224))  # Adjust the resize dimensions to (224, 224)
        input_img = resized_img / 255.0  # Normalize input images to [0, 1]
        input_img = np.expand_dims(input_img, axis=0)  # Add batch dimension
            
        predictions1 = model1.predict(input_img)

            
        pred = "FAKE" if predictions1 >= 0.7 else "REAL"
        print(f'The predicted class of the video is {pred}')


        cv2.putText(frame, f"PREDICTION: {pred}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        

        ret, jpeg = cv2.imencode('.jpg', frame)
        frame_bytes = jpeg.tobytes()

        yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

def generate(camera):
    global is_detection_running
    while is_detection_running:
        yield next(detect_barcodes(camera))


# Define routes
@app.route('/')
def index():
    return render_template('upload.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return redirect(request.url)
    
    file = request.files['file']

    if file.filename == '':
        return redirect(request.url)
    
    if file:
        # Perform deepfake detection on the uploaded image
        detection_result, accuracy = detect_deepfake(file)
        return render_template('result.html', detection_result=detection_result, accuracy=accuracy)

@app.route('/result')
def result():
    return render_template('result.html')

@app.route('/start_detection')
def start_detection():
    global is_detection_running
    if not is_detection_running:
        is_detection_running = True
        camera = cv2.VideoCapture(0)
        return Response(detect_barcodes(camera), mimetype='multipart/x-mixed-replace; boundary=frame')
    return 'Barcode detection is already running.'

@app.route('/stop_detection')
def stop_detection():
    global is_detection_running
    is_detection_running = False
    time.sleep(2) 
    return 'Barcode detection stopped.'


if __name__ == '__main__':
    app.run(debug=True)
