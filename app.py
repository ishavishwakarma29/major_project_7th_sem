from flask import Flask, render_template, Response, request, redirect, url_for, flash
from imutils.video import VideoStream
from imutils import face_utils
from scipy.spatial import distance as dist
from werkzeug.utils import secure_filename
import cv2
import dlib
import numpy as np
import time
from PIL import Image
import pickle
import os

app = Flask(__name__, template_folder='./templates', static_folder='./static')

app.config['UPLOAD_FOLDER'] = './static/uploads/' 
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024 

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
model = pickle.load(open("F:\Major_Project\model.sav", "rb"))


detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
video_stream = None
drowsy_count = 0
frame_count = 0
trip_active = False

EYE_AR_THRESH = 0.35
EYE_AR_CONSEC_FRAMES = 30
COUNTER = 0

def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)

def final_ear(shape):
    (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
    leftEye = shape[lStart:lEnd]
    rightEye = shape[rStart:rEnd]
    leftEAR = eye_aspect_ratio(leftEye)
    rightEAR = eye_aspect_ratio(rightEye)
    return (leftEAR + rightEAR) / 2.0

def generate_frames():
    global COUNTER, drowsy_count, frame_count, trip_active
    vs = VideoStream(src=0).start()
    time.sleep(1.0)
    while trip_active:
        frame = vs.read()
        frame = cv2.resize(frame, (640, 480))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        rects = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        for (x, y, w, h) in rects:
            rect = dlib.rectangle(int(x), int(y), int(x + w), int(y + h))
            shape = predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)
            ear = final_ear(shape)

            frame_count += 1
            if ear < EYE_AR_THRESH:
                COUNTER += 1
                if COUNTER >= EYE_AR_CONSEC_FRAMES:
                    drowsy_count += 1
                    cv2.putText(frame, "DROWSY ALERT!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            else:
                COUNTER = 0

            cv2.putText(frame, "Press 'End Trip' to stop", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    vs.stop()

def make_prediction(image_path):
    img_array = preprocess_image(image_path, target_size=(224, 224))
    prediction = model.predict(img_array)
    return prediction

def preprocess_image(image_path, target_size=(224, 224)):
    img = Image.open(image_path)
    img = img.resize(target_size)
    img_array = np.array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array

@app.route('/', methods=['GET', 'POST']) 
def index():
    if request.method == 'POST':
       
       if request.method == 'POST':
        file = request.files['image']
        
        if file.filename == '':
            return "No selected file"

        if file:
           filename = secure_filename(file.filename)
           filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
           file.save(filepath)
           prediction = make_prediction(filepath)
           if prediction[0][0]>prediction[0][1]:
                return render_template('prediction.html', file_path=filename, drowsiness_status="Drowsy")   
           else:
               return render_template('prediction.html', file_path=filename, drowsiness_status="Non Drowsy")   
    return render_template('index.html')

@app.route('/start_trip')
def start_trip():
    global trip_active, drowsy_count, frame_count
    trip_active = True
    drowsy_count = 0
    frame_count = 0
    return render_template('trip.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/end_trip')
def end_trip():
    global trip_active, drowsy_count, frame_count
    trip_active = False
    avg_drowsiness = (drowsy_count / frame_count) * 100 if frame_count > 0 else 0
    return render_template('result.html', avg_drowsiness=avg_drowsiness)

if __name__ == '__main__':
    app.run(debug=True)
