from flask import Blueprint, render_template, request, jsonify, Response
import cv2
import numpy as np
from app.utils.face_utils import FaceDetector, FaceDatabase
from app.models.siamese_network import SiameseNetwork
import os
from werkzeug.utils import secure_filename
import face_recognition

main = Blueprint('main', __name__)

# Initialize components
face_detector = FaceDetector()
face_db = FaceDatabase('organized_faces_dataset')
siamese_net = SiameseNetwork()

# Load the trained model if it exists
if os.path.exists('models/siamese_model.h5'):
    siamese_net.load_model('models/siamese_model.h5')

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'png', 'jpg', 'jpeg'}

@main.route('/')
def index():
    return render_template('index.html')

@main.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'})
        
        file = request.files['file']
        person_name = request.form.get('name')
        
        if file.filename == '':
            return jsonify({'error': 'No selected file'})
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join('data/raw', filename)
            file.save(file_path)
            
            # Process the image
            image = cv2.imread(file_path)
            success = face_db.add_new_face(image, person_name)
            
            if success:
                return jsonify({'message': 'Face registered successfully'})
            else:
                return jsonify({'error': 'No face detected in the image'})
    
    return render_template('register.html')

@main.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

def gen_frames():
    camera = cv2.VideoCapture(0)
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            # Detect faces
            faces = face_detector.detect_faces(frame)
            
            # Process each face
            for face in faces:
                x, y, w, h = face['box']
                face_img = face_detector.extract_face(frame, face['box'])
                
                # Get face encoding
                face_encoding = face_recognition.face_encodings(face_img)[0]
                
                # Recognize face
                name, confidence = face_db.recognize_face(face_encoding)
                
                # Draw rectangle and label
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(frame, f"{name} ({confidence:.2f})", (x, y-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@main.route('/upload', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join('data/raw', filename)
        file.save(file_path)
        
        # Process the image
        image = cv2.imread(file_path)
        faces = face_detector.detect_faces(image)
        
        results = []
        for face in faces:
            x, y, w, h = face['box']
            face_img = face_detector.extract_face(image, face['box'])
            
            # Get face encoding
            face_encoding = face_recognition.face_encodings(face_img)[0]
            
            # Recognize face
            name, confidence = face_db.recognize_face(face_encoding)
            
            results.append({
                'name': name,
                'confidence': float(confidence),
                'box': [x, y, w, h]
            })
        
        return jsonify({'results': results})
    
    return jsonify({'error': 'Invalid file type'}) 