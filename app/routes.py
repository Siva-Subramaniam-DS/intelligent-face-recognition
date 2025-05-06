from flask import Blueprint, render_template, request, jsonify, Response
import cv2
import numpy as np
from app.utils.face_utils import FaceDetector, FaceDatabase
from app.models.siamese_network import SiameseNetwork
from app.utils.statistics import RecognitionStatistics
import os
from werkzeug.utils import secure_filename
import face_recognition

main = Blueprint('main', __name__)

# Initialize components
face_detector = FaceDetector()
face_db = FaceDatabase('organized_faces_dataset')
siamese_net = SiameseNetwork()
stats_handler = RecognitionStatistics()

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
            success, message = face_db.add_new_face(image, person_name)
            
            return jsonify({'success': success, 'message': message})
    
    return render_template('register.html')

@main.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

def gen_frames():
    try:
        camera = cv2.VideoCapture(0)
        if not camera.isOpened():
            print("Error: Could not open camera")
            return
        
        while True:
            success, frame = camera.read()
            if not success:
                print("Error: Could not read frame from camera")
                break
            
            try:
                # Convert frame to RGB for face detection
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Detect faces
                faces = face_detector.detect_faces(rgb_frame)
                
                if faces:
                    # Process each face
                    for face in faces:
                        x, y, w, h = face['box']
                        face_img = face_detector.extract_face(rgb_frame, face['box'])
                        
                        try:
                            # Get face encoding
                            face_encodings = face_recognition.face_encodings(face_img)
                            if face_encodings:
                                face_encoding = face_encodings[0]
                                
                                # Recognize face
                                name, confidence = face_db.recognize_face(face_encoding)
                                
                                # Log the recognition
                                status = 'success' if name != 'Unknown' else 'failed'
                                stats_handler.add_log(name, confidence, status)
                                
                                # Draw rectangle and label
                                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                                cv2.putText(frame, f"{name} ({confidence:.2f})", (x, y-10),
                                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                            else:
                                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
                                cv2.putText(frame, "No face encoding", (x, y-10),
                                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                        except Exception as e:
                            print(f"Error in face recognition: {str(e)}")
                            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
                            cv2.putText(frame, "Recognition Error", (x, y-10),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                
                ret, buffer = cv2.imencode('.jpg', frame)
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
                
            except Exception as e:
                print(f"Error processing frame: {str(e)}")
                continue
                
    except Exception as e:
        print(f"Camera error: {str(e)}")
    finally:
        if 'camera' in locals():
            camera.release()

@main.route('/statistics')
def statistics():
    page = request.args.get('page', 1, type=int)
    per_page = 20  # Fixed at 20 rows per page as requested
    stats = stats_handler.get_statistics(page=page, per_page=per_page)
    return render_template('statistics.html', **stats)

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