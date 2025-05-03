import cv2
import numpy as np
from mtcnn import MTCNN
import face_recognition
import os

class FaceDetector:
    def __init__(self):
        self.detector = MTCNN()
        
    def detect_faces(self, image):
        """Detect faces in an image using MTCNN"""
        faces = self.detector.detect_faces(image)
        return faces
    
    def extract_face(self, image, face_box, target_size=(224, 224)):
        """Extract and preprocess a face from the image"""
        x, y, w, h = face_box
        face = image[y:y+h, x:x+w]
        face = cv2.resize(face, target_size)
        return face
    
    def preprocess_image(self, image):
        """Preprocess image for the model"""
        # Convert to RGB if needed
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
        
        # Normalize pixel values
        image = image.astype(np.float32) / 255.0
        
        return image

class FaceDatabase:
    def __init__(self, database_path):
        self.database_path = database_path
        self.known_faces = {}
        self.known_encodings = []
        self.known_names = []
        
    def load_known_faces(self):
        """Load known faces from the database"""
        for person_name in os.listdir(self.database_path):
            person_path = os.path.join(self.database_path, person_name)
            if os.path.isdir(person_path):
                for image_name in os.listdir(person_path):
                    image_path = os.path.join(person_path, image_name)
                    image = cv2.imread(image_path)
                    if image is not None:
                        face_locations = face_recognition.face_locations(image)
                        if face_locations:
                            face_encoding = face_recognition.face_encodings(image, face_locations)[0]
                            self.known_encodings.append(face_encoding)
                            self.known_names.append(person_name)
                            if person_name not in self.known_faces:
                                self.known_faces[person_name] = []
                            self.known_faces[person_name].append(face_encoding)
    
    def recognize_face(self, face_encoding, tolerance=0.6):
        """Recognize a face by comparing with known faces"""
        if not self.known_encodings:
            return "Unknown", 0.0
            
        face_distances = face_recognition.face_distance(self.known_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)
        
        if face_distances[best_match_index] < tolerance:
            return self.known_names[best_match_index], 1 - face_distances[best_match_index]
        else:
            return "Unknown", 0.0
    
    def add_new_face(self, image, person_name):
        """Add a new face to the database"""
        # Check if person already exists
        if person_name in self.known_faces:
            return False, "Person already registered"
            
        face_locations = face_recognition.face_locations(image)
        if face_locations:
            face_encoding = face_recognition.face_encodings(image, face_locations)[0]
            person_path = os.path.join(self.database_path, person_name)
            os.makedirs(person_path, exist_ok=True)
            
            # Save the image
            image_count = len(os.listdir(person_path))
            image_path = os.path.join(person_path, f"{image_count + 1}.jpg")
            cv2.imwrite(image_path, image)
            
            # Update database
            self.known_encodings.append(face_encoding)
            self.known_names.append(person_name)
            if person_name not in self.known_faces:
                self.known_faces[person_name] = []
            self.known_faces[person_name].append(face_encoding)
            
            return True, "Face registered successfully"
        return False, "No face detected in the image" 