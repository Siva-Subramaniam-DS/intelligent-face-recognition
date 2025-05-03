# Intelligent Face Recognition System

A modern, intelligent face recognition system built with Python, Flask, and OpenCV. This system provides real-time face detection, recognition, and registration capabilities with detailed statistics and logging.

## Features

- **Real-time Face Detection**: Uses MTCNN for accurate face detection
- **Face Recognition**: Implements face recognition using face encodings
- **Face Registration**: Register new faces with names
- **Live Video Feed**: Real-time camera feed with face detection and recognition
- **Statistics Dashboard**: Track recognition attempts, success rates, and detailed logs
- **Modern UI**: Clean, responsive interface with real-time updates
- **Duplicate Prevention**: Prevents duplicate registrations of the same person

## System Architecture

```
intelligent-face-recognition/
├── app/
│   ├── models/
│   │   └── siamese_network.py
│   ├── static/
│   │   ├── css/
│   │   └── js/
│   ├── templates/
│   │   ├── base.html
│   │   ├── index.html
│   │   ├── register.html
│   │   └── statistics.html
│   ├── utils/
│   │   ├── face_utils.py
│   │   └── statistics.py
│   ├── __init__.py
│   └── routes.py
├── data/
│   └── raw/
├── models/
├── recognition_logs.json
└── run.py
```

## Prerequisites

- Python 3.8+
- OpenCV
- Flask
- MTCNN
- face_recognition
- NumPy
- Bootstrap 5

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Siva-Subramaniam-DS/intelligent-face-recognition.git
cd intelligent-face-recognition
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Start the application:
```bash
python run.py
```

2. Access the web interface at `http://localhost:5000`

3. Available pages:
   - **Home**: Live face recognition feed
   - **Register**: Register new faces
   - **Statistics**: View recognition statistics and logs

## Features in Detail

### Face Detection and Recognition
- Uses MTCNN for accurate face detection
- Implements face encoding for recognition
- Real-time processing of video feed
- Confidence scoring for recognition results

### Registration System
- Upload face images with names
- Automatic face detection and extraction
- Duplicate registration prevention
- Database storage of face encodings

### Statistics and Logging
- Real-time recognition statistics
- Detailed logs of all recognition attempts
- Success/failure tracking
- Per-person recognition statistics
- Accuracy calculations

### User Interface
- Modern, responsive design
- Real-time video feed
- Camera on/off toggle
- Dynamic status updates
- Clean statistics dashboard

## API Endpoints

- `GET /`: Home page with live face recognition
- `GET /register`: Face registration page
- `POST /register`: Register new face
- `GET /video_feed`: Live video feed endpoint
- `GET /statistics`: Statistics and logs page
- `POST /upload`: Upload and recognize faces

## Configuration

The system can be configured by modifying the following:

- `app/utils/face_utils.py`: Face detection and recognition parameters
- `app/utils/statistics.py`: Logging and statistics settings
- `app/routes.py`: API endpoints and routes

## Security Considerations

- Face data is stored locally
- No external API calls for face recognition
- Secure file handling for uploads
- Input validation for all user inputs

## Performance

- Optimized face detection using MTCNN
- Efficient face encoding storage
- Real-time processing capabilities
- Responsive UI with minimal latency

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Author 
Siva Subramaniam R

## Acknowledgments

- OpenCV for computer vision capabilities
- Flask for web framework
- MTCNN for face detection
- face_recognition library for face recognition
- Bootstrap for UI components 