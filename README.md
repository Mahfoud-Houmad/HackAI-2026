# RaiiVision — AI-Powered Livestock Surveillance & Protection

## 1. Project Title

**RaiiVision** — AI-Powered Livestock Surveillance & Protection System

---

## 2. Project Description

RaiiVision is an Edge AI surveillance system designed for rural livestock protection. The project combines Computer Vision, local AI processing, and GSM-based communication to help farmers monitor animals, detect intrusions, and identify suspicious or abnormal situations in real time.

The system is designed to operate locally on a Raspberry Pi without requiring internet connectivity, making it suitable for isolated rural environments.

The current implementation focuses mainly on:

- Animal detection and counting
- Human intrusion detection
- Face recognition optimization
- Local real-time video analysis
- Alert preparation through GSM/Twilio communication

---

## 3. Business Context

Rural farmers frequently face major challenges such as:

- Livestock theft
- Intrusions into livestock enclosures
- Difficulty monitoring animal activity continuously
- Limited access to internet infrastructure
- Limited access to advanced smart farming technologies

Most existing smart farming systems rely heavily on cloud services and internet connectivity.

RaiiVision addresses this problem by providing:

- Offline AI monitoring
- Real-time livestock surveillance
- Lightweight Edge AI processing
- Simple communication methods (SMS / local calls)
- Affordable deployment using Raspberry Pi hardware

The project aims to make AI-based rural protection accessible, scalable, and easy to use.

---

## 4. Features

### Current Features Identified from the Codebase

#### Animal Detection
- Detects animals using YOLOv8
- Supports cow and sheep recognition
- Displays detection bounding boxes and confidence scores

#### Real-Time Animal Counting
- Counts animals crossing a virtual line
- Uses object tracking with ByteTrack
- Maintains separate counters for cows and sheep

#### Human Intrusion Detection
- Detects human presence using YOLOv8
- Optimizes processing by triggering FaceNet only when a person is detected

#### Face Recognition
- Captures face datasets from a webcam
- Generates and stores facial embeddings using FaceNet
- Compares live faces against known embeddings

#### Alerting Logic
- Prepares SMS and voice call alerts using Twilio APIs
- Includes cooldown and intrusion duration logic

#### Edge AI Processing
- Runs locally using OpenCV, YOLOv8, and FaceNet
- Designed for Raspberry Pi deployment

---

## 5. Project Architecture / Structure

```text
projet/
│
├── capture_faces.py          # Captures face images from webcam
├── save_embedding.py         # Generates FaceNet embeddings
├── detect_animals.py         # Real-time animal detection
├── count_animals.py          # Animal counting + tracking
├── reco_optimizer.py         # Human detection + face recognition + alert logic
├── yolov8n.pt                # YOLOv8 nano model weights
└── embadings/
    ├──person1.npy            # Stored facial embedding
```

---

## 6. Technologies Used

### Programming Language
- Python

### Computer Vision & AI
- OpenCV
- YOLOv8 (Ultralytics)
- FaceNet
- facenet-pytorch
- MTCNN
- ByteTrack

### Machine Learning / Deep Learning
- PyTorch
- NumPy

### Communication
- Twilio API
- GSM-based alert concept

### Hardware Target
- Raspberry Pi
- USB Camera / Pi Camera
- GSM Module (SIM800L / SIM900)

---

## 7. Installation Guide

### Prerequisites

- Python 3.9+
- Webcam or Pi Camera
- GPU optional (CPU supported)

### Clone the Repository

```bash
git clone <repository-url>
cd projet
```

### Create a Virtual Environment

```bash
python -m venv venv
```

### Activate the Environment

#### Windows

```bash
venv\Scripts\activate
```

#### Linux / macOS

```bash
source venv/bin/activate
```

### Install Dependencies

```bash
pip install opencv-python ultralytics facenet-pytorch torch torchvision pillow python-dotenv twilio numpy
```

---

## 8. Usage Instructions

### 1. Capture Face Dataset

```bash
python capture_faces.py
```

Controls:
- Press `S` to save a face image
- Press `Q` to quit

### 2. Generate Face Embeddings

```bash
python save_embedding.py
```

### 3. Run Animal Detection

```bash
python detect_animals.py
```

### 4. Run Animal Counting

```bash
python count_animals.py
```

### 5. Run Intrusion Detection & Recognition

```bash
python reco_optimizer.py
```

---

## 9. Configuration

The system uses environment variables for Twilio communication.

Create a `.env` file:

```env
TWILIO_ACCOUNT_SID=your_sid
TWILIO_AUTH_TOKEN=your_token
TWILIO_PHONE_NUMBER=your_twilio_number
ALERT_PHONE_NUMBER=target_phone_number
TWILIO_VOICE_URL=http://demo.twilio.com/docs/voice.xml
```

---

## 10. API / Main Modules Overview

### `capture_faces.py`
Captures images from a webcam and stores them locally for face recognition training.

### `save_embedding.py`
Processes captured faces using FaceNet and generates a mean embedding vector.

### `detect_animals.py`
Performs real-time detection of cows and sheep using YOLOv8.

### `count_animals.py`
Uses YOLO tracking and ByteTrack to count animals crossing a virtual line.

### `reco_optimizer.py`
Main intelligent security module:

- Detects humans
- Runs face recognition
- Filters unknown individuals
- Handles Twilio alerts
- Optimizes processing using staged detection

---

## 11. Example Workflow

```text
Camera Feed
      ↓
YOLOv8 Detection
      ↓
Animal / Human Identification
      ↓
Tracking & Counting
      ↓
Face Recognition (if human detected)
      ↓
Threat Evaluation
      ↓
SMS / Voice Alert
```

---

## 12. Contributors / Authors

- Mondir-Ayoub
- Mahfoud-Houmad
- Hafsa-IDYOUSS
- EL MOUAFIK Fatima-Ezzahra
