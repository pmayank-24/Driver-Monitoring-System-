# Driver Monitoring System

An AI-based project designed to detect driver drowsiness and mobile phone distraction using Computer Vision and Machine Learning.

### Technology Used
Python, OpenCV, MediaPipe Face Mesh, Scikit-learn, Random Forest Classifier, NumPy, Pandas, Streamlit, and YOLO/Object Detection.

### Key Features
- Real-time camera processing.
- Facial landmark detection using MediaPipe Face Mesh.
- EAR and MAR calculation for drowsiness analysis.
- Driver-state classification using Random Forest.
- Mobile phone detection using object detection.
- Planned risk assessment, alerts, and Streamlit dashboard.

### System Workflow

```text
Camera Input
     ↓
OpenCV Video Processing
     ↓
Driver Monitoring
     ↓
 ┌──────────────────┬──────────────────┐
 ↓                  ↓
Drowsiness          Mobile Phone
Detection           Detection
 ↓                  ↓
MediaPipe Face      YOLO / Object
Mesh                Detection
 ↓                  ↓
EAR & MAR           Phone Detection
Calculation         Status
 ↓                  │
Random Forest       │
Classifier          │
 ↓                  │
Drowsiness Status   │
 └────────┬─────────┘
          ↓
   Risk Assessment
          ↓
    Alert System
          ↓
 Streamlit Dashboard
```

*Note: Mobile phone detection, risk assessment, alerts, and complete dashboard integration are still in development.*

### Project Status
**Approximately 50% complete.** Camera processing, facial landmark detection, EAR/MAR calculation, and the initial Machine Learning pipeline have been developed.

### Objective
To develop an affordable camera-based system that helps identify potentially unsafe driving behaviour and supports driver safety.
