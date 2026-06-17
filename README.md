# Advanced Drowsiness Detection System

This is a comprehensive computer vision-based driver monitoring system. It uses a combination of MediaPipe and YOLOv8 to track facial landmarks, head pose, and object detection (like cell phones) to ensure the driver remains alert and focused on the road.

## Features
- **Drowsiness Detection**: Measures Eye Aspect Ratio (EAR) to detect if the driver is falling asleep.
- **Yawning Detection**: Measures Mouth Aspect Ratio (MAR) to detect if the driver is yawning continuously.
- **Distraction & Head Pose**: Calculates pitch, yaw, and roll to detect if the driver is looking away or tilting their head.
- **Head Nodding Detection**: specialized logic to detect sudden drops in head pitch or sustained nodding angles.
- **Eye Rubbing Detection**: Checks if hands are placed near the eyes indicating fatigue.
- **Phone Usage Detection**: Uses a YOLOv8 object detection model to detect if the driver is holding a cell phone.
- **Multi-Level Alerts**: 
  - Audio warnings with text-to-speech.
  - Email alerts with captured photo frames if an alert persists for more than 8 seconds.

## Requirements
* Python 3.8+
* A webcam
* Dependencies listed in `requirements.txt`:
  * `ultralytics` (for YOLOv8)
  * `mediapipe`
  * `opencv-python`
  * `numpy`
  * `scipy`

## Installation
1. Clone or download this repository.
2. Create and activate a Python virtual environment:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```
3. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. *Note: On the first run, the system will automatically download the necessary MediaPipe `.task` models and the YOLOv8 `.pt` model.*

## Usage
Run the main detector script:
```bash
python detector.py
```
A window will pop up showing your webcam feed with real-time detection metrics.
Press **`q`** to quit the application.

## Core Thresholds & Math
* **EAR (Eyes)**: Triggers when the Eye Aspect Ratio drops below `0.25` for 20 frames.
* **MAR (Mouth)**: Triggers when the Mouth Aspect Ratio goes above `0.50` for 10 frames.
* **Nodding (Pitch)**: Triggers if the head pitch stays between `-20 and -3` OR `5 and 20` for 4 seconds, OR if there's a sudden head drop of `> 20` degrees.

*(For a more detailed breakdown, refer to the generated `Mathematical_Details.docx` document.)*
