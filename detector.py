import cv2
import time
import threading
import os
import urllib.request
import ssl
from collections import deque
from ultralytics import YOLO

# Fix for SSL: CERTIFICATE_VERIFY_FAILED on macOS
ssl._create_default_https_context = ssl._create_unverified_context

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

from utils import calculate_ear, calculate_mar, get_head_pose, check_eye_rubbing, check_head_nodding, LEFT_EYE, RIGHT_EYE, draw_info
from emailer import send_alert_email

# 1. Download MediaPipe Models
FACE_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
FACE_MODEL_PATH = "face_landmarker.task"
HAND_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
HAND_MODEL_PATH = "hand_landmarker.task"

if not os.path.exists(FACE_MODEL_PATH):
    print("Downloading MediaPipe Face Landmarker model...")
    urllib.request.urlretrieve(FACE_MODEL_URL, FACE_MODEL_PATH)
    
if not os.path.exists(HAND_MODEL_PATH):
    print("Downloading MediaPipe Hand Landmarker model...")
    urllib.request.urlretrieve(HAND_MODEL_URL, HAND_MODEL_PATH)

# 2. Initialize YOLOv8 model for person (0) and cell phone (67) detection
try:
    yolo_model = YOLO('yolov8n.pt')
except Exception as e:
    print(f"Error loading YOLO model: {e}")
    yolo_model = None

# 3. Initialize MediaPipe Face Landmarker
face_base_options = mp_python.BaseOptions(model_asset_path=FACE_MODEL_PATH)
face_options = vision.FaceLandmarkerOptions(
    base_options=face_base_options,
    output_face_blendshapes=False,
    output_facial_transformation_matrixes=False,
    num_faces=1
)
face_landmarker = vision.FaceLandmarker.create_from_options(face_options)

# 4. Initialize MediaPipe Hand Landmarker
hand_base_options = mp_python.BaseOptions(model_asset_path=HAND_MODEL_PATH)
hand_options = vision.HandLandmarkerOptions(
    base_options=hand_base_options,
    num_hands=2
)
hand_landmarker = vision.HandLandmarker.create_from_options(hand_options)

# Constants & Thresholds
EAR_THRESHOLD = 0.25
MAR_THRESHOLD = 0.5
CONSECUTIVE_FRAMES_DROWSY = 20
CONSECUTIVE_FRAMES_YAWN = 10
DISTRACTION_PITCH_THRESH = 20
DISTRACTION_YAW_THRESH = 25
DISTRACTION_ROLL_THRESH = 20
CONSECUTIVE_FRAMES_DISTRACTED = 20
CONSECUTIVE_FRAMES_PHONE = 15

# State variables
drowsy_counter = 0
yawn_counter = 0
distracted_start_time = 0
tilt_start_time = 0
phone_counter = 0
alarm_on = False

# Global Alert Tracking for Email
global_alert_start_time = 0
last_email_sent_time = 0

# History for Head Nodding
pitch_history = deque(maxlen=30)

def play_alarm(message=None):
    """Plays an alarm sound in a separate thread. If message is provided, reads it via TTS."""
    global alarm_on
    if not alarm_on:
        alarm_on = True
        os.system("afplay alarm.wav")
        if message:
            os.system(f"say '{message}'")
        time.sleep(1) # Prevent spam
        alarm_on = False

def process_frame(frame):
    global drowsy_counter, yawn_counter, distracted_start_time, tilt_start_time, phone_counter
    global alarm_on, pitch_history, global_alert_start_time, last_email_sent_time
    
    frame_height, frame_width, _ = frame.shape
    
    person_detected = False
    phone_detected = False
    current_alert = None
    
    # Run YOLOv8
    if yolo_model:
        # Detect person (0) and cell phone (67)
        results = yolo_model(frame, classes=[0, 67], verbose=False) 
        for r in results:
            boxes = r.boxes
            for box in boxes:
                cls_id = int(box.cls[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                if cls_id == 0:
                    person_detected = True
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    cv2.putText(frame, "Driver Detected", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                elif cls_id == 67:
                    phone_detected = True
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.putText(frame, "Phone", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    else:
        person_detected = True
        
    info_dict = {
        "EAR": "N/A",
        "MAR": "N/A",
        "Pitch": "N/A",
        "Yaw": "N/A"
    }
    
    # Phone Alert Logic
    if phone_detected:
        phone_counter += 1
        if phone_counter >= CONSECUTIVE_FRAMES_PHONE:
            info_dict["Phone Alert"] = "PUT PHONE AWAY!"
            current_alert = "Phone Usage"
            # As requested, no "wakeup driver" voice alert for phone, just beep.
            if not alarm_on:
                threading.Thread(target=play_alarm).start()
    else:
        phone_counter = 0

    if person_detected:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        
        # Face Detection
        face_result = face_landmarker.detect(mp_image)
        # Hand Detection
        hand_result = hand_landmarker.detect(mp_image)
        
        if face_result.face_landmarks:
            face_landmarks = face_result.face_landmarks[0] # Assume 1 driver
            
            # EAR
            left_ear = calculate_ear(face_landmarks, LEFT_EYE, frame_width, frame_height)
            right_ear = calculate_ear(face_landmarks, RIGHT_EYE, frame_width, frame_height)
            ear = (left_ear + right_ear) / 2.0
            info_dict["EAR"] = f"{ear:.2f}"
            
            # MAR
            mar = calculate_mar(face_landmarks, frame_width, frame_height)
            info_dict["MAR"] = f"{mar:.2f}"
            
            # Head Pose
            pitch, yaw, roll = get_head_pose(face_landmarks, frame_width, frame_height)
            info_dict["Pitch"] = f"{pitch:.0f}"
            info_dict["Yaw"] = f"{yaw:.0f}"
            
            pitch_history.append(pitch)
            
            # 1. Drowsiness Logic
            if ear < EAR_THRESHOLD:
                drowsy_counter += 1
                if drowsy_counter >= CONSECUTIVE_FRAMES_DROWSY:
                    info_dict["Drowsiness Alert"] = "DROWSY!"
                    current_alert = "Drowsiness"
                    if not alarm_on:
                        threading.Thread(target=play_alarm, args=("Wake up driver!",)).start()
            else:
                drowsy_counter = 0
                
            # 2. Yawning Logic
            if mar > MAR_THRESHOLD:
                yawn_counter += 1
                if yawn_counter >= CONSECUTIVE_FRAMES_YAWN:
                    info_dict["Yawning Alert"] = "YAWNING!"
                    current_alert = "Yawning"
            else:
                yawn_counter = 0
                
            # 3. Distraction Logic
            if abs(pitch) > DISTRACTION_PITCH_THRESH or abs(yaw) > DISTRACTION_YAW_THRESH:
                if distracted_start_time == 0:
                    distracted_start_time = time.time()
                elif time.time() - distracted_start_time >= 4.0:
                    info_dict["Distraction Alert"] = "DISTRACTED!"
                    current_alert = "Distraction"
                    if not alarm_on:
                        threading.Thread(target=play_alarm, args=("Warning! You have been looking away for 4 seconds. Keep eyes on the road!",)).start()
            else:
                distracted_start_time = 0
                
            # Head Tilt Logic
            if abs(roll) > DISTRACTION_ROLL_THRESH:
                if tilt_start_time == 0:
                    tilt_start_time = time.time()
                elif time.time() - tilt_start_time >= 4.0:
                    info_dict["Head Tilt Alert"] = "HEAD TILTED!"
                    current_alert = "Head Tilt"
                    if not alarm_on:
                        threading.Thread(target=play_alarm, args=("Warning! Your head has been tilted for 4 seconds. Please keep your head straight!",)).start()
            else:
                tilt_start_time = 0
                
            # 4. Head Nodding Logic
            if check_head_nodding(pitch_history):
                info_dict["Nodding Alert"] = "NODDING OFF!"
                current_alert = "Nodding Off"
                if not alarm_on:
                    threading.Thread(target=play_alarm, args=("Warning! Head nodding detected!",)).start()
                    
            # 5. Eye Rubbing Logic
            if hand_result.hand_landmarks:
                if check_eye_rubbing(hand_result.hand_landmarks, face_landmarks, frame_width, frame_height):
                    info_dict["Eye Rubbing Alert"] = "EYE RUBBING!"
                    current_alert = "Eye Rubbing"
                    if not alarm_on:
                        threading.Thread(target=play_alarm, args=("You seem tired. Please take a break.",)).start()

            # Draw facial landmarks
            for idx in LEFT_EYE:
                lm = face_landmarks[idx]
                cv2.circle(frame, (int(lm.x * frame_width), int(lm.y * frame_height)), 1, (0, 255, 0), -1)
            for idx in RIGHT_EYE:
                lm = face_landmarks[idx]
                cv2.circle(frame, (int(lm.x * frame_width), int(lm.y * frame_height)), 1, (0, 255, 0), -1)
                
    else:
        info_dict["Status"] = "No Driver Detected"

    # --- EMAIL ALERT TRACKING ---
    if current_alert is not None:
        if global_alert_start_time == 0:
            global_alert_start_time = time.time()
            
        elapsed_time = time.time() - global_alert_start_time
        
        # If alert has been active for 8 or more seconds
        if elapsed_time >= 8.0:
            # Check cooldown so we don't spam emails (e.g., once every 60 seconds)
            if time.time() - last_email_sent_time > 60:
                print(f"[!] Alert '{current_alert}' active for 8 seconds. Sending email...")
                # Save the frame as an image
                timestamp_str = time.strftime("%Y%m%d-%H%M%S")
                image_path = f"alert_{timestamp_str}.jpg"
                cv2.imwrite(image_path, frame)
                threading.Thread(target=send_alert_email, args=("abhinavaby07@gmail.com", current_alert, image_path)).start()
                last_email_sent_time = time.time()
    else:
        # Reset tracker if no alerts are active
        global_alert_start_time = 0

    frame = draw_info(frame, info_dict)
    return frame

def main():
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    print("Starting Drowsiness Detection System. Press 'q' to quit.")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break
            
        processed_frame = process_frame(frame)
        
        cv2.imshow('Advanced Drowsiness Detection', processed_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
