import math
import numpy as np
import cv2

# Mediapipe landmark indices
LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]

# For MAR
LIP_TOP = 13
LIP_BOTTOM = 14
LIP_LEFT = 78
LIP_RIGHT = 308

# For Head Pose
FACE_2D_INDICES = [1, 152, 263, 33, 291, 61] # Nose, Chin, Right Eye, Left Eye, Right Mouth, Left Mouth

def distance(p1, p2):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

def calculate_ear(landmarks, eye_indices, frame_width, frame_height):
    """
    Calculates the Eye Aspect Ratio (EAR).
    """
    pts = []
    for idx in eye_indices:
        lm = landmarks[idx]
        pts.append((int(lm.x * frame_width), int(lm.y * frame_height)))
    
    v1 = distance(pts[1], pts[5])
    v2 = distance(pts[2], pts[4])
    h = distance(pts[0], pts[3])
    
    ear = (v1 + v2) / (2.0 * h) if h != 0 else 0
    return ear

def calculate_mar(landmarks, frame_width, frame_height):
    """
    Calculates the Mouth Aspect Ratio (MAR) for yawning detection.
    """
    p_top = (int(landmarks[LIP_TOP].x * frame_width), int(landmarks[LIP_TOP].y * frame_height))
    p_bottom = (int(landmarks[LIP_BOTTOM].x * frame_width), int(landmarks[LIP_BOTTOM].y * frame_height))
    p_left = (int(landmarks[LIP_LEFT].x * frame_width), int(landmarks[LIP_LEFT].y * frame_height))
    p_right = (int(landmarks[LIP_RIGHT].x * frame_width), int(landmarks[LIP_RIGHT].y * frame_height))
    
    v = distance(p_top, p_bottom)
    h = distance(p_left, p_right)
    
    mar = v / h if h != 0 else 0
    return mar

def get_head_pose(landmarks, frame_width, frame_height):
    """
    Calculates head pose (pitch, yaw, roll) using solvePnP.
    """
    face_2d = []
    face_3d = []
    
    for idx in FACE_2D_INDICES:
        lm = landmarks[idx]
        x, y = int(lm.x * frame_width), int(lm.y * frame_height)
        face_2d.append([x, y])
        face_3d.append([x, y, lm.z])
        
    face_2d = np.array(face_2d, dtype=np.float64)
    face_3d = np.array(face_3d, dtype=np.float64)
    
    focal_length = 1 * frame_width
    cam_matrix = np.array([
        [focal_length, 0, frame_height / 2],
        [0, focal_length, frame_width / 2],
        [0, 0, 1]
    ])
    dist_matrix = np.zeros((4, 1), dtype=np.float64)
    
    success, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)
    
    rmat, _ = cv2.Rodrigues(rot_vec)
    angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)
    
    pitch = angles[0] * 360
    yaw = angles[1] * 360
    roll = angles[2] * 360
    
    return pitch, yaw, roll

def check_eye_rubbing(hand_landmarks_list, face_landmarks, frame_width, frame_height, distance_threshold=40):
    """
    Checks if any hand landmark is very close to the eye landmarks (Eye Rubbing).
    """
    if not hand_landmarks_list or not face_landmarks:
        return False
        
    eye_pts = []
    for idx in LEFT_EYE + RIGHT_EYE:
        lm = face_landmarks[idx]
        eye_pts.append((int(lm.x * frame_width), int(lm.y * frame_height)))
        
    for hand_landmarks in hand_landmarks_list:
        # Check a subset of hand landmarks (fingertips and base) for proximity
        # 8: Index tip, 12: Middle tip, 4: Thumb tip
        for h_idx in [4, 8, 12, 16, 20]: 
            hlm = hand_landmarks[h_idx]
            h_pt = (int(hlm.x * frame_width), int(hlm.y * frame_height))
            
            for e_pt in eye_pts:
                if distance(h_pt, e_pt) < distance_threshold:
                    return True
    return False

def check_head_nodding(pitch_history, drop_threshold=-15, window_size=15):
    """
    Checks for sudden drops in pitch (nodding off).
    """
    if len(pitch_history) < window_size:
        return False
    
    recent_pitches = list(pitch_history)[-window_size:]
    max_pitch = max(recent_pitches)
    min_pitch = min(recent_pitches)
    
    # A nod is characterized by a rapid drop in pitch
    # Also ensuring the minimum pitch went below a certain threshold (e.g. looking down heavily)
    if (max_pitch - min_pitch > 20) and min_pitch < drop_threshold:
        return True
        
    return False

def draw_info(frame, info_dict):
    """
    Draws text info on the frame.
    """
    y_offset = 30
    for key, val in info_dict.items():
        text = f"{key}: {val}"
        color = (0, 255, 0)
        if "Warning" in key or "Alert" in key:
            color = (0, 0, 255)
        cv2.putText(frame, text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        y_offset += 30
    return frame
