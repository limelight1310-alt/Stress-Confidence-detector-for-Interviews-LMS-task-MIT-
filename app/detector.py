import mediapipe as mp
import cv2
import numpy as np

mp_face_mesh = mp.solutions.face_mesh

face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6,
    refine_landmarks=True
)

# Personal calibration baselines — set during calibration phase
calibration = {
    'done': False,
    'brow_diff_baseline': 0.0,
    'lip_dist_baseline': 0.0,
    'head_tilt_baseline': 0.0,
    'brow_readings': [],
    'lip_readings': [],
    'head_readings': []
}

# Previous frame nose position for stability
prev_nose_x = None
prev_nose_y = None


def reset_calibration():
    """Call this when a new session starts"""
    global calibration, prev_nose_x, prev_nose_y
    calibration = {
        'done': False,
        'brow_diff_baseline': 0.0,
        'lip_dist_baseline': 0.0,
        'head_tilt_baseline': 0.0,
        'brow_readings': [],
        'lip_readings': [],
        'head_readings': []
    }
    prev_nose_x = None
    prev_nose_y = None


def calibrate_frame(landmarks, width, height):
    """
    Collect one frame of calibration data.
    Returns True when calibration is complete (after 30 frames)
    """
    # Brow difference — inner vs outer brow y position
    left_inner_brow_y = landmarks[70].y
    left_outer_brow_y = landmarks[105].y
    brow_diff = left_inner_brow_y - left_outer_brow_y
    calibration['brow_readings'].append(brow_diff)

    # Lip distance
    upper_lip_y = landmarks[13].y
    lower_lip_y = landmarks[14].y
    lip_dist = abs(lower_lip_y - upper_lip_y)
    calibration['lip_readings'].append(lip_dist)

    # Head tilt
    nose_y = landmarks[1].y
    forehead_y = landmarks[10].y
    chin_y = landmarks[152].y
    face_center = (forehead_y + chin_y) / 2
    head_tilt = nose_y - face_center
    calibration['head_readings'].append(head_tilt)

    # After 30 frames set personal baselines
    if len(calibration['brow_readings']) >= 30:
        calibration['brow_diff_baseline'] = np.mean(calibration['brow_readings'])
        calibration['lip_dist_baseline'] = np.mean(calibration['lip_readings'])
        calibration['head_tilt_baseline'] = np.mean(calibration['head_readings'])
        calibration['done'] = True
        return True

    return False


def detect_face(frame, is_calibrating=False):
    """
    Takes a single webcam frame and calibration flag.
    Returns:
    - annotated_frame
    - cropped_face
    - is_centered
    - face_found
    - landmarks_data
    - calibration_done
    """
    global prev_nose_x, prev_nose_y

    height, width, _ = frame.shape
    annotated_frame = frame.copy()
    cropped_face = None
    is_centered = False
    face_found = False
    calibration_done = calibration['done']

    landmarks_data = {
        'eye_contact': False,
        'brow_tension': False,
        'lip_tension': False,
        'authentic_smile': False,
        'head_level': True,
        'face_stable': True,
        'signals_available': False
    }

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:
        face_found = True
        landmarks = results.multi_face_landmarks[0].landmark

        # Bounding box
        x_coords = [int(l.x * width) for l in landmarks]
        y_coords = [int(l.y * height) for l in landmarks]
        x = max(0, min(x_coords) - 10)
        y = max(0, min(y_coords) - 10)
        x2 = min(width, max(x_coords) + 10)
        y2 = min(height, max(y_coords) + 10)

        # Orange box during calibration, green during recording
        box_color = (0, 165, 255) if is_calibrating else (0, 255, 0)
        cv2.rectangle(annotated_frame, (x, y), (x2, y2), box_color, 2)

        if is_calibrating:
            cv2.putText(annotated_frame, "Calibrating...",
                       (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                       0.6, (0, 165, 255), 2)

        cropped_face = frame[y:y2, x:x2]

        # Centring check
        face_center_x = (x + x2) // 2
        face_center_y = (y + y2) // 2
        is_centered = (
            abs(face_center_x - width // 2) < 150 and
            abs(face_center_y - height // 2) < 150
        )

        # If calibrating collect baseline data
        if is_calibrating and not calibration['done']:
            calibration_done = calibrate_frame(landmarks, width, height)
            return annotated_frame, cropped_face, is_centered, face_found, landmarks_data, calibration_done

        # Only compute signals after calibration is done
        if calibration['done']:
            landmarks_data['signals_available'] = True

            # Eye contact
            left_iris_x = landmarks[468].x
            right_iris_x = landmarks[473].x
            left_eye_center = (landmarks[33].x + landmarks[133].x) / 2
            right_eye_center = (landmarks[362].x + landmarks[263].x) / 2
            left_gaze_offset = abs(left_iris_x - left_eye_center)
            right_gaze_offset = abs(right_iris_x - right_eye_center)
            eye_contact = (left_gaze_offset < 0.025 and right_gaze_offset < 0.025)
            landmarks_data['eye_contact'] = eye_contact

            eye_color = (0, 255, 0) if eye_contact else (0, 0, 255)
            cv2.circle(annotated_frame,
                      (int(landmarks[468].x * width), int(landmarks[468].y * height)),
                      3, eye_color, -1)
            cv2.circle(annotated_frame,
                      (int(landmarks[473].x * width), int(landmarks[473].y * height)),
                      3, eye_color, -1)

            # Brow tension compared to personal baseline
            current_brow_diff = landmarks[70].y - landmarks[105].y
            brow_tension = current_brow_diff > calibration['brow_diff_baseline'] + 0.012
            landmarks_data['brow_tension'] = brow_tension

            brow_color = (0, 0, 255) if brow_tension else (0, 255, 0)
            cv2.circle(annotated_frame,
                      (int(landmarks[70].x * width), int(landmarks[70].y * height)),
                      4, brow_color, -1)

            # Lip tension compared to personal baseline
            current_lip_dist = abs(landmarks[14].y - landmarks[13].y)
            lip_tension = current_lip_dist < calibration['lip_dist_baseline'] * 0.7
            landmarks_data['lip_tension'] = lip_tension

            # Authentic smile
            mouth_left_y = landmarks[61].y
            mouth_right_y = landmarks[291].y
            cheek_left_y = landmarks[117].y
            cheek_right_y = landmarks[346].y
            mouth_raised = (mouth_left_y < landmarks[17].y and
                          mouth_right_y < landmarks[17].y)
            cheeks_raised = (cheek_left_y < landmarks[234].y + 0.02 and
                           cheek_right_y < landmarks[454].y + 0.02)
            authentic_smile = mouth_raised and cheeks_raised
            landmarks_data['authentic_smile'] = authentic_smile

            # Head level compared to personal baseline
            nose_y = landmarks[1].y
            forehead_y = landmarks[10].y
            chin_y = landmarks[152].y
            face_center_y_val = (forehead_y + chin_y) / 2
            current_tilt = nose_y - face_center_y_val
            head_level = current_tilt < calibration['head_tilt_baseline'] + 0.025
            landmarks_data['head_level'] = head_level

            head_color = (0, 255, 0) if head_level else (0, 165, 255)
            cv2.circle(annotated_frame,
                      (int(landmarks[1].x * width), int(landmarks[1].y * height)),
                      5, head_color, -1)

            # Face stability
            current_nose_x = landmarks[1].x
            current_nose_y = landmarks[1].y
            if prev_nose_x is not None:
                movement = abs(current_nose_x - prev_nose_x) + abs(current_nose_y - prev_nose_y)
                landmarks_data['face_stable'] = movement < 0.015
            prev_nose_x = current_nose_x
            prev_nose_y = current_nose_y

    return annotated_frame, cropped_face, is_centered, face_found, landmarks_data, calibration_done