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


def reset_calibration():
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
    left_inner_brow_y = landmarks[70].y
    left_outer_brow_y = landmarks[105].y
    brow_diff = left_inner_brow_y - left_outer_brow_y
    calibration['brow_readings'].append(brow_diff)

    upper_lip_y = landmarks[13].y
    lower_lip_y = landmarks[14].y
    lip_dist = abs(lower_lip_y - upper_lip_y)
    calibration['lip_readings'].append(lip_dist)

    nose_y = landmarks[1].y
    forehead_y = landmarks[10].y
    chin_y = landmarks[152].y
    face_center = (forehead_y + chin_y) / 2
    head_tilt = nose_y - face_center
    calibration['head_readings'].append(head_tilt)

    if len(calibration['brow_readings']) >= 30:
        calibration['brow_diff_baseline'] = np.mean(calibration['brow_readings'])
        calibration['lip_dist_baseline'] = np.mean(calibration['lip_readings'])
        calibration['head_tilt_baseline'] = np.mean(calibration['head_readings'])
        calibration['done'] = True
        return True

    return False


def detect_face(frame, is_calibrating=False):
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

        x_coords = [int(l.x * width) for l in landmarks]
        y_coords = [int(l.y * height) for l in landmarks]
        x = max(0, min(x_coords) - 10)
        y = max(0, min(y_coords) - 10)
        x2 = min(width, max(x_coords) + 10)
        y2 = min(height, max(y_coords) + 10)

        box_color = (0, 165, 255) if is_calibrating else (0, 255, 0)
        cv2.rectangle(annotated_frame, (x, y), (x2, y2), box_color, 2)

        if is_calibrating:
            cv2.putText(annotated_frame, "Calibrating...",
                       (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                       0.6, (0, 165, 255), 2)

        cropped_face = frame[y:y2, x:x2]

        face_center_x = (x + x2) // 2
        face_center_y = (y + y2) // 2
        is_centered = (
            abs(face_center_x - width // 2) < 150 and
            abs(face_center_y - height // 2) < 150
        )

        if is_calibrating and not calibration['done']:
            calibration_done = calibrate_frame(landmarks, width, height)
            return annotated_frame, cropped_face, is_centered, face_found, landmarks_data, calibration_done

        if calibration['done']:
            landmarks_data['signals_available'] = True

            # --- EYE CONTACT ---
            # Use iris center vs eye corner midpoint
            # Much stricter threshold — only true if looking directly at camera
            left_iris_x = landmarks[468].x
            left_iris_y = landmarks[468].y
            right_iris_x = landmarks[473].x
            right_iris_y = landmarks[473].y

            # Eye corners
            left_eye_left_x = landmarks[33].x
            left_eye_right_x = landmarks[133].x
            right_eye_left_x = landmarks[362].x
            right_eye_right_x = landmarks[263].x

            # Eye top and bottom for vertical gaze
            left_eye_top_y = landmarks[159].y
            left_eye_bottom_y = landmarks[145].y
            right_eye_top_y = landmarks[386].y
            right_eye_bottom_y = landmarks[374].y

            left_eye_center_x = (left_eye_left_x + left_eye_right_x) / 2
            right_eye_center_x = (right_eye_left_x + right_eye_right_x) / 2
            left_eye_center_y = (left_eye_top_y + left_eye_bottom_y) / 2
            right_eye_center_y = (right_eye_top_y + right_eye_bottom_y) / 2

            # Horizontal gaze offset
            left_gaze_x = abs(left_iris_x - left_eye_center_x)
            right_gaze_x = abs(right_iris_x - right_eye_center_x)

            # Vertical gaze offset — catches looking up or down
            left_gaze_y = abs(left_iris_y - left_eye_center_y)
            right_gaze_y = abs(right_iris_y - right_eye_center_y)

            # Both horizontal AND vertical must be within tight threshold
            # 0.010 horizontal, 0.008 vertical — catches side glances and up/down
            # Head tilt check — if head is down, eye contact is automatically false
# regardless of iris position because person is reading off something
            current_head_tilt = landmarks[1].y - ((landmarks[10].y + landmarks[152].y) / 2)
            head_is_down = current_head_tilt > calibration['head_tilt_baseline'] + 0.008

# Gaze direction check
            gaze_straight = (
                left_gaze_x < 0.009 and
                right_gaze_x < 0.009 and
                left_gaze_y < 0.008 and
                right_gaze_y < 0.008
                )

# Eye contact is only true if head is not down AND gaze is straight
            eye_contact = gaze_straight and not head_is_down
            landmarks_data['eye_contact'] = eye_contact

            eye_color = (0, 255, 0) if eye_contact else (0, 0, 255)
            cv2.circle(annotated_frame,
                      (int(landmarks[468].x * width), int(landmarks[468].y * height)),
                      3, eye_color, -1)
            cv2.circle(annotated_frame,
                      (int(landmarks[473].x * width), int(landmarks[473].y * height)),
                      3, eye_color, -1)

            # --- BROW TENSION ---
            # Compare BOTH inner brows and use a tighter threshold
            # Uses right brow too for better accuracy
            left_brow_diff = landmarks[70].y - landmarks[105].y
            right_brow_diff = landmarks[300].y - landmarks[334].y

            # Tense if EITHER brow is significantly lower than baseline
            # Threshold tightened from 0.012 to 0.006
            brow_tension = (
                left_brow_diff > calibration['brow_diff_baseline'] + 0.006 or
                right_brow_diff > calibration['brow_diff_baseline'] + 0.006
            )
            landmarks_data['brow_tension'] = brow_tension

            brow_color = (0, 0, 255) if brow_tension else (0, 255, 0)
            cv2.circle(annotated_frame,
                      (int(landmarks[70].x * width), int(landmarks[70].y * height)),
                      4, brow_color, -1)

            # --- LIP TENSION ---
            current_lip_dist = abs(landmarks[14].y - landmarks[13].y)
            lip_tension = current_lip_dist < calibration['lip_dist_baseline'] * 0.7
            landmarks_data['lip_tension'] = lip_tension

            # --- AUTHENTIC SMILE ---
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

            # --- HEAD LEVEL ---
            # Stricter — catches looking down at laptop/script
            nose_y = landmarks[1].y
            forehead_y = landmarks[10].y
            chin_y = landmarks[152].y
            face_center_y_val = (forehead_y + chin_y) / 2
            current_tilt = nose_y - face_center_y_val
            head_level = current_tilt < calibration['head_tilt_baseline'] + 0.010
            landmarks_data['head_level'] = head_level

            head_color = (0, 255, 0) if head_level else (0, 165, 255)
            cv2.circle(annotated_frame,
                      (int(landmarks[1].x * width), int(landmarks[1].y * height)),
                      5, head_color, -1)

            # --- FACE STABILITY ---
            # Much stricter — catches key fidgeting and restlessness
            current_nose_x = landmarks[1].x
            current_nose_y = landmarks[1].y
            if prev_nose_x is not None:
                movement = abs(current_nose_x - prev_nose_x) + abs(current_nose_y - prev_nose_y)
                landmarks_data['face_stable'] = movement < 0.006
            prev_nose_x = current_nose_x
            prev_nose_y = current_nose_y

    return annotated_frame, cropped_face, is_centered, face_found, landmarks_data, calibration_done