import cv2
import os
import numpy as np

# Load the XML file we downloaded
haar_path = os.path.join(os.path.dirname(__file__), 'haarcascade_frontalface_default.xml')

def detect_face(frame):
    """
    Takes a single webcam frame
    Returns:
    - annotated_frame: frame with green box drawn around face
    - cropped_face: just the face region cropped out
    - is_centered: True if face is in the middle of the frame
    - face_found: True if a face was detected at all
    """

    height, width, _ = frame.shape
    annotated_frame = frame.copy()
    cropped_face = None
    is_centered = False
    face_found = False

    # Convert to greyscale
    grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Use legacy cascade detector from objdetect module
    # This is the correct way to call it in OpenCV 5
    face_cascade = cv2.CascadeClassifier()
    face_cascade.load(haar_path)

    if face_cascade.empty():
        st.error("Could not load face detector file")
        return annotated_frame, cropped_face, is_centered, face_found

    faces = face_cascade.detectMultiScale(
        grey,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )

    if len(faces) > 0:
        face_found = True
        x, y, w, h = faces[0]

        # Draw green box
        cv2.rectangle(annotated_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Crop face
        cropped_face = frame[y:y+h, x:x+w]

        # Check centring
        face_center_x = x + w // 2
        face_center_y = y + h // 2
        frame_center_x = width // 2
        frame_center_y = height // 2

        tolerance = 150
        is_centered = (
            abs(face_center_x - frame_center_x) < tolerance and
            abs(face_center_y - frame_center_y) < tolerance
        )

    return annotated_frame, cropped_face, is_centered, face_found