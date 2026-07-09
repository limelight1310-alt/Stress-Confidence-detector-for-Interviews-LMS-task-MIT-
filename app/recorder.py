import cv2
import pandas as pd
import sys
import os

sys.path.append(os.path.dirname(__file__))

from detector import detect_face
from classifier import classify_emotion
from scorer import calculate_scores

def record_session(duration_seconds=30):
    """
    Opens webcam, captures frames, classifies emotion on each frame
    Returns averaged emotion scores across the full session
    """

    cap = cv2.VideoCapture(0)
    
    # Store results from each frame in a list
    all_emotions = []
    frame_count = 0
    
    print(f"Recording for {duration_seconds} seconds...")
    print("Press Q to stop early")
    
    import time
    start_time = time.time()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Could not read frame")
            break
        
        # Check if time is up
        elapsed = time.time() - start_time
        if elapsed > duration_seconds:
            break
        
        # Detect face in frame
        annotated_frame, cropped_face, is_centered, face_found = detect_face(frame)
        
        # Only classify if face was found
        if face_found and cropped_face is not None:
            emotions = classify_emotion(cropped_face)
            all_emotions.append(emotions)
            frame_count += 1
        
        # Show live feed in a window
        cv2.imshow('Recording - Press Q to stop', annotated_frame)
        
        # Press Q to stop early
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    
    # If no frames were captured return neutral defaults
    if len(all_emotions) == 0:
        print("No faces detected during recording")
        return {
            'angry': 0, 'disgust': 0, 'fear': 0,
            'happy': 0, 'sad': 0, 'surprise': 0, 'neutral': 100
        }, 0
    
    # Convert list of emotion dictionaries to a DataFrame
    # Each row is one frame, each column is one emotion
    df = pd.DataFrame(all_emotions)
    
    # Average each emotion across all frames
    averaged = df.mean().to_dict()
    averaged = {k: round(v, 2) for k, v in averaged.items()}
    
    print(f"Recorded {frame_count} frames with faces detected")
    
    return averaged, frame_count