print("Script started")

import sys
import os

# This line tells Python where to find scorer.py
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'app'))

from scorer import calculate_scores

# Test 1 — Stressed person
# High fear, angry, sad, disgust
stressed_emotions = {
    'fear': 40,
    'angry': 20,
    'sad': 10,
    'disgust': 8,
    'happy': 10,
    'neutral': 10,
    'surprise': 2
}

stress, confidence, dominant = calculate_scores(stressed_emotions)
print(f"Test 1 — Stressed person:")
print(f"Stress Score: {stress}%")
print(f"Confidence Score: {confidence}%")
print(f"Dominant Emotion: {dominant}")
print()

# Test 2 — Confident person
# High happy and neutral
confident_emotions = {
    'fear': 5,
    'angry': 3,
    'sad': 2,
    'disgust': 2,
    'happy': 50,
    'neutral': 35,
    'surprise': 3
}

stress, confidence, dominant = calculate_scores(confident_emotions)
print(f"Test 2 — Confident person:")
print(f"Stress Score: {stress}%")
print(f"Confidence Score: {confidence}%")
print(f"Dominant Emotion: {dominant}")
print()

# Test 3 — Mixed/neutral person
# Roughly equal stress and confidence signals
mixed_emotions = {
    'fear': 15,
    'angry': 15,
    'sad': 10,
    'disgust': 10,
    'happy': 25,
    'neutral': 23,
    'surprise': 2
}

stress, confidence, dominant = calculate_scores(mixed_emotions)
print(f"Test 3 — Mixed person:")
print(f"Stress Score: {stress}%")
print(f"Confidence Score: {confidence}%")
print(f"Dominant Emotion: {dominant}")


# Test feedback
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'app'))
from feedback import get_feedback

print("--- Feedback Test 1: Looks stressed, feels stressed ---")
tips = get_feedback(81.6, 18.4, 'fear', True)
for tip in tips:
    print(f"- {tip}")
print()

print("--- Feedback Test 2: Looks stressed, feels confident ---")
tips = get_feedback(81.6, 18.4, 'fear', False)
for tip in tips:
    print(f"- {tip}")
print()

print("--- Feedback Test 3: Looks confident ---")
tips = get_feedback(16.7, 83.3, 'happy', False)
for tip in tips:
    print(f"- {tip}")


print("--- Feedback Test 4: Neutral state ---")
tips = get_feedback(50.0, 50.0, 'neutral', False)
for tip in tips:
    print(f"- {tip}")

print()

print("--- Classifier Test ---")
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'app'))
from classifier import classify_emotion
import numpy as np

# Create a dummy face image — just a blank grey square
# 48x48 pixels, 3 colour channels
dummy_face = np.zeros((48, 48, 3), dtype=np.uint8)
dummy_face[:] = (128, 128, 128)  # fill with grey

emotions = classify_emotion(dummy_face)
print("Emotion probabilities:")
for emotion, prob in emotions.items():
    print(f"  {emotion}: {prob}%")


print("--- Reporter Test ---")
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'app'))
from reporter import generate_report

test_emotions = {
    'angry': 15.0, 'disgust': 8.0, 'fear': 25.0,
    'happy': 10.0, 'sad': 12.0, 'surprise': 5.0, 'neutral': 25.0
}

report = generate_report(test_emotions, True, 45)
print(f"Overall State: {report['overall_state']}")
print(f"Stress Score: {report['stress_score']}%")
print(f"Confidence Score: {report['confidence_score']}%")
print(f"Dominant Emotion: {report['dominant_emotion']}")
print(f"Tips: {len(report['tips'])} tips generated")
print(f"Frames analysed: {report['frames_analysed']}")