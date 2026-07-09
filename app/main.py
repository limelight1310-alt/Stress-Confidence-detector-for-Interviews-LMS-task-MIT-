import streamlit as st
import cv2
import sys
import os
import numpy as np

# Add app folder to path so imports work
sys.path.append(os.path.dirname(__file__))

from detector import detect_face
from classifier import classify_emotion
from scorer import calculate_scores
from feedback import get_feedback
from reporter import generate_report
from grapher import create_emotion_chart, create_score_chart

# Page config
st.set_page_config(
    page_title="Interview Confidence Coach",
    page_icon="🎯",
    layout="centered"
)

# Title and description
st.title("🎯 Interview Confidence Coach")
st.write("Practice your interview and get feedback on how confident you appear.")
st.divider()

# Disclaimer
st.caption("⚠️ This tool estimates facial expression signals only. It does not detect real psychological stress or confidence.")

# Session state setup
# These variables persist across Streamlit reruns
if "recording" not in st.session_state:
    st.session_state.recording = False
if "report" not in st.session_state:
    st.session_state.report = None
if "all_emotions" not in st.session_state:
    st.session_state.all_emotions = []
if "frame_count" not in st.session_state:
    st.session_state.frame_count = 0
if "show_question" not in st.session_state:
    st.session_state.show_question = False

# Status and webcam display
status_box = st.empty()
frame_window = st.image([])

# Buttons
col1, col2 = st.columns(2)
with col1:
    start = st.button("▶️ Start Recording", use_container_width=True)
with col2:
    stop = st.button("⏹️ Stop Recording", use_container_width=True)

st.divider()

# Start button logic
if start:
    st.session_state.recording = True
    st.session_state.report = None
    st.session_state.all_emotions = []
    st.session_state.frame_count = 0
    st.session_state.show_question = False

# Stop button logic
if stop:
    st.session_state.recording = False
    if len(st.session_state.all_emotions) > 0:
        st.session_state.show_question = True

# Recording loop
if st.session_state.recording:
    cap = cv2.VideoCapture(0)

    while st.session_state.recording:
        ret, frame = cap.read()

        if not ret:
            status_box.error("Could not access webcam")
            break

        # Detect face
        annotated_frame, cropped_face, is_centered, face_found = detect_face(frame)

        # Show status message
        if not face_found:
            status_box.warning("⚠️ No face detected — look at the camera")
        elif not is_centered:
            status_box.warning("⚠️ Move your face to the centre of the frame")
        else:
            status_box.success("✅ Recording — face detected and centred")

        # Classify emotion if face found
        if face_found and cropped_face is not None:
            emotions = classify_emotion(cropped_face)
            st.session_state.all_emotions.append(emotions)
            st.session_state.frame_count += 1

        # Display frame
        frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
        frame_window.image(frame_rgb)

    cap.release()

# After recording — ask the question
if st.session_state.show_question and st.session_state.report is None:
    st.subheader("Quick question before your results:")
    st.write(f"We analysed **{st.session_state.frame_count} frames** of your practice session.")
    st.write("During that session — how did you actually feel?")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("😟 I felt stressed", use_container_width=True):
            import pandas as pd
            df = pd.DataFrame(st.session_state.all_emotions)
            averaged = df.mean().to_dict()
            averaged = {k: round(v, 2) for k, v in averaged.items()}
            st.session_state.report = generate_report(averaged, True, st.session_state.frame_count)
            st.session_state.show_question = False
            st.rerun()

    with col2:
        if st.button("😊 I felt confident", use_container_width=True):
            import pandas as pd
            df = pd.DataFrame(st.session_state.all_emotions)
            averaged = df.mean().to_dict()
            averaged = {k: round(v, 2) for k, v in averaged.items()}
            st.session_state.report = generate_report(averaged, False, st.session_state.frame_count)
            st.session_state.show_question = False
            st.rerun()

# Display the final report
if st.session_state.report is not None:
    report = st.session_state.report

    st.subheader("📊 Your Results")

    # Overall state with colour coding
    state = report['overall_state']
    if state == "Stressed":
        st.error(f"Overall Appearance: **{state}**")
    elif state == "Confident":
        st.success(f"Overall Appearance: **{state}**")
    else:
        st.info(f"Overall Appearance: **{state}**")

    # Score metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Stress Score", f"{report['stress_score']}%")
    with col2:
        st.metric("Confidence Score", f"{report['confidence_score']}%")
    with col3:
        st.metric("Dominant Emotion", report['dominant_emotion'].capitalize())

    st.write(f"*Based on {report['frames_analysed']} frames analysed*")

    # Charts
    st.subheader("📈 Emotion Breakdown")
    emotion_chart = create_emotion_chart(report['emotions'])
    st.pyplot(emotion_chart)

    score_chart = create_score_chart(report['stress_score'], report['confidence_score'])
    st.pyplot(score_chart)

    # Tips
    st.subheader("💡 Personalised Tips")
    for tip in report['tips']:
        st.info(f"→ {tip}")

    # Retry button
    st.divider()
    if st.button("🔄 Practice Again", use_container_width=True):
        st.session_state.report = None
        st.session_state.all_emotions = []
        st.session_state.frame_count = 0
        st.session_state.show_question = False
        st.rerun()