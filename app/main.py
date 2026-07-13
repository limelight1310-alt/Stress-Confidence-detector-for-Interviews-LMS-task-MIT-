import streamlit as st
from detector import detect_face, reset_calibration
import cv2
import sys
import os
import numpy as np
import pandas as pd
import time

sys.path.append(os.path.dirname(__file__))

from classifier import classify_emotion
from scorer import calculate_scores
from feedback import get_feedback
from reporter import generate_report, generate_timeline, find_stressful_moments
from grapher import create_emotion_chart, create_score_chart, create_timeline_chart
from recorder import save_frame, compile_video, cleanup_session_video, cleanup_clips, cleanup_frames, cleanup_timestamps

QUESTIONS = [
    "Tell me about yourself.",
    "Why should we hire you?",
    "Walk me through your resume.",
    "What are you most proud of?",
    "Why do you want this role?",
    "What is your biggest weakness?",
    "Where do you see yourself in 5 years?",
    "Tell me about a time you failed.",
    "How do you handle pressure?",
    "Why did you leave your last job?",
    "What makes you different from other candidates?",
    "Describe a conflict you had and how you resolved it.",
    "What would your biggest critic say about you?",
]

QUESTION_TIPS = {
    0: "Be concise. Aim for 90 seconds. Past → Present → Future.",
    1: "Connect your skills directly to the role.",
    2: "Keep it relevant. Focus on achievements.",
    3: "Pick something specific and impactful.",
    4: "Research the company. Show genuine interest.",
    5: "Turn it into a growth story. Show self-awareness.",
    6: "Be specific. Link it to your goals and the role.",
    7: "Use STAR method: Situation, Task, Action, Result.",
    8: "Show a system you use. Give a real example.",
    9: "Keep it professional. Focus on growth.",
    10: "Be specific. Avoid generic answers.",
    11: "Stay calm. Show empathy and resolution.",
    12: "Be honest but frame it as something you are working on.",
}

st.set_page_config(page_title="Interview Confidence Coach", page_icon="🎯", layout="wide")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Space+Grotesk:wght@400;500;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    .stApp { background: #0a0e1a; color: #e8eaf6; }
    .hero-title { font-family: 'Space Grotesk', sans-serif; font-size: 2rem; font-weight: 700; color: #ffffff; line-height: 1.15; margin-bottom: 0.2rem; }
    .hero-accent { color: #4f8ef7; }
    .hero-sub { font-size: 0.95rem; color: #7986a3; font-weight: 400; margin-bottom: 1rem; }
    .disclaimer { background: #0d1526; border-left: 3px solid #4f8ef7; padding: 0.5rem 1rem; border-radius: 0 6px 6px 0; font-size: 0.75rem; color: #7986a3; margin-bottom: 1rem; }
    .guide-card { background: #111827; border: 1px solid #1e2d45; border-radius: 12px; padding: 1.5rem; text-align: center; min-height: 180px; }
    .guide-icon { font-size: 2rem; margin-bottom: 0.6rem; }
    .guide-step-title { font-family: 'Space Grotesk', sans-serif; font-size: 0.95rem; font-weight: 700; color: #4f8ef7; margin-bottom: 0.4rem; }
    .guide-step-text { font-size: 0.75rem; color: #7986a3; line-height: 1.6; }
    .live-panel { background: #111827; border: 1px solid #1e2d45; border-radius: 12px; padding: 1.2rem; }
    .live-panel-title { font-size: 0.65rem; font-weight: 700; letter-spacing: 0.12em; text-transform: uppercase; color: #4f8ef7; margin-bottom: 1rem; }
    .live-score-label { font-size: 0.65rem; color: #7986a3; text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 0.2rem; }
    .live-score-stress { font-family: 'Space Grotesk', sans-serif; font-size: 2.2rem; font-weight: 700; color: #f87171; line-height: 1; }
    .live-score-confidence { font-family: 'Space Grotesk', sans-serif; font-size: 2.2rem; font-weight: 700; color: #34d399; line-height: 1; }
    .live-score-block { margin-bottom: 1rem; }
    .live-bar-bg { background: #1e2d45; border-radius: 4px; height: 6px; margin-top: 0.3rem; overflow: hidden; }
    .live-signal { display: flex; align-items: center; justify-content: space-between; padding: 0.3rem 0; border-bottom: 1px solid #1e2d45; font-size: 0.72rem; color: #c5cde8; }
    .live-signal:last-child { border-bottom: none; }
    .sig-good { color: #34d399; font-weight: 600; }
    .sig-bad { color: #f87171; font-weight: 600; }
    .emotion-badge { display: inline-block; background: #0d1526; border: 1px solid #1e2d45; border-radius: 6px; padding: 0.3rem 0.7rem; font-size: 0.8rem; color: #4f8ef7; font-weight: 500; margin-top: 0.5rem; }
    .question-panel { background: #111827; border: 1px solid #1e2d45; border-radius: 12px; padding: 1.2rem; }
    .question-label { font-size: 0.65rem; font-weight: 700; letter-spacing: 0.12em; text-transform: uppercase; color: #4f8ef7; margin-bottom: 0.8rem; }
    .question-number { font-size: 0.7rem; color: #7986a3; margin-bottom: 0.5rem; }
    .question-text { font-family: 'Space Grotesk', sans-serif; font-size: 1.15rem; font-weight: 600; color: #ffffff; line-height: 1.5; }
    .question-tip { font-size: 0.72rem; color: #7986a3; margin-top: 1rem; padding-top: 0.8rem; border-top: 1px solid #1e2d45; line-height: 1.5; }
    .score-card { background: #111827; border: 1px solid #1e2d45; border-radius: 12px; padding: 1.2rem 1.5rem; text-align: center; }
    .score-label { font-size: 0.75rem; font-weight: 600; letter-spacing: 0.1em; text-transform: uppercase; color: #7986a3; margin-bottom: 0.4rem; }
    .score-value-stress { font-family: 'Space Grotesk', sans-serif; font-size: 2.4rem; font-weight: 700; color: #f87171; }
    .score-value-confidence { font-family: 'Space Grotesk', sans-serif; font-size: 2.4rem; font-weight: 700; color: #34d399; }
    .score-value-neutral { font-family: 'Space Grotesk', sans-serif; font-size: 2.4rem; font-weight: 700; color: #4f8ef7; }
    .state-banner-stressed { background: linear-gradient(135deg, #2d0f0f, #3d1515); border: 1px solid #f87171; border-radius: 10px; padding: 1rem 1.5rem; font-family: 'Space Grotesk', sans-serif; font-size: 1.1rem; font-weight: 600; color: #f87171; text-align: center; margin-bottom: 1.5rem; }
    .state-banner-confident { background: linear-gradient(135deg, #0d2d1a, #0f3d22); border: 1px solid #34d399; border-radius: 10px; padding: 1rem 1.5rem; font-family: 'Space Grotesk', sans-serif; font-size: 1.1rem; font-weight: 600; color: #34d399; text-align: center; margin-bottom: 1.5rem; }
    .state-banner-neutral { background: linear-gradient(135deg, #0d1a2d, #0f223d); border: 1px solid #4f8ef7; border-radius: 10px; padding: 1rem 1.5rem; font-family: 'Space Grotesk', sans-serif; font-size: 1.1rem; font-weight: 600; color: #4f8ef7; text-align: center; margin-bottom: 1.5rem; }
    .tip-card { background: #111827; border: 1px solid #1e2d45; border-left: 3px solid #4f8ef7; border-radius: 0 8px 8px 0; padding: 0.9rem 1.2rem; margin-bottom: 0.7rem; font-size: 0.95rem; color: #c5cde8; line-height: 1.5; }
    .signal-card { background: #111827; border: 1px solid #1e2d45; border-radius: 8px; padding: 0.7rem 1rem; text-align: center; font-size: 0.8rem; color: #7986a3; }
    .signal-good { color: #34d399; font-weight: 600; }
    .signal-bad { color: #f87171; font-weight: 600; }
    .section-label { font-size: 0.7rem; font-weight: 700; letter-spacing: 0.15em; text-transform: uppercase; color: #4f8ef7; margin-bottom: 0.8rem; margin-top: 1.8rem; }
    .frames-note { font-size: 0.78rem; color: #7986a3; text-align: center; margin-top: 0.5rem; }
    .moment-card { background: #111827; border: 1px solid #f87171; border-radius: 10px; padding: 1rem 1.2rem; margin-bottom: 1rem; }
    .best-moment-card { background: #111827; border: 1px solid #34d399; border-radius: 10px; padding: 1rem 1.2rem; margin-bottom: 1rem; }
    .moment-header { font-family: 'Space Grotesk', sans-serif; font-size: 0.95rem; font-weight: 600; color: #f87171; margin-bottom: 0.5rem; }
    .best-moment-header { font-family: 'Space Grotesk', sans-serif; font-size: 0.95rem; font-weight: 600; color: #34d399; margin-bottom: 0.5rem; }
    .moment-reason { font-size: 0.85rem; color: #7986a3; margin-bottom: 0.2rem; }
    div[data-testid="stButton"] > button { background: #4f8ef7; color: white; border: none; border-radius: 8px; font-family: 'Inter', sans-serif; font-weight: 600; font-size: 0.9rem; padding: 0.6rem 1.2rem; width: 100%; }
    .recording-status { font-size: 0.85rem; font-weight: 500; color: #f87171; margin-bottom: 0.5rem; }
    .live-indicator { display: inline-block; width: 8px; height: 8px; background: #f87171; border-radius: 50%; margin-right: 6px; animation: pulse 1.2s infinite; }
    @keyframes pulse { 0% { opacity: 1; } 50% { opacity: 0.3; } 100% { opacity: 1; } }
    .calibrating-status { font-size: 0.85rem; font-weight: 500; color: #fbbf24; margin-bottom: 0.5rem; }
</style>
""", unsafe_allow_html=True)

for key, default in {
    "recording": False,
    "report": None,
    "all_emotions": [],
    "all_frames_data": [],
    "frame_count": 0,
    "show_question": False,
    "avg_landmarks": {},
    "calibrating": False,
    "calibration_done": False,
    "timeline": [],
    "stressful_moments": [],
    "best_moment": None,
    "session_start_time": None,
    "recording_fps": 15,
    "recording_started": False,
    "question_index": 0,
    "live_stress": 0.0,
    "live_confidence": 0.0,
    "live_emotion": "—",
    "live_landmarks": {}
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# HERO
st.markdown("""
<div class="hero-title">Interview <span class="hero-accent">Confidence</span> Coach</div>
<div class="hero-sub">Record your practice. See how you look. Improve before it counts.</div>
<div class="disclaimer">⚠️ This tool estimates visible facial expression signals only — it does not measure real psychological stress or confidence.</div>
""", unsafe_allow_html=True)

# HOW IT WORKS — always visible above buttons
st.markdown('<div class="section-label">How it works</div>', unsafe_allow_html=True)
col1, col2, col3 = st.columns(3)
with col1:
    st.markdown('<div class="guide-card"><div class="guide-icon">📷</div><div class="guide-step-title">Step 1 — Calibrate</div><div class="guide-step-text">Hold a natural expression for 3 seconds while we read your personal face baseline</div></div>', unsafe_allow_html=True)
with col2:
    st.markdown('<div class="guide-card"><div class="guide-icon">🎯</div><div class="guide-step-title">Step 2 — Practice</div><div class="guide-step-text">Answer the interview question on screen. Speak naturally for at least 60 seconds</div></div>', unsafe_allow_html=True)
with col3:
    st.markdown('<div class="guide-card"><div class="guide-icon">📊</div><div class="guide-step-title">Step 3 — Review</div><div class="guide-step-text">See your stress and confidence scores, timeline, clips and personalised tips</div></div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# BUTTONS
col_start, col_stop = st.columns(2)
with col_start:
    start = st.button("▶ Start Recording")
with col_stop:
    stop = st.button("⏹ Stop Recording")

if start:
    cleanup_session_video()
    cleanup_frames()
    cleanup_clips()
    cleanup_timestamps()
    st.session_state.recording = True
    st.session_state.calibrating = True
    st.session_state.calibration_done = False
    st.session_state.report = None
    st.session_state.all_emotions = []
    st.session_state.all_frames_data = []
    st.session_state.frame_count = 0
    st.session_state.show_question = False
    st.session_state.avg_landmarks = {}
    st.session_state.timeline = []
    st.session_state.stressful_moments = []
    st.session_state.best_moment = None
    st.session_state.session_start_time = None
    st.session_state.recording_fps = 15
    st.session_state.recording_started = False
    st.session_state.question_index = 0
    st.session_state.live_stress = 0.0
    st.session_state.live_confidence = 0.0
    st.session_state.live_emotion = "—"
    st.session_state.live_landmarks = {}
    reset_calibration()

if stop:
    st.session_state.recording = False
    if st.session_state.recording_started:
        compile_video(st.session_state.recording_fps)
    if len(st.session_state.all_emotions) > 0:
        st.session_state.show_question = True

if st.session_state.recording:
    left_col, mid_col, right_col = st.columns([1, 2, 1])

    with left_col:
        live_panel = st.empty()
    with mid_col:
        status_box = st.empty()
        frame_window = st.image([])
    with right_col:
        question_panel = st.empty()
        if st.button("Next Question →", key="next_q"):
            st.session_state.question_index = (st.session_state.question_index + 1) % len(QUESTIONS)

    cap = cv2.VideoCapture(0)

    while st.session_state.recording:
        ret, frame = cap.read()
        if not ret:
            status_box.markdown('<div class="recording-status">⚠️ Could not access webcam</div>', unsafe_allow_html=True)
            break

        is_calibrating = st.session_state.calibrating
        annotated_frame, cropped_face, is_centered, face_found, landmarks_data, calibration_done = detect_face(frame, is_calibrating)

        if is_calibrating:
            status_box.markdown('<div class="calibrating-status">🔧 Calibrating — hold a natural expression...</div>', unsafe_allow_html=True)
            if calibration_done:
                st.session_state.calibrating = False
                st.session_state.calibration_done = True
                st.session_state.session_start_time = time.time()
        else:
            if not st.session_state.recording_started:
                actual_fps = cap.get(cv2.CAP_PROP_FPS)
                actual_fps = actual_fps if actual_fps > 0 and actual_fps <= 60 else 15
                st.session_state.recording_fps = actual_fps
                st.session_state.recording_started = True

            elapsed = time.time() - st.session_state.session_start_time if st.session_state.session_start_time else 0
            save_frame(frame, st.session_state.frame_count, elapsed)

            if not face_found:
                status_box.markdown('<div class="recording-status">⚠️ No face detected — look at the camera</div>', unsafe_allow_html=True)
            elif not is_centered:
                status_box.markdown('<div class="recording-status">⚠️ Move your face to the centre of the frame</div>', unsafe_allow_html=True)
            else:
                mins = int(elapsed // 60)
                secs = int(elapsed % 60)
                status_box.markdown(f'<div class="recording-status"><span class="live-indicator"></span>Recording {mins:02d}:{secs:02d}</div>', unsafe_allow_html=True)

            if face_found and cropped_face is not None:
                emotions = classify_emotion(cropped_face)
                emotions['_landmarks'] = landmarks_data
                frame_stress, frame_confidence, frame_dominant = calculate_scores(
                    {k: v for k, v in emotions.items() if k != '_landmarks'}, landmarks_data)
                st.session_state.live_stress = frame_stress
                st.session_state.live_confidence = frame_confidence
                st.session_state.live_emotion = frame_dominant
                st.session_state.live_landmarks = landmarks_data
                st.session_state.all_frames_data.append({
                    'timestamp': elapsed,
                    'stress_score': frame_stress,
                    'confidence_score': frame_confidence,
                    'dominant_emotion': frame_dominant,
                    'landmarks_data': landmarks_data
                })
                st.session_state.all_emotions.append(emotions)

            st.session_state.frame_count += 1

        lm = st.session_state.live_landmarks
        eye = "✓ Good" if lm.get('eye_contact') else "✗ Poor"
        eye_c = "sig-good" if lm.get('eye_contact') else "sig-bad"
        brow = "✓ Relaxed" if not lm.get('brow_tension') else "✗ Tense"
        brow_c = "sig-good" if not lm.get('brow_tension') else "sig-bad"
        head = "✓ Upright" if lm.get('head_level', True) else "✗ Tilted"
        head_c = "sig-good" if lm.get('head_level', True) else "sig-bad"
        stable = "✓ Stable" if lm.get('face_stable', True) else "✗ Fidgeting"
        stable_c = "sig-good" if lm.get('face_stable', True) else "sig-bad"
        stress_bar = int(st.session_state.live_stress)
        conf_bar = int(st.session_state.live_confidence)

        live_panel.markdown(f"""
        <div class="live-panel">
            <div class="live-panel-title">Live Feedback</div>
            <div class="live-score-block">
                <div class="live-score-label">Stress Score</div>
                <div class="live-score-stress">{st.session_state.live_stress:.0f}%</div>
                <div class="live-bar-bg"><div style="width:{stress_bar}%;height:6px;background:#f87171;border-radius:4px"></div></div>
            </div>
            <div class="live-score-block">
                <div class="live-score-label">Confidence Score</div>
                <div class="live-score-confidence">{st.session_state.live_confidence:.0f}%</div>
                <div class="live-bar-bg"><div style="width:{conf_bar}%;height:6px;background:#34d399;border-radius:4px"></div></div>
            </div>
            <div class="emotion-badge">{st.session_state.live_emotion.capitalize()}</div>
            <div style="margin-top:1rem">
                <div class="live-signal">Eye Contact <span class="{eye_c}">{eye}</span></div>
                <div class="live-signal">Brow <span class="{brow_c}">{brow}</span></div>
                <div class="live-signal">Head <span class="{head_c}">{head}</span></div>
                <div class="live-signal">Composure <span class="{stable_c}">{stable}</span></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        q_index = st.session_state.question_index
        tip = QUESTION_TIPS.get(q_index, "Take a breath. Answer clearly and confidently.")
        question_panel.markdown(f"""
        <div class="question-panel">
            <div class="question-label">Interview Question</div>
            <div class="question-number">{q_index + 1} of {len(QUESTIONS)}</div>
            <div class="question-text">"{QUESTIONS[q_index]}"</div>
            <div class="question-tip">💡 {tip}</div>
        </div>
        """, unsafe_allow_html=True)

        frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
        frame_window.image(frame_rgb)

    cap.release()


def process_results(user_feels_stressed):
    df = pd.DataFrame(st.session_state.all_emotions)
    emotion_df = df.drop(columns=['_landmarks'], errors='ignore')
    averaged = {k: round(v, 2) for k, v in emotion_df.mean().to_dict().items()}
    avg_landmarks = {
        'signals_available': True,
        'eye_contact': df['_landmarks'].apply(lambda x: x.get('eye_contact', False) if isinstance(x, dict) else False).mean() > 0.7,
        'brow_tension': df['_landmarks'].apply(lambda x: x.get('brow_tension', False) if isinstance(x, dict) else False).mean() > 0.4,
        'lip_tension': df['_landmarks'].apply(lambda x: x.get('lip_tension', False) if isinstance(x, dict) else False).mean() > 0.4,
        'authentic_smile': df['_landmarks'].apply(lambda x: x.get('authentic_smile', False) if isinstance(x, dict) else False).mean() > 0.5,
        'head_level': df['_landmarks'].apply(lambda x: x.get('head_level', True) if isinstance(x, dict) else True).mean() > 0.5,
        'face_stable': df['_landmarks'].apply(lambda x: x.get('face_stable', True) if isinstance(x, dict) else True).mean() > 0.5,
    }
    st.session_state.avg_landmarks = avg_landmarks
    st.session_state.report = generate_report(averaged, user_feels_stressed, len(st.session_state.all_emotions), avg_landmarks)
    timeline = generate_timeline(st.session_state.all_frames_data, window_size=5)
    st.session_state.timeline = timeline
    if timeline:
        moments = find_stressful_moments(timeline, st.session_state.all_frames_data, st.session_state.recording_fps)
        st.session_state.stressful_moments = moments
        best_window = min(timeline, key=lambda x: x['stress_score'])
        clip_start = max(0, best_window['start'] - 5)
        clip_end = best_window['end'] + 5
        from recorder import extract_clip
        best_clip = extract_clip(clip_start, clip_end, 99, fps=st.session_state.recording_fps)
        start_fmt = f"{int(best_window['start'] // 60):02d}:{int(best_window['start'] % 60):02d}"
        end_fmt = f"{int(best_window['end'] // 60):02d}:{int(best_window['end'] % 60):02d}"
        st.session_state.best_moment = {
            'start_fmt': start_fmt, 'end_fmt': end_fmt,
            'confidence_score': best_window['confidence_score'], 'clip_path': best_clip
        }
    st.session_state.show_question = False


if st.session_state.show_question and st.session_state.report is None:
    st.markdown('<div class="section-label">Before your results</div>', unsafe_allow_html=True)
    st.markdown(f"**{len(st.session_state.all_emotions)} frames analysed.** During that session — how did you actually feel inside?")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("😟  I felt stressed"):
            process_results(True)
            st.rerun()
    with col2:
        if st.button("😊  I felt confident"):
            process_results(False)
            st.rerun()

if st.session_state.report is not None:
    report = st.session_state.report
    state = report['overall_state']
    if state == "Stressed":
        st.markdown(f'<div class="state-banner-stressed">⚠ You appear Stressed — {report["stress_score"]}% stress signals detected</div>', unsafe_allow_html=True)
    elif state == "Confident":
        st.markdown(f'<div class="state-banner-confident">✓ You appear Confident — {report["confidence_score"]}% confidence signals detected</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="state-banner-neutral">◎ You appear Neutral — scores are close</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-label">Your Scores</div>', unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f'<div class="score-card"><div class="score-label">Stress Score</div><div class="score-value-stress">{report["stress_score"]}%</div></div>', unsafe_allow_html=True)
    with col2:
        st.markdown(f'<div class="score-card"><div class="score-label">Confidence Score</div><div class="score-value-confidence">{report["confidence_score"]}%</div></div>', unsafe_allow_html=True)
    with col3:
        st.markdown(f'<div class="score-card"><div class="score-label">Dominant Emotion</div><div class="score-value-neutral">{report["dominant_emotion"].capitalize()}</div></div>', unsafe_allow_html=True)

    st.markdown(f'<div class="frames-note">Based on {report["frames_analysed"]} frames analysed</div>', unsafe_allow_html=True)

    if st.session_state.avg_landmarks.get('signals_available'):
        st.markdown('<div class="section-label">Facial Behaviour Signals</div>', unsafe_allow_html=True)
        lm = st.session_state.avg_landmarks
        col1, col2, col3 = st.columns(3)
        with col1:
            ec = "✓ Good" if lm.get('eye_contact') else "✗ Poor"
            ec_class = "signal-good" if lm.get('eye_contact') else "signal-bad"
            st.markdown(f'<div class="signal-card">Eye Contact<br><span class="{ec_class}">{ec}</span></div>', unsafe_allow_html=True)
        with col2:
            bt = "✗ Tense" if lm.get('brow_tension') else "✓ Relaxed"
            bt_class = "signal-bad" if lm.get('brow_tension') else "signal-good"
            st.markdown(f'<div class="signal-card">Brow Position<br><span class="{bt_class}">{bt}</span></div>', unsafe_allow_html=True)
        with col3:
            hl = "✓ Upright" if lm.get('head_level') else "✗ Tilted Down"
            hl_class = "signal-good" if lm.get('head_level') else "signal-bad"
            st.markdown(f'<div class="signal-card">Head Position<br><span class="{hl_class}">{hl}</span></div>', unsafe_allow_html=True)
        col4, col5, col6 = st.columns(3)
        with col4:
            fs = "✓ Stable" if lm.get('face_stable') else "✗ Fidgeting"
            fs_class = "signal-good" if lm.get('face_stable') else "signal-bad"
            st.markdown(f'<div class="signal-card">Composure<br><span class="{fs_class}">{fs}</span></div>', unsafe_allow_html=True)
        with col5:
            lt = "✗ Tense" if lm.get('lip_tension') else "✓ Relaxed"
            lt_class = "signal-bad" if lm.get('lip_tension') else "signal-good"
            st.markdown(f'<div class="signal-card">Lip Tension<br><span class="{lt_class}">{lt}</span></div>', unsafe_allow_html=True)
        with col6:
            sm = "✓ Genuine" if lm.get('authentic_smile') else "— Not detected"
            sm_class = "signal-good" if lm.get('authentic_smile') else "signal-bad"
            st.markdown(f'<div class="signal-card">Smile Quality<br><span class="{sm_class}">{sm}</span></div>', unsafe_allow_html=True)

    if st.session_state.timeline:
        st.markdown('<div class="section-label">Interview Timeline</div>', unsafe_allow_html=True)
        timeline_chart = create_timeline_chart(st.session_state.timeline)
        if timeline_chart:
            st.pyplot(timeline_chart)

    if st.session_state.best_moment:
        st.markdown('<div class="section-label">Your Best Moment</div>', unsafe_allow_html=True)
        bm = st.session_state.best_moment
        st.markdown(f'<div class="best-moment-card"><div class="best-moment-header">✓ {bm["start_fmt"]} – {bm["end_fmt"]} &nbsp;·&nbsp; Confidence {bm["confidence_score"]:.1f}%</div><div class="moment-reason">This was your most confident window — review it to understand what you did right.</div></div>', unsafe_allow_html=True)
        if bm['clip_path'] and os.path.exists(bm['clip_path']):
            st.video(bm['clip_path'])

    if st.session_state.stressful_moments:
        st.markdown('<div class="section-label">Most Stressful Moments</div>', unsafe_allow_html=True)
        for moment in st.session_state.stressful_moments:
            st.markdown(f'<div class="moment-card"><div class="moment-header">⚠ {moment["start_fmt"]} – {moment["end_fmt"]} &nbsp;·&nbsp; {moment["state"]} ({moment["stress_score"]}%)</div>{"".join(f"<div class=moment-reason>→ {r}</div>" for r in moment["reasons"])}</div>', unsafe_allow_html=True)
            if moment['clip_path'] and os.path.exists(moment['clip_path']):
                st.video(moment['clip_path'])
            else:
                st.caption("Clip not available for this moment")

    st.markdown('<div class="section-label">Emotion Breakdown</div>', unsafe_allow_html=True)
    st.pyplot(create_emotion_chart(report['emotions']))
    st.pyplot(create_score_chart(report['stress_score'], report['confidence_score']))

    st.markdown('<div class="section-label">Your Personalised Tips</div>', unsafe_allow_html=True)
    for tip in report['tips']:
        st.markdown(f'<div class="tip-card">→ {tip}</div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("🔄  Practice Again"):
        cleanup_clips()
        cleanup_session_video()
        cleanup_frames()
        cleanup_timestamps()
        for key in ["report", "all_emotions", "all_frames_data", "show_question", "avg_landmarks",
                    "timeline", "stressful_moments", "best_moment", "session_start_time",
                    "recording_started", "question_index", "live_stress", "live_confidence",
                    "live_emotion", "live_landmarks"]:
            st.session_state[key] = {"report": None, "all_emotions": [], "all_frames_data": [],
                "show_question": False, "avg_landmarks": {}, "timeline": [], "stressful_moments": [],
                "best_moment": None, "session_start_time": None, "recording_started": False,
                "question_index": 0, "live_stress": 0.0, "live_confidence": 0.0,
                "live_emotion": "—", "live_landmarks": {}}.get(key)
        st.session_state.frame_count = 0
        st.rerun()