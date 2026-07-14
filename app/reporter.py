from scorer import calculate_scores
from feedback import get_feedback
from recorder import extract_clip


def generate_report(averaged_emotions, user_feels_stressed, frame_count, landmarks_data=None):
    """Generate the main summary report"""
    stress_score, confidence_score, dominant_emotion = calculate_scores(averaged_emotions, landmarks_data)
    tips = get_feedback(stress_score, confidence_score, dominant_emotion, user_feels_stressed)

    if stress_score > confidence_score + 10:
        overall_state = "Stressed"
    elif confidence_score > stress_score + 10:
        overall_state = "Confident"
    else:
        overall_state = "Neutral"

    report = {
        'stress_score': stress_score,
        'confidence_score': confidence_score,
        'dominant_emotion': dominant_emotion,
        'overall_state': overall_state,
        'tips': tips,
        'emotions': averaged_emotions,
        'frames_analysed': frame_count
    }

    return report


def generate_timeline(all_frames_data, window_size=10):
    """
    Groups per-frame data into 10 second windows.
    Returns a list of window summaries.
    """
    if not all_frames_data:
        return []

    max_time = max(f['timestamp'] for f in all_frames_data)
    timeline = []
    window_start = 0

    while window_start < max_time:
        window_end = window_start + window_size
        window_frames = [
            f for f in all_frames_data
            if window_start <= f['timestamp'] < window_end
        ]

        if window_frames:
            avg_stress = sum(f['stress_score'] for f in window_frames) / len(window_frames)
            avg_confidence = sum(f['confidence_score'] for f in window_frames) / len(window_frames)

            emotions = [f['dominant_emotion'] for f in window_frames]
            dominant = max(set(emotions), key=emotions.count)

            eye_contact_rate = sum(
                1 for f in window_frames
                if f.get('landmarks_data', {}).get('eye_contact', False)
            ) / len(window_frames)

            brow_tension_rate = sum(
                1 for f in window_frames
                if f.get('landmarks_data', {}).get('brow_tension', False)
            ) / len(window_frames)

            if avg_stress > 65:
                state = "High Stress"
            elif avg_stress > 45:
                state = "Slight Stress"
            elif avg_confidence > 65:
                state = "Confident"
            else:
                state = "Neutral"

            timeline.append({
                'start': window_start,
                'end': min(window_end, max_time),
                'stress_score': round(avg_stress, 1),
                'confidence_score': round(avg_confidence, 1),
                'dominant_emotion': dominant,
                'state': state,
                'eye_contact_rate': eye_contact_rate,
                'brow_tension_rate': brow_tension_rate
            })

        window_start += window_size

    return timeline


def find_stressful_moments(timeline, all_frames_data, recording_fps=15):
    """
    Finds top 3 most stressful windows.
    Extracts clips for each and builds explanation cards.
    """
    if not timeline:
        return []

    sorted_windows = sorted(timeline, key=lambda x: x['stress_score'], reverse=True)
    top_3 = sorted_windows[:3]
    top_3 = sorted(top_3, key=lambda x: x['start'])

    moments = []
    for i, window in enumerate(top_3):
        reasons = []

        reasons.append(f"Stress score was {window['stress_score']}% during this window")

        if window['eye_contact_rate'] < 0.5:
            reasons.append("Poor eye contact detected")

        if window['brow_tension_rate'] > 0.5:
            reasons.append("Brow tension detected")

        if window['dominant_emotion'] in ['fear', 'angry', 'sad', 'disgust']:
            reasons.append(f"Dominant emotion: {window['dominant_emotion'].capitalize()}")

        if not reasons:
            reasons.append(f"Stress signals detected at {window['stress_score']}%")

        start_fmt = f"{int(window['start'] // 60):02d}:{int(window['start'] % 60):02d}"
        end_fmt = f"{int(window['end'] // 60):02d}:{int(window['end'] % 60):02d}"

        # 5 seconds padding each side
        clip_start = max(0, window['start'] - 2)
        clip_end = window['end'] + 2

        # Pass recording fps so extraction uses correct frame rate
        clip_path = extract_clip(clip_start, clip_end, i, fps=recording_fps)

        moments.append({
            'start': window['start'],
            'end': window['end'],
            'start_fmt': start_fmt,
            'end_fmt': end_fmt,
            'stress_score': window['stress_score'],
            'state': window['state'],
            'reasons': reasons,
            'clip_path': clip_path,
            'dominant_emotion': window['dominant_emotion']
        })

    return moments