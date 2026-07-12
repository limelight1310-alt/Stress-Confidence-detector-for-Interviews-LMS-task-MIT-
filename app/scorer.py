def calculate_scores(emotions, landmarks_data=None):
    """
    Confidence-weighted fusion of emotion model + MediaPipe signals.
    
    When neutral is high the emotion model is uncertain.
    In that case we shift weight toward MediaPipe behavioural signals
    which are more reliable than uncertain emotion predictions.
    """

    fear = emotions.get('fear', 0)
    angry = emotions.get('angry', 0)
    sad = emotions.get('sad', 0)
    disgust = emotions.get('disgust', 0)
    happy = emotions.get('happy', 0)
    neutral = emotions.get('neutral', 0)

    # Raw emotion stress and confidence signals
    # Neutral is excluded from both — it now means uncertainty only
    emotion_stress = fear + angry + sad + disgust
    emotion_confidence = happy

    # Normalise emotion signals so they are comparable
    emotion_total = emotion_stress + emotion_confidence
    if emotion_total == 0:
        emotion_stress_pct = 50
        emotion_confidence_pct = 50
    else:
        emotion_stress_pct = (emotion_stress / emotion_total) * 100
        emotion_confidence_pct = (emotion_confidence / emotion_total) * 100

    # How uncertain is the emotion model
    # High neutral = high uncertainty = trust MediaPipe more
    emotion_weight = 1 - (neutral / 100)
    mediapipe_weight = neutral / 100

    # Build MediaPipe stress and confidence signals
    mediapipe_stress = 50    # default neutral if no landmarks
    mediapipe_confidence = 50

    if landmarks_data and landmarks_data.get('signals_available', False):
        stress_signals = 0
        confidence_signals = 0
        total_signals = 0

        # Eye contact — highest weight (most validated signal)
        total_signals += 3
        if landmarks_data.get('eye_contact', True):
            confidence_signals += 3  # reward good eye contact
        else:
            stress_signals += 3      # penalise broken eye contact

        # Brow tension
        total_signals += 2
        if landmarks_data.get('brow_tension', False):
            stress_signals += 2
        else:
            confidence_signals += 2

        # Head level
        total_signals += 2
        if landmarks_data.get('head_level', True):
            confidence_signals += 2
        else:
            stress_signals += 2

        # Face stability
        total_signals += 1
        if landmarks_data.get('face_stable', True):
            confidence_signals += 1
        else:
            stress_signals += 1

        # Lip tension
        total_signals += 1
        if landmarks_data.get('lip_tension', False):
            stress_signals += 1
        else:
            confidence_signals += 1

        # Authentic smile — bonus only
        if landmarks_data.get('authentic_smile', False):
            confidence_signals += 1
            total_signals += 1

        # Convert to percentages
        mediapipe_stress = (stress_signals / total_signals) * 100
        mediapipe_confidence = (confidence_signals / total_signals) * 100

    # Weighted fusion
    # When neutral is high mediapipe signals dominate
    # When neutral is low emotion model dominates
    final_stress = (emotion_stress_pct * emotion_weight) + (mediapipe_stress * mediapipe_weight)
    final_confidence = (emotion_confidence_pct * emotion_weight) + (mediapipe_confidence * mediapipe_weight)

    # Normalise to 100
    total = final_stress + final_confidence
    if total == 0:
        stress_score = 50
        confidence_score = 50
    else:
        stress_score = round((final_stress / total) * 100, 1)
        confidence_score = round((final_confidence / total) * 100, 1)

    # Dominant emotion is still from the raw emotion model
    # Exclude neutral from dominant since it means uncertainty
    emotions_without_neutral = {k: v for k, v in emotions.items() if k != 'neutral'}
    if emotions_without_neutral:
        dominant = max(emotions_without_neutral, key=emotions_without_neutral.get)
    else:
        dominant = 'neutral'

    return stress_score, confidence_score, dominant