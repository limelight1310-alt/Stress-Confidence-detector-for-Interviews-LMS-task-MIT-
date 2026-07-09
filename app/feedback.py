def get_feedback(stress_score, confidence_score, dominant_emotion, user_feels_stressed):
    """
    Takes the scores, dominant emotion, and user's self-assessment
    Returns a list of specific actionable tips
    
    user_feels_stressed is True if user said Yes, False if they said No
    """

    tips = []

    # Case 1 — User looks stressed AND feels stressed
    # They need calming and reset tips
    if stress_score > confidence_score and user_feels_stressed:
        tips.append("Take 3 slow deep breaths before continuing — this visibly relaxes your face.")
        tips.append("Roll your shoulders back and sit up straight — tension shows in your posture.")
        tips.append("Pause for a moment before answering — it signals confidence not uncertainty.")

    # Case 2 — User looks stressed but feels confident
    # They need physical appearance tips only
    elif stress_score > confidence_score and not user_feels_stressed:
        tips.append("You feel confident — but try relaxing your brow, it may appear tense on camera.")
        tips.append("Make direct eye contact with the camera lens, not the screen.")
        tips.append("Let your mouth relax slightly — a natural expression reads as calm and assured.")

    # Case 3 — User looks confident
    # Positive reinforcement plus small improvements
    elif confidence_score > stress_score:
        tips.append("You are projecting confidence well — keep this energy throughout.")
        tips.append("Maintain your natural eye contact with the camera.")
        if dominant_emotion == 'happy':
            tips.append("Your expression is warm and approachable — excellent for interviews.")
        elif dominant_emotion == 'neutral':
            tips.append("Consider adding a slight natural smile to appear more engaged.")

    # Case 4 — Scores are roughly equal — neutral state
    else:
        tips.append("Your expression is neutral — try to show slightly more engagement.")
        tips.append("A small natural smile makes you appear more approachable and confident.")

    # Add emotion specific tips on top of the above
    if dominant_emotion == 'fear' and stress_score > 50:
        tips.append("High anxiety detected — remind yourself you are prepared and focus on breathing.")
    elif dominant_emotion == 'angry':
        tips.append("Relax your eyebrows and jaw — facial tension can read as aggression on camera.")
    elif dominant_emotion == 'sad':
        tips.append("Try to lift your gaze and engage with the camera — downward eyes signal disengagement.")
    elif dominant_emotion == 'disgust':
        tips.append("Your expression may appear uncomfortable — take a breath and reset your face to neutral.")

    return tips