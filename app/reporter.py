from scorer import calculate_scores
from feedback import get_feedback

def generate_report(averaged_emotions, user_feels_stressed, frame_count):
    """
    Takes averaged emotions from the recording session
    Returns a complete report dictionary with all results
    """

    # Calculate stress and confidence scores
    stress_score, confidence_score, dominant_emotion = calculate_scores(averaged_emotions)

    # Get personalised feedback tips
    tips = get_feedback(stress_score, confidence_score, dominant_emotion, user_feels_stressed)

    # Determine overall state label
    if stress_score > confidence_score + 10:
        overall_state = "Stressed"
    elif confidence_score > stress_score + 10:
        overall_state = "Confident"
    else:
        overall_state = "Neutral"

    # Build the full report
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