def calculate_scores(emotions):
    """
    Takes a dictionary of emotion percentages
    Returns stress score, confidence score, and dominant emotion
    
    Stress formula: fear + angry + sad + disgust
    Confidence formula: happy + neutral
    Surprise is excluded as it is ambiguous
    """

    # Pull each emotion value from the dictionary
    # .get() means if the emotion is missing use 0 instead of crashing
    fear = emotions.get('fear', 0)
    angry = emotions.get('angry', 0)
    sad = emotions.get('sad', 0)
    disgust = emotions.get('disgust', 0)
    happy = emotions.get('happy', 0)
    neutral = emotions.get('neutral', 0)

    # Calculate raw scores
    stress_raw = fear + angry + sad + disgust
    confidence_raw = happy + neutral

    # Normalise so both scores add to 100%
    # This makes them fairly comparable
    total = stress_raw + confidence_raw

    # Avoid dividing by zero if all emotions are 0
    if total == 0:
        return 0, 0, 'neutral'

    stress_score = round((stress_raw / total) * 100, 1)
    confidence_score = round((confidence_raw / total) * 100, 1)

    # Find dominant emotion — the single highest scoring one
    dominant = max(emotions, key=emotions.get)

    return stress_score, confidence_score, dominant