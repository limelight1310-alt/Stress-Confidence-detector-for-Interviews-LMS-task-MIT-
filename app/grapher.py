import matplotlib.pyplot as plt
import matplotlib
import numpy as np

# Use non-interactive backend so it works inside Streamlit
matplotlib.use('Agg')

def create_emotion_chart(emotions):
    """
    Creates a bar chart of emotion probabilities
    Returns a matplotlib figure
    """

    # Sort emotions by value for cleaner display
    sorted_emotions = dict(sorted(emotions.items(), key=lambda x: x[1], reverse=True))

    labels = list(sorted_emotions.keys())
    values = list(sorted_emotions.values())

    # Colour code — stress emotions red, confidence emotions green
    stress_emotions = ['angry', 'fear', 'sad', 'disgust']
    colors = ['#e74c3c' if label in stress_emotions else '#2ecc71' for label in labels]

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(labels, values, color=colors, edgecolor='white', linewidth=0.5)

    # Add percentage labels on top of each bar
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            f'{value:.1f}%',
            ha='center', va='bottom', fontsize=9
        )

    ax.set_title('Emotion Breakdown', fontsize=14, fontweight='bold')
    ax.set_ylabel('Percentage (%)')
    ax.set_ylim(0, max(values) + 15)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    return fig


def create_score_chart(stress_score, confidence_score):
    """
    Creates a simple comparison chart of stress vs confidence
    Returns a matplotlib figure
    """

    fig, ax = plt.subplots(figsize=(6, 3))

    categories = ['Stress', 'Confidence']
    values = [stress_score, confidence_score]
    colors = ['#e74c3c', '#2ecc71']

    bars = ax.barh(categories, values, color=colors, height=0.4)

    # Add score labels
    for bar, value in zip(bars, values):
        ax.text(
            value + 1,
            bar.get_y() + bar.get_height() / 2,
            f'{value:.1f}%',
            va='center', fontsize=11, fontweight='bold'
        )

    ax.set_xlim(0, 120)
    ax.set_title('Stress vs Confidence Score', fontsize=14, fontweight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    return fig