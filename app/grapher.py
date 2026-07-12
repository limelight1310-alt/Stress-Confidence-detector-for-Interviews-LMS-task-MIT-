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

    # Remove neutral from display — it represents uncertainty not emotion
    display_emotions = {k: v for k, v in emotions.items() if k != 'neutral'}
    
    sorted_emotions = dict(sorted(display_emotions.items(), key=lambda x: x[1], reverse=True))

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


def create_timeline_chart(timeline):
    """
    Creates a colour coded timeline chart showing stress level over time.
    Returns a matplotlib figure.
    """
    if not timeline:
        return None

    fig, ax = plt.subplots(figsize=(10, 2.5))

    for window in timeline:
        start = window['start']
        width = window['end'] - window['start']
        stress = window['stress_score']

        # Colour by stress level
        if stress > 65:
            color = '#f87171'   # red — high stress
        elif stress > 45:
            color = '#fbbf24'   # amber — slight stress
        else:
            color = '#34d399'   # green — confident

        ax.barh(0, width, left=start, height=0.6, color=color, edgecolor='#0a0e1a', linewidth=1)

        # Add stress % label on each bar if wide enough
        if width >= 8:
            ax.text(
                start + width / 2, 0,
                f"{stress:.0f}%",
                ha='center', va='center',
                fontsize=8, fontweight='bold', color='white'
            )

    # Format x axis as MM:SS
    max_time = timeline[-1]['end']
    ax.set_xlim(0, max_time)
    ax.set_ylim(-0.5, 0.5)

    xticks = range(0, int(max_time) + 1, 10)
    ax.set_xticks(xticks)
    ax.set_xticklabels([
        f"{int(t // 60):02d}:{int(t % 60):02d}" for t in xticks
    ], fontsize=8)

    ax.set_yticks([])
    ax.set_title('Interview Timeline', fontsize=13, fontweight='bold', pad=10)
    ax.set_xlabel('Time', fontsize=9)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)

    # Legend
    from matplotlib.patches import Patch
    legend = [
        Patch(color='#34d399', label='Confident'),
        Patch(color='#fbbf24', label='Slight Stress'),
        Patch(color='#f87171', label='High Stress')
    ]
    ax.legend(handles=legend, loc='upper right', fontsize=8, framealpha=0.3)

    plt.tight_layout()
    return fig