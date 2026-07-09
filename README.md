# Stress-Confidence-detector-for-Interviews-LMS-task-MIT-

A CNN-based tool that analyses facial expressions during interview practice 
and tells you whether you look stressed or confident — then helps you improve.

## The Problem

When people communicate, they unintentionally reveal how they feel through 
their facial expressions. In high-stakes situations like job or college 
interviews, looking stressed — even when you feel confident inside — can cost 
you the opportunity. Interviewers form first impressions within seconds, and 
a stressed expression signals unreliability before you say a word.

## My Solution

A CNN model analyses the user's facial expressions during a recorded practice 
interview and returns 7 emotion percentages adding up to 100% — for example 
60% fear, 15% anger, 10% disgust, 4% sad, 6% surprised, 3% neutral, 2% happy. 
A stress formula (Fear + Anger + Sad + Disgust) and confidence formula 
(Happy + Neutral) then calculate which score is higher. The user is asked 
whether they truly feel stressed — if yes they receive tips like breathe and 
reset, if no they receive tips like relax your brow and make eye contact. 
This helps fresh graduates and college applicants practise looking as 
confident as they feel.

## Dataset

FER2013 is an open-source facial emotion dataset containing 35,887 grayscale 
images classified into 7 emotions — Happy, Sad, Angry, Neutral, Disgust, 
Surprise and Fear — collected from the web and widely validated across 
peer-reviewed research in facial emotion recognition.

## Technology Stack

- Python
- OpenCV
- TensorFlow/Keras
- HuggingFace Transformers (trpakov/vit-face-expression)
- MediaPipe (replaced with OpenCV Haar Cascade due to Python 3.13 compatibility)
- Streamlit
- NumPy
- Pandas
- Matplotlib
- PyTorch

## How to Run

**Step 1 — Clone the repository:**
```bash
git clone https://github.com/limelight1310-alt/Stress-Confidence-detector-for-Interviews-LMS-task-MIT-.git
cd Stress-Confidence-detector-for-Interviews-LMS-task-MIT-
```

**Step 2 — Create and activate virtual environment:**
```bash
python3 -m venv venv
source venv/bin/activate
```

**Step 3 — Install dependencies:**
```bash
pip install -r requirements.txt
```

**Step 4 — Run the app:**
```bash
streamlit run app/main.py
```

**Note:** On first run the emotion detection model will download automatically 
— approximately 300MB. This requires an internet connection and takes 1-2 
minutes. Subsequent runs load instantly from cache.

## How It Works

1. User opens the app and clicks Start Recording
2. Webcam activates and OpenCV Haar Cascade detects the face in real time
3. Each frame is sent to a pretrained Vision Transformer model which returns 
   7 emotion probabilities via softmax
4. Stress score = Fear + Angry + Sad + Disgust
5. Confidence score = Happy + Neutral
6. After recording the user is asked whether they truly feel stressed or confident
7. Based on scores and user answer, specific actionable tips are displayed

## Disclaimer

This application estimates how facial expressions may appear during an 
interview setting. It does not detect, diagnose, or measure real 
psychological stress or confidence. All scores are based solely on 
visible facial expression patterns.

## Author

Arhan Goyal — Grade 9, Pathways School Noida
MIT LMS AI Fellowship 2026
