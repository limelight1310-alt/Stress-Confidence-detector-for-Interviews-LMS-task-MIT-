# Stress-Confidence-detector-for-Interviews-LMS-task-MIT-

A webcam-based interview practice tool that analyses facial expressions 
during interview practice, calculates a stress and confidence score, 
and delivers personalised coaching tips — including asking whether you 
truly feel stressed, something no existing tool does.

## What is New in This Version

- Live stress and confidence scores updating in real time during recording
- Interview question prompts on screen while you practice
- Auto-calibration to your personal facial baseline at session start
- Session timeline showing stress and confidence across the full recording
- Video clips of your most stressed and most confident moments
- Best moment card showing your peak confidence window
- 6 MediaPipe facial behaviour signals — eye contact, brow tension, 
  lip tension, smile authenticity, head level, and face stability

## The Problem

Most people prepare what to say in an interview — not how they look 
saying it. Facial expressions account for 55% of first impressions, 
yet no accessible tool gives candidates private feedback on how they 
appear on camera. This project closes that gap.

## My Solution

A CNN model analyses facial expressions during a recorded practice 
interview and returns 7 emotion percentages adding up to 100%. A 
stress formula (Fear + Anger + Sad + Disgust) and confidence formula 
(Happy + Neutral) calculate which score is higher. The system then 
asks whether you truly feel stressed — if yes you receive tips like 
breathe and reset, if no you receive tips like relax your brow and 
make eye contact.

## Dataset

AffectNet — 450,000+ real-world facial images across 8 emotion 
categories, accessed via the pretrained trpakov/vit-face-expression 
Vision Transformer model on HuggingFace.

## Technology Stack

- Python
- Streamlit
- OpenCV
- MediaPipe
- HuggingFace Transformers
- PyTorch
- Pandas
- Matplotlib
- NumPy

## How to Run

**Step 1 — Clone the repository:**
git clone https://github.com/limelight1310-alt/Stress-Confidence-detector-for-Interviews-LMS-task-MIT-.git
cd Stress-Confidence-detector-for-Interviews-LMS-task-MIT-

**Step 2 — Create and activate virtual environment with Python 3.10:**
python3.10 -m venv venv
source venv/bin/activate

**Step 3 — Install dependencies:**
pip install streamlit opencv-python mediapipe==0.10.14 --no-deps
pip install protobuf==4.25.3 absl-py numpy
pip install torch torchvision transformers pandas matplotlib

**Step 4 — Run the app:**
streamlit run app/main.py

**Note:** On first run the emotion detection model downloads automatically 
— approximately 300MB. Requires internet and takes 1-2 minutes. 
Subsequent runs load instantly from cache.

## How It Works

1. User reads the 3-step guide and clicks Start Recording
2. App calibrates to the user's natural facial baseline for 3 seconds
3. Webcam activates with live stress and confidence scores on the left
4. Interview question appears on the right — user answers naturally
5. After recording the user is asked whether they truly feel stressed
6. Final report shows scores, timeline, video clips and personalised tips

## Disclaimer

This application estimates how facial expressions may appear during an 
interview setting. It does not detect, diagnose, or measure real 
psychological stress or confidence. All scores are based solely on 
visible facial expression patterns.

## Author

Arhan Goyal — Grade 9, Pathways School Noida
MIT LMS AI Fellowship 2026
GitHub: github.com/limelight1310-alt
Portfolio: arhan-portals.lovable.app
