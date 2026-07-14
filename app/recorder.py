import cv2
import os

SESSION_DIR = os.path.join(os.path.dirname(__file__), '..', 'sessions')
FRAMES_DIR = os.path.join(SESSION_DIR, 'frames')
VIDEO_PATH = os.path.join(SESSION_DIR, 'last_session.mp4')
CLIPS_DIR = os.path.join(SESSION_DIR, 'clips')
TIMESTAMPS_PATH = os.path.join(SESSION_DIR, 'timestamps.txt')


def ensure_dirs():
    os.makedirs(FRAMES_DIR, exist_ok=True)
    os.makedirs(CLIPS_DIR, exist_ok=True)


def save_frame(frame, frame_index, elapsed_time):
    ensure_dirs()
    frame_path = os.path.join(FRAMES_DIR, f'frame_{frame_index:06d}.jpg')
    cv2.imwrite(frame_path, frame)
    with open(TIMESTAMPS_PATH, 'a') as f:
        f.write(f'{frame_index},{elapsed_time}\n')


def compile_video(fps=15):
    ensure_dirs()
    frames = sorted([f for f in os.listdir(FRAMES_DIR) if f.endswith('.jpg')])
    if not frames:
        return False

    # Calculate actual FPS from timestamps file
    actual_fps = fps
    if os.path.exists(TIMESTAMPS_PATH):
        with open(TIMESTAMPS_PATH, 'r') as f:
            lines = [l.strip() for l in f.readlines() if l.strip()]
        if len(lines) >= 2:
            try:
                first_time = float(lines[0].split(',')[1])
                last_time = float(lines[-1].split(',')[1])
                duration = last_time - first_time
                if duration > 0:
                    actual_fps = len(lines) / duration
                    actual_fps = min(actual_fps, 30)
                    actual_fps = max(actual_fps, 1)
            except Exception:
                actual_fps = fps

    first = cv2.imread(os.path.join(FRAMES_DIR, frames[0]))
    if first is None:
        return False

    height, width, _ = first.shape
    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    out = cv2.VideoWriter(VIDEO_PATH, fourcc, actual_fps, (width, height))

    for frame_file in frames:
        frame = cv2.imread(os.path.join(FRAMES_DIR, frame_file))
        if frame is not None:
            out.write(frame)

    out.release()
    cleanup_frames()
    return True


def extract_clip(start_time, end_time, clip_index, fps=15):
    if not os.path.exists(VIDEO_PATH):
        return None
    if not os.path.exists(TIMESTAMPS_PATH):
        return None

    timestamps = {}
    with open(TIMESTAMPS_PATH, 'r') as f:
        for line in f:
            line = line.strip()
            if ',' in line:
                idx, t = line.split(',')
                timestamps[int(idx)] = float(t)

    if not timestamps:
        return None

    frames_in_range = sorted([
        idx for idx, t in timestamps.items()
        if start_time <= t <= end_time
    ])

    if not frames_in_range:
        return None

    all_frame_indexes = sorted(timestamps.keys())
    start_position = all_frame_indexes.index(frames_in_range[0])
    end_position = all_frame_indexes.index(frames_in_range[-1]) + 1

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        return None

    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    if actual_fps <= 0:
        actual_fps = fps

    end_position = min(end_position, int(total_frames))

    if start_position >= end_position:
        cap.release()
        return None

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_position)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if width == 0 or height == 0:
        cap.release()
        return None

    clip_path = os.path.join(CLIPS_DIR, f'clip_{clip_index}.mp4')
    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    out = cv2.VideoWriter(clip_path, fourcc, actual_fps, (width, height))

    frames_written = 0
    for _ in range(end_position - start_position):
        ret, frame = cap.read()
        if not ret:
            break
        out.write(frame)
        frames_written += 1

    cap.release()
    out.release()

    return clip_path if frames_written > 0 else None


def cleanup_frames():
    if os.path.exists(FRAMES_DIR):
        for f in os.listdir(FRAMES_DIR):
            if f.endswith('.jpg'):
                os.remove(os.path.join(FRAMES_DIR, f))


def cleanup_timestamps():
    if os.path.exists(TIMESTAMPS_PATH):
        os.remove(TIMESTAMPS_PATH)


def cleanup_session_video():
    if os.path.exists(VIDEO_PATH):
        os.remove(VIDEO_PATH)
    cleanup_timestamps()


def cleanup_clips():
    if os.path.exists(CLIPS_DIR):
        for f in os.listdir(CLIPS_DIR):
            if f.endswith('.mp4'):
                os.remove(os.path.join(CLIPS_DIR, f))