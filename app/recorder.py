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
    """
    Save one webcam frame as a jpg file.
    Also write the frame index and its timestamp to a text file.
    This lets us find exactly which frames correspond to any time range.
    """
    ensure_dirs()
    frame_path = os.path.join(FRAMES_DIR, f'frame_{frame_index:06d}.jpg')
    cv2.imwrite(frame_path, frame)
    with open(TIMESTAMPS_PATH, 'a') as f:
        f.write(f'{frame_index},{elapsed_time}\n')


def compile_video(fps=15):
    """
    After recording stops, combine all saved jpg frames into one mp4 video.
    Frames are named frame_000001.jpg, frame_000002.jpg etc so they sort correctly.
    """
    ensure_dirs()
    frames = sorted([f for f in os.listdir(FRAMES_DIR) if f.endswith('.jpg')])
    if not frames:
        return False

    first = cv2.imread(os.path.join(FRAMES_DIR, frames[0]))
    if first is None:
        return False

    height, width, _ = first.shape
    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    out = cv2.VideoWriter(VIDEO_PATH, fourcc, fps, (width, height))

    for frame_file in frames:
        frame = cv2.imread(os.path.join(FRAMES_DIR, frame_file))
        if frame is not None:
            out.write(frame)

    out.release()
    cleanup_frames()
    return True


def extract_clip(start_time, end_time, clip_index, fps=15):
    """
    Extract a short clip from the session video.
    Uses the timestamps.txt file to find exactly which frames
    fall between start_time and end_time seconds.
    """
    if not os.path.exists(VIDEO_PATH):
        return None
    if not os.path.exists(TIMESTAMPS_PATH):
        return None

    # Read the timestamp mapping file
    # Each line is: frame_index,elapsed_time
    timestamps = {}
    with open(TIMESTAMPS_PATH, 'r') as f:
        for line in f:
            line = line.strip()
            if ',' in line:
                idx, t = line.split(',')
                timestamps[int(idx)] = float(t)

    if not timestamps:
        return None

    # Find all frame indexes that fall in our time range
    frames_in_range = sorted([
        idx for idx, t in timestamps.items()
        if start_time <= t <= end_time
    ])

    if not frames_in_range:
        return None

    # The first frame in range = where to start in the video
    # We need to convert frame index to position in the compiled video
    # The compiled video has frames in order 0,1,2,3...
    # so we need the position of our frame in the sorted list of all frames
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
    """Delete individual frame jpg files after video is compiled"""
    if os.path.exists(FRAMES_DIR):
        for f in os.listdir(FRAMES_DIR):
            if f.endswith('.jpg'):
                os.remove(os.path.join(FRAMES_DIR, f))


def cleanup_timestamps():
    """Delete the timestamps mapping file"""
    if os.path.exists(TIMESTAMPS_PATH):
        os.remove(TIMESTAMPS_PATH)


def cleanup_session_video():
    """Delete the compiled session video"""
    if os.path.exists(VIDEO_PATH):
        os.remove(VIDEO_PATH)
    cleanup_timestamps()


def cleanup_clips():
    """Delete all clip files"""
    if os.path.exists(CLIPS_DIR):
        for f in os.listdir(CLIPS_DIR):
            if f.endswith('.mp4'):
                os.remove(os.path.join(CLIPS_DIR, f))