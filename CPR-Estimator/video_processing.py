import cv2
import numpy as np

def calculate_optical_flow(prev_frame, curr_frame):
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
    flow = cv2.calcOpticalFlowFarneback(prev_gray, curr_gray, None, 0.5, 1, 15, 1, 3, 1.2, 0)
    return flow

def process_video(filepath, start):
    cap = cv2.VideoCapture(filepath)
    if not cap.isOpened():
        print(f"Error opening video file: {filepath}")
        return None

    frames = []
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    for _ in range(150):
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (224, 224))
        frames.append(frame)
    cap.release()

    if len(frames) < 150:
        print("Not enough frames for processing")
        return None

    prev_frame = None
    sign_history = []
    saved_images = []
    count = 0
    for frame in frames:
        if prev_frame is not None:
            optical_flow = calculate_optical_flow(prev_frame, frame)
            vertical_mean = np.mean(optical_flow[..., 1])
            current_sign = np.sign(vertical_mean)
            if len(sign_history) > 0 and current_sign != sign_history[-1]:
                if len(sign_history) >= 3:
                    saved_images.append(frame)
                    count += 0.5
                sign_history = [current_sign]
            else:
                sign_history.append(current_sign)
        prev_frame = frame

    if saved_images:
        critical_avg_image = np.average(np.array(saved_images), axis=0)
        return critical_avg_image, count

    print("No significant frames found")
    return None
