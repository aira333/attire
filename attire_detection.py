import cv2
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import time
import sys

# Load MoveNet
movenet_model = hub.load("https://tfhub.dev/google/movenet/singlepose/lightning/4")
movenet = movenet_model.signatures['serving_default']

def extract_frames(video_path, n=3):
    "Extract 3 frames"
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    indices = [total_frames//4, total_frames//2, 3*total_frames//4]
    frames = []
    
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            if frame.shape[0] > 480:
                scale = 480 / frame.shape[0]
                new_w, new_h = int(frame.shape[1] * scale), int(frame.shape[0] * scale)
                frame = cv2.resize(frame, (new_w, new_h))
            frames.append(frame)
    
    cap.release()
    return frames

def detect_pose_simple(frame):
    """Get torso region"""
    input_frame = cv2.resize(frame, (192, 192))
    input_frame = cv2.cvtColor(input_frame, cv2.COLOR_BGR2RGB)
    input_frame = tf.cast(input_frame, dtype=tf.int32)
    input_frame = tf.expand_dims(input_frame, axis=0)
    
    outputs = movenet(input_frame)
    return outputs['output_0'].numpy()[0, 0, :, :]

def get_torso_region(frame, keypoints):
    """classification based on 3 reliable indicators"""
    h, w = frame.shape[:2]
    
    kp = keypoints.copy()
    kp[:, 1] *= w
    kp[:, 0] *= h
    
    nose = kp[0]
    left_shoulder = kp[5]
    right_shoulder = kp[6]
    left_hip = kp[11]
    right_hip = kp[12]
    
    if left_shoulder[2] < 0.3 or right_shoulder[2] < 0.3:
        return frame[:h//2, w//4:3*w//4]
    
    top = int(min(left_shoulder[0], right_shoulder[0]) - 20)
    bottom = int(max(left_hip[0], right_hip[0])) if left_hip[2] > 0.3 and right_hip[2] > 0.3 else int(left_shoulder[0] + 150)
    left = int(min(left_shoulder[1], right_shoulder[1]) - 30)
    right = int(max(left_shoulder[1], right_shoulder[1]) + 30)
    
    top = max(0, top)
    bottom = min(h, bottom)
    left = max(0, left)
    right = min(w, right)
    
    if bottom <= top or right <= left:
        return frame[:h//2, w//4:3*w//4]
    
    return frame[top:bottom, left:right]

def classify_simple(torso_region):
    if torso_region.size == 0:
        return "informal", 0.3, ["no_torso_region"]
    
    gray = cv2.cvtColor(torso_region, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(torso_region, cv2.COLOR_BGR2HSV)
    
    formal_score = 0.0
    factors = []
    
    darkness = 1.0 - (np.mean(gray) / 255.0)
    saturation = np.mean(hsv[:, :, 1]) / 255.0
    
    if darkness > 0.6:
        formal_score += 0.4
        factors.append(f"dark:{darkness:.2f}")
    elif darkness > 0.4 and saturation < 0.3:
        formal_score += 0.3
        factors.append(f"neutral:{darkness:.2f}")
    elif saturation > 0.5:
        formal_score -= 0.3
        factors.append(f"bright_color:{saturation:.2f}")
    
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    if laplacian_var < 300:
        formal_score += 0.2
        factors.append(f"smooth_texture:{laplacian_var:.0f}")
    elif laplacian_var > 800:
        formal_score -= 0.2
        factors.append(f"rough_texture:{laplacian_var:.0f}")
    
    edges = cv2.Canny(gray, 30, 100)
    edge_density = np.sum(edges > 0) / edges.size
    
    if edge_density > 0.15:
        formal_score -= 0.3
        factors.append(f"patterns:{edge_density:.2f}")
    elif edge_density < 0.05:
        formal_score += 0.1
        factors.append(f"clean:{edge_density:.2f}")
    
    final_score = 0.5 + formal_score
    final_score = np.clip(final_score, 0.0, 1.0)
    
    if final_score >= 0.7:
        return "formal", final_score, factors
    elif final_score <= 0.3:
        return "informal", final_score, factors
    else:
        if formal_score > 0.2:
            return "formal", final_score, factors + ["lean_formal"]
        else:
            return "informal", final_score, factors + ["lean_informal"]

def process_video_minimal(video_path):
    """processing pipeline"""
    start_time = time.time()
    
    print(f" Processing: {video_path}")
    
    frames = extract_frames(video_path, 3)
    
    print(f" analysis of {len(frames)} frames...")
    
    results = []
    for i, frame in enumerate(frames):
        keypoints = detect_pose_simple(frame)
        torso = get_torso_region(frame, keypoints)
        
        style, confidence, factors = classify_simple(torso)
        results.append((style, confidence))
        
        factors_str = ", ".join(factors[:2])
        print(f"   Frame {i+1}: {style.upper()} ({confidence:.2f}) - {factors_str}")
    
    formal_votes = sum(1 for style, _ in results if style == "formal")
    avg_confidence = np.mean([conf for _, conf in results])
    
    final_style = "formal" if formal_votes > len(results) / 2 else "informal"
    processing_time = time.time() - start_time
    
    print(f"\n{'='*30}")
    print("RESULT")
    print('='*30)
    print(f"Style: {final_style.upper()}")
    print(f"Confidence: {avg_confidence:.2f}")
    print(f"Time: {processing_time:.1f}s")
    
    return final_style, avg_confidence, processing_time

if __name__ == "__main__":
    video_path = sys.argv[1]
    result = process_video_minimal(video_path)