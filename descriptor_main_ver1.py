import os
import cv2
import json
import requests
import streamlit as st
import numpy as np
import tempfile

# Constants
SAVE_DIR = "extracted_frames"
CAPTIONS_FILE = "video_captions.json"
API_URL = "https://api-inference.huggingface.co/models/nlpconnect/vit-gpt2-image-captioning"
HEADERS = {"Authorization": "Bearer hf_xxxxxxxxxxxxxxxxxxxxxxxx"}  # Replace with your actual token
NUM_FRAMES_TO_EXTRACT = 100  # Hardcoded number of frames per video

# Ensure save directory exists
os.makedirs(SAVE_DIR, exist_ok=True)

def detect_scene_changes(video_path):
    """Extracts frames based on scene changes using histogram comparison."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        st.error("Error opening video")
        return []

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    scene_frames = []
    last_histogram = None
    frame_step = max(1, total_frames // NUM_FRAMES_TO_EXTRACT)  # Adjust step size

    for i in range(0, total_frames, frame_step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if not ret:
            continue

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        histogram = cv2.calcHist([gray_frame], [0], None, [256], [0, 256])
        histogram = cv2.normalize(histogram, histogram).flatten()

        if last_histogram is not None:
            similarity = cv2.compareHist(last_histogram, histogram, cv2.HISTCMP_CORREL)
            if similarity < 0.9:  # Threshold for scene change
                scene_frames.append((i, frame))
        else:
            scene_frames.append((i, frame))

        last_histogram = histogram

        if len(scene_frames) >= NUM_FRAMES_TO_EXTRACT:
            break

    cap.release()
    return scene_frames

def save_frames(scene_frames, video_id):
    """Saves detected scene change frames to disk."""
    frame_paths = []
    frame_save_path = os.path.join(SAVE_DIR, str(video_id))
    os.makedirs(frame_save_path, exist_ok=True)

    for idx, (frame_pos, frame) in enumerate(scene_frames):
        frame_filename = os.path.join(frame_save_path, f"{idx}.png")
        cv2.imwrite(frame_filename, frame)
        frame_paths.append(frame_filename)

    return frame_paths

def query_api(image_path):
    #Sends an image to the Hugging Face API and returns the generated caption.
    try:
        with open(image_path, "rb") as f:
            data = f.read()
        response = requests.post(API_URL, headers=HEADERS, data=data)
        result = response.json()

        if isinstance(result, list) and "generated_text" in result[0]:
            return result[0]["generated_text"]
        else:
            return "Error generating caption"
    except Exception as e:
        return "API request failed"

def generate_captions(video_id, frame_paths):
    #Generates captions for extracted frames and saves results in JSON.
    captions = {"video_id": video_id, "captions": {}}

    for idx, frame_path in enumerate(frame_paths):
        caption = query_api(frame_path)
        captions["captions"][f"frame_{idx}"] = caption

    return captions

def save_captions(captions_list):
    Saves all video captions into a JSON file.
    try:
        if os.path.exists(CAPTIONS_FILE):
            with open(CAPTIONS_FILE, "r") as f:
                existing_data = json.load(f)
        else:
            existing_data = []

        existing_data.append(captions_list)

        with open(CAPTIONS_FILE, "w") as f:
            json.dump(existing_data, f, indent=4)

        st.success("Captions saved successfully!")
    except Exception as e:
        st.error(f"Error saving captions: {e}")

def process_uploaded_video(uploaded_file):
    Processes the user-uploaded video.
    if uploaded_file is None:
        st.warning("Please upload a video file.")
        return

    video_id = uploaded_file.name.split('.')[0]

    #Save the uploaded video to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_video:
        temp_video.write(uploaded_file.read())
        temp_video_path = temp_video.name

    st.success(f"Uploaded video: {uploaded_file.name}")

    #Detect scene changes and extract frames
    scene_frames = detect_scene_changes(temp_video_path)
    frame_paths = save_frames(scene_frames, video_id)

    if frame_paths:
        captions = generate_captions(video_id, frame_paths)
        save_captions(captions)

        st.subheader("Generated Captions:")
        for key, caption in captions["captions"].items():
            st.write(f"**{key.replace('_', ' ').title()}**: {caption}")

    # Cleanup temporary video file
    os.remove(temp_video_path)

# Streamlit UI
st.title("🎥 Smart Video Captioning App")
st.write("Upload a video, extract key frames based on scene changes, and generate captions using AI.")

uploaded_video = st.file_uploader("Upload your video (MP4)", type=["mp4"])

if uploaded_video:
    process_uploaded_video(uploaded_video)
