import cv2
import streamlit as st
import tempfile
import os
import base64
import numpy as np
from groq import Groq
from gtts import gTTS

# ------------------ Configuration ------------------
vision_model = "llama-3.2-11b-vision-preview"
groq_api_key = "your_groq_api_key_here"
language_model  = "llama-3.1-8b-instant"
groq_client = Groq(api_key="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")

# ------------------ Helper Functions ------------------
def create_frame_grid(frames):
    """Creates a 2x3 grid from six frames using OpenCV."""
    if len(frames) < 6:
        return None
    try:
        row1 = np.hstack(frames[:3])
        row2 = np.hstack(frames[3:])
        return np.vstack([row1, row2])
    except Exception as e:
        print(f"Error creating frame grid: {e}")
        return None

def query_vision(image):
    try:
        success, img_encoded = cv2.imencode(".png", image)
        if not success:
            return "Image encoding failed"
        
        b64_data = base64.b64encode(img_encoded).decode("utf-8")
        data_url = f"data:image/png;base64,{b64_data}"
        messages = [
            {"role": "user", "content": "Generate a descriptive caption for this scene grid of 6 frames from the same video in a way that conveys the scene to a visually impaired person under 20 words"},
            {"role": "user", "content": [
                {"type": "text", "text": ""},
                {"type": "image_url", "image_url": {"url": data_url}}
            ]}
        ]
        
        completion = groq_client.chat.completions.create(
            model=vision_model,
            messages=messages,
            temperature=1,
            max_completion_tokens=100,
            top_p=1,
            stream=False,
        )
        return completion.choices[0].message.content.strip() if completion.choices else "No caption generated"
    except Exception as e:
        print(f"Error querying Groq API: {e}")
        return None

def generate_tts(prompt):
    try:
        tts = gTTS(prompt, lang='en')
        audio_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3").name
        tts.save(audio_path)
        return audio_path
    except Exception as e:
        print(f"Error generating TTS: {e}")
        return None

def summarize_captions(captions, length):
    try:
        prompt = f"Summarize the following descriptions of every 15th frame of the same video under {length} words into a concise video description for a visually impaired person: " + " ".join(captions)
        completion = groq_client.chat.completions.create(
            model=vision_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5,
            max_completion_tokens=200,
            top_p=1,
            stream=False,
        )
        return completion.choices[0].message.content.strip() if completion.choices else "No final caption generated"
    except Exception as e:
        print(f"Error summarizing captions: {e}")
        return None

def play_video_with_audio(video_path, audio_path):
    try:
        video_bytes = open(video_path, "rb").read()
        audio_bytes = open(audio_path, "rb").read()
        
        video_base64 = base64.b64encode(video_bytes).decode("utf-8")
        audio_base64 = base64.b64encode(audio_bytes).decode("utf-8")
        
        video_html = f'''
        <video id="video" width="640" height="360" controls autoplay>
            <source src="data:video/mp4;base64,{video_base64}" type="video/mp4">
        </video>
        <audio id="audio" autoplay>
            <source src="data:audio/mp3;base64,{audio_base64}" type="audio/mp3">
        </audio>
        <script>
            document.getElementById('video').addEventListener('play', function() {{
                document.getElementById('audio').play();
            }});
        </script>
        '''
        st.markdown(video_html, unsafe_allow_html=True)
    except Exception as e:
        print(f"Error playing video with audio: {e}")

# ------------------ Streamlit UI ------------------
st.title("🎥 Live Video Captioning")
st.write("Upload a video, and a final detailed caption will be generated.")

uploaded_video = st.file_uploader("Upload your video (MP4/MOV)", type=["mp4", "mov"])

if uploaded_video:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_video:
        temp_video.write(uploaded_video.read())
        temp_video_path = temp_video.name
    
    cap = cv2.VideoCapture(temp_video_path)
    if not cap.isOpened():
        st.error("Error: Could not open video file")
    else:
        all_captions = []
        frame_count = 0
        frame_skip = 15
        frame_buffer = []
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        progress_bar = st.progress(0)
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_count % frame_skip == 0:
                frame_buffer.append(frame)
                if len(frame_buffer) == 6:
                    grid = create_frame_grid(frame_buffer)
                    if grid is not None:
                        caption = query_vision(grid)
                        if caption:
                            all_captions.append(caption)
                    frame_buffer.clear()
            
            frame_count += 1
            progress_bar.progress(min(100, int((frame_count / max(1, total_frames)) * 100)))
        
        cap.release()
        
        if all_captions:
            final_caption = summarize_captions(all_captions, max(1, (total_frames // 30) * 2))
            st.markdown(f"## Final Caption: {final_caption}")
            audio_path = generate_tts(final_caption)
            if audio_path:
                play_video_with_audio(temp_video_path, audio_path)
        else:
            st.error("No captions generated.")
