# Movie Descriptor Project 🎥 

## Project Overview
The **Movie Descriptor Project** is designed to create an audio description system for movies, helping visually impaired users by generating detailed scene narrations. The system processes video content, recognizes objects, actions, and activities, and provides emotionally engaging audio descriptions.

## Branch Information

### 1. **testing**  
   - Active testing branch where new features and fixes are tested and evaluated before merger into the dev (MAIN) branch.

### 2. **info**


This project extracts key frames from user-uploaded videos, generates AI-based captions using the **ViT-GPT2 Image Captioning model**, and saves the results in JSON format.  

---

## **🚀 Features**
- **User uploads a video** via Streamlit UI  
- **Frame extraction based on scene changes** (100 frames per 10 seconds)  
- **Captions generated using Hugging Face API**  
- **Captions are stored in JSON with video ID**  
- **Inference time**: ~4-5 seconds per 10 seconds of video  

---

## **🛠️ Setup & Installation**  

### **1️⃣ Clone the Repository**
```bash
git clone https://github.com/yourusername/video-captioning-app.git
cd video-captioning-app
python -m venv venv
source venv/bin/activate  # macOS/Linux
venv\Scripts\activate  # Windows
pip install -r requirements.txt

🎯 Usage
1️⃣ Run the Streamlit App
bash
Copy
Edit
streamlit run app.py
2️⃣ Upload a Video
Supported format: MP4
Video is processed, frames are extracted, and captions are generated
3️⃣ View Captions
Captions are displayed in the UI
Results are saved in video_captions.json

🖼️ Model Details
Model Name: nlpconnect/vit-gpt2-image-captioning
API: Hugging Face Inference API
Frame Size: 100 frames per 10 seconds
Inference Time: ~4-5 seconds per 10s of video

🔧 Configuration
Edit app.py for custom settings:

python
Copy
Edit
NUM_FRAMES_TO_EXTRACT = 100  # Frames per 10s of video

📩 API Key
Replace "Bearer hf_xxxxxxxxxxxxxxxxxxxxxxxx" in app.py with your actual Hugging Face API key.

🔗 References
ViT-GPT2 Model
Streamlit Documentation
OpenCV


📌 requirements.txt 

streamlit
opencv-python
numpy
requests
pandas
pillow
