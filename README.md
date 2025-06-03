# tvtxt 📺✨

> **⚠️ Work in Progress - Technology Showcase**  
> This is an experimental MVP demonstrating real-time AI capabilities. Not intended as a production-ready product.

**Turn any live TV stream into a real-time movie script. AI watches, transcribes, and writes television as cinema.**

Ever wondered what your favorite TV show would look like as a screenplay? tvtxt is an AI-powered pipeline that watches live television streams and transforms them into properly formatted movie scripts in real-time. Think of it as having a tireless scriptwriter that never blinks, never sleeps, and never misses a moment.

## Live Demo
Watch the magic unfold in real-time: [tvtxt live demo](https://tvtxt.com/)

![screenshot](/tvtxt.PNG)


## Project Status

This is a **proof-of-concept showcase** built to demonstrate the integration of several cutting-edge technologies:
- Real-time speech recognition.
- Vision-language understanding.
- Cloud-native AI inference.
- Live streaming media processing.

**What this is:**
- A technology demonstration.
- An experimental MVP.
- A learning playground for AI + media processing.

**What this is NOT:**
- A production-ready application.
- A commercial product.
- A fully-featured streaming service.

## The magic behind the curtain

**tvtxt** combines cutting-edge AI models with cloud infrastructure to create a TV-to-screenplay transformation:


### **[Modal](https://modal.com/)**
Modal handles our cloud GPU infrastructure, running two critical AI workloads:
- **[Parakeet ASR Model (NVIDIA)](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v2)** : Transcribes speech with remarkable accuracy and speed.
- **[Qwen2-VL Vision-Language Model](https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct)**: Describes visual scenes with cinematic flair.

### **[Outlines](https://github.com/dottxt-ai/outlines)**
Ensures our vision model outputs perfectly formatted JSON responses:
- **Schema enforcement**: Guarantees consistent screenplay structure.

### Others

- **[Azure Blob Storage:](https://azure.microsoft.com/en-us/products/storage/blobs)**Temporarily stores captured video frames for visual analysis.
- **[Redis Cloud:](https://redis.io/cloud/)** Acts as the bridge between our backend pipeline and frontend display.
- **[FastHTML:](https://www.fastht.ml/)** Creates our live web interface with authentic screenplay styling.
- **[FFmpeg:](https://ffmpeg.org/)** The unsung hero that handles all media processing.

## How the Magic Happens

1. **🎥 Stream Capture**: FFmpeg latches onto a live TV stream, extracting both audio and video.
2. **🎧 Audio Analysis**: Every 10 seconds, audio chunks are sent to Modal's Parakeet ASR model for transcription.
3. **📸 Frame Extraction**: When speech is detected, FFmpeg captures a corresponding video frame.
4. **☁️ Image Upload**: The frame is uploaded to Azure Blob Storage and gets a public URL.
5. **👁️ Visual Understanding**: Modal's Qwen2-VL model analyzes the image and generates a screenplay-formatted scene description.
6. **💾 Memory Update**: The latest transcription and scene description are saved to Redis Cloud.
7. **🖥️ Live Display**: FastHTML serves a web page that auto-refreshes, showing the generated screenplay.
8. **🔄 Repeat**: The cycle continues, creating an ever-updating script of live television.

## Installation & Setup

### 1. **Environment Setup**
```bash
uv venv
source venv/bin/activate  # or `.venv\Scripts\activate` on Windows
uv pip install -r requirements.txt
modal token new
```

### 2. **Configure Your Credentials**
Create a `.env` file with your secret weapons:
```env
# Azure Blob Storage (for frame storage)
AZURE_STORAGE_CONNECTION_STRING=your_azure_connection_string

# Redis Cloud (for state management)
REDIS_HOST=your_redis_host
REDIS_PORT=your_redis_port
REDIS_USERNAME=your_redis_username
REDIS_PASSWORD=your_redis_password

# HuggingFace (for model access)
HF_TOKEN=your_huggingface_token

# Modal endpoint (will be generated after deployment)
IMAGE_DESCRIBER_URL=your_modal_endpoint_url
```

### 3. **Deploy the Vision AI**
Launch your scene description model to the cloud:
```bash
modal deploy scene_describer.py
```
*Note: Copy the generated endpoint URL to your `.env` file as `IMAGE_DESCRIBER_URL`*

### 4. **Start the Show**
Fire up the transcription pipeline:
```bash
modal run ingest.py
```

### 5. **Watch the Magic**
Launch the web interface:
```bash
cd app
python main.py
```

Open your browser to `http://localhost:5001` and watch as live TV transforms into screenplay format before your eyes!

## Philosophy

tvtxt embraces ephemerality by design. Like live theater, each moment exists only in the present:
- **No databases**: Only the current scene matters.
- **No history**: Previous scripts vanish like morning mist.
- **No storage**: Frames and audio exist only long enough to be processed.


## Disclaimer

This project demonstrates real-time AI transcription and visual analysis using Al Jazeera English as a public live stream. No content is stored, archived, or redistributed. The system processes live broadcasts in real-time for educational and demonstration purposes only.