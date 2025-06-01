# Leotele: Live TV Transcription & Description

**Leotele** is a Python app that combines real-time audio transcription and visual scene description from a live news stream (Al Jazeera English). It runs on cloud GPUs (via Modal), uses cutting-edge speech and vision models, and offers a simple web interface to follow everything live.

## What It Does

- 🗣️ **Live audio transcription** from an m3u8 video stream (Al Jazeera English)
- 🎞️ **Scene descriptions** from video frames using a Vision-Language Model (Qwen2-VL via SGLang)
- ☁️ **Runs in the cloud** using Modal for both transcription and visual analysis
- 🌐 **Live web interface** that updates every 2 seconds with the latest text
- 🛑 **No storage**: Only the latest result is kept in a local JSON file (`stream.json`)
- 🛠️ **Easy to set up and customize**

## How It Works

1. **Grabs audio and video**: `ingest.py` pulls frames and audio from a live stream using ffmpeg.
2. **Transcribes the audio**: Every 10 seconds (if there's speech), it sends the audio to NVIDIA’s Parakeet ASR on Modal.
3. **Describes what’s on screen**: Grabs a matching frame and sends it to Qwen2-VL (via SGLang) for a description.
4. **Saves the result**: Updates `stream.json` with the latest transcription and description.
5. **Shows it live**: `view.py` runs a basic web viewer that updates automatically, styled like a screenplay.

## What’s in the Repo

- `ingest.py` — Main pipeline: grabs audio/video, talks to Modal, saves output
- `scene_describer.py` — Modal endpoint for visual scene description
- `view.py` — Simple web interface built with Fasthtml
- `stream.json` — Keeps the latest result (gets overwritten each time)
- `test_m3u8_stream.py`, `test_capture_frame.py` — Tools to test stream input
- `requirements.txt` — Dependencies you’ll need

## Get Started

1. **Install everything**

   ```bash
   pip install -r requirements.txt


## Quickstart

1. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

2. **Set up environment variables**
   - Configure your Azure Blob Storage and Modal credentials in a `.env` file.

3. **Run the pipeline**

   ```bash
   python ingest.py
   ```

4. **Start the web frontend**

   ```bash
   python view.py
   ```

5. **Open your browser** to `http://localhost:8000` (or the port shown) to see live results.

## Disclaimer

This is a live transcription and visual description of the Al Jazeera English channel. No data is stored.