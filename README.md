# Leotele: Real-Time News Transcription & Scene Description

**Leotele** is a Python pipeline and webapp that performs real-time audio transcription and visual scene description from a live news video stream (Al Jazeera English). It leverages cloud GPU infrastructure (Modal), state-of-the-art speech-to-text and vision-language models, Redis Cloud for state sharing, and provides a simple web frontend for live monitoring.

## Features

- **Real-time audio transcription** from an m3u8 video stream (Al Jazeera English)
- **Scene description** for each video frame using a Vision-Language Model (Qwen2-VL via Outlines/SGLang)
- **Cloud GPU inference** using Modal for both ASR and VLM
- **Live web frontend** (Fasthtml) that displays the latest transcription and scene description, auto-refreshing every 2 seconds
- **No persistent data storage**: Only the latest result is kept in Redis Cloud
- **Easy to deploy and extend**

## How it works

1. **Audio & Video Ingestion**: `ingest.py` reads audio and video frames from a live m3u8 stream using ffmpeg.
2. **Transcription**: Audio is transcribed every 10 seconds (if not silent) using NVIDIA's Parakeet ASR model running on Modal.
3. **Scene Description**: For each transcription, a video frame is captured and described using Qwen2-VL (via Outlines/SGLang, also on Modal). The endpoint returns a JSON with `heading`, `action`, and `character` fields.
4. **Result Storage**: The latest transcription and scene description are saved to Redis Cloud (key: `leotele:latest`).
5. **Web Frontend**: `view.py` serves a simple web page that auto-refreshes to show the latest transcription and scene description, styled as a screenplay.

## File Overview

- `ingest.py` — Main pipeline: audio/video ingestion, Modal integration, Redis Cloud integration
- `scene_describer.py` — Modal endpoint for Qwen2-VL scene description (Outlines/SGLang)
- `view.py` — Web frontend (Fasthtml) for live display, reads from Redis Cloud
- `.env` — Stores credentials for Azure, Modal, and Redis Cloud
- `requirements.txt` — Python dependencies (includes modal, outlines, sglang, redis, etc.)
- `test_m3u8_stream.py`, `test_capture_frame.py` — Utilities for testing stream ingestion

## Quickstart

1. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

2. **Set up environment variables**
   - Configure your Azure Blob Storage, Modal, and Redis Cloud credentials in a `.env` file.

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