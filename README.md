# Leotele: Real-Time News Transcription & Scene Description

**Leotele** is a Python pipeline and webapp that performs real-time audio transcription and visual scene description from a live news video stream (Al Jazeera English). It leverages cloud GPU infrastructure (Modal), state-of-the-art speech-to-text and vision-language models, and provides a simple web frontend for live monitoring.

## Features

- **Real-time audio transcription** from an m3u8 video stream (Al Jazeera English)
- **Scene description** for each video frame using a Vision-Language Model (Qwen2-VL via SGLang)
- **Cloud GPU inference** using Modal for both ASR and VLM
- **Live web frontend** that displays the latest transcription and scene description, auto-refreshing every 2 seconds
- **No data is stored**: Only the latest result is kept in a local JSON file (`stream.json`)
- **Easy to deploy and extend**

## How it works

1. **Audio & Video Ingestion**: `ingest.py` reads audio and video frames from a live m3u8 stream using ffmpeg.
2. **Transcription**: Audio is transcribed every 10 seconds (if not silent) using NVIDIA's Parakeet ASR model running on Modal.
3. **Scene Description**: For each transcription, a video frame is captured and described using Qwen2-VL (via SGLang, also on Modal).
4. **Result Storage**: The latest transcription and scene description are saved to `stream.json` (overwriting previous content).
5. **Web Frontend**: `view.py` serves a simple web page that auto-refreshes to show the latest transcription and scene description, styled as a screenplay.

## File Overview

- `ingest.py` — Main pipeline: audio/video ingestion, Modal integration, result saving
- `scene_describer.py` — Modal endpoint for Qwen2-VL scene description (SGLang)
- `view.py` — Web frontend (Fasthtml) for live display
- `stream.json` — Stores only the latest result (overwritten each update)
- `test_m3u8_stream.py`, `test_capture_frame.py` — Utilities for testing stream ingestion
- `requirements.txt` — Python dependencies

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

This is a live transcription and visual description of the Al Jazeera English channel. No data is stored. You can view the code at [https://github.com/aastroza/leotele](https://github.com/aastroza/leotele)

## License

MIT License