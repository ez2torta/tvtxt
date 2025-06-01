import subprocess
import uuid

AUDIO_STREAM_URL = "https://live-hls-web-aje.getaj.net/AJE/03.m3u8"

def capture_frame(m3u8_url, output_path=None):
    if output_path is None:
        output_path = f"frame_{uuid.uuid4().hex}.jpg"
    ffmpeg_cmd = [
        "ffmpeg", "-y", "-loglevel", "quiet", "-i", m3u8_url,
        "-frames:v", "1", "-q:v", "2", output_path
    ]
    subprocess.run(ffmpeg_cmd, check=True)
    return output_path

if __name__ == "__main__":
    print(f"Capturing frame from {AUDIO_STREAM_URL} ...")
    image_path = capture_frame(AUDIO_STREAM_URL)
    print(f"Frame saved to: {image_path}")
