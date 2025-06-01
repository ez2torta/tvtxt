import asyncio
import os
import sys
import subprocess
import re

from loguru import logger

import modal

# URL of the m3u8 stream to use for audio input
AUDIO_STREAM_URL = "https://live-hls-web-aje.getaj.net/AJE/03.m3u8"

# Audio processing parameters
TARGET_SAMPLE_RATE = 16_000
CHUNK_SIZE = 16_000  # 0.25 second of audio at 16kHz, mono, 16-bit PCM
FORCE_TRANSCRIBE_MS = 10000  # Force transcription if buffer exceeds this many ms (default: 5 seconds)

app = modal.App("leotele-realtime-m3u8")

model_cache = modal.Volume.from_name("model-cache", create_if_missing=True)

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.0-cudnn-devel-ubuntu22.04", add_python="3.12"
    )
    .env(
        {
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            "HF_HOME": "/cache",
            "DEBIAN_FRONTEND": "noninteractive",
            "CXX": "g++",
            "CC": "g++",
        }
    )
    .apt_install("ffmpeg")
    .pip_install(
        "hf_transfer==0.1.9",
        "huggingface_hub[hf-xet]==0.31.2",
        "nemo_toolkit[asr]==2.3.0",
        "cuda-python==12.8.0",
        "fastapi==0.115.12",
        "numpy<2",
        "pydub==0.25.1",
    )
    .entrypoint([])
)

END_OF_STREAM = b"END_OF_STREAM_8f13d09"

def stream_audio_from_m3u8(m3u8_url=None, chunk_size=None):
    if m3u8_url is None:
        m3u8_url = AUDIO_STREAM_URL
    if chunk_size is None:
        chunk_size = CHUNK_SIZE
    ffmpeg_cmd = [
        "ffmpeg", "-loglevel", "quiet", "-i", m3u8_url,
        "-ac", "1", "-ar", str(TARGET_SAMPLE_RATE), "-f", "s16le", "-"
    ]
    proc = subprocess.Popen(ffmpeg_cmd, stdout=subprocess.PIPE)
    try:
        while True:
            chunk = proc.stdout.read(chunk_size * 2)
            if not chunk or len(chunk) < chunk_size * 2:
                break
            yield chunk
    finally:
        proc.terminate()
        proc.wait()

@app.cls(volumes={"/cache": model_cache}, gpu="a10g", image=image)
@modal.concurrent(max_inputs=14, target_inputs=10)
class Parakeet:
    @modal.enter()
    def load(self):
        import logging
        import nemo.collections.asr as nemo_asr

        logging.getLogger("nemo_logger").setLevel(logging.CRITICAL)

        self.model = nemo_asr.models.ASRModel.from_pretrained(
            model_name="nvidia/parakeet-tdt-0.6b-v2"
        )

    def transcribe(self, audio_bytes: bytes) -> str:
        import numpy as np
        audio_data = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32)

        with NoStdStreams():
            output = self.model.transcribe([audio_data])

        return output[0].text

    @modal.method()
    async def run_with_queue(self, q: modal.Queue):
        from pydub import AudioSegment

        audio_segment = AudioSegment.empty()
        accumulated_ms = 0
        chunk_ms = int(1000 * CHUNK_SIZE / TARGET_SAMPLE_RATE)  # ms per chunk
        try:
            while True:
                chunk = await q.get.aio(partition="audio")
                if chunk == END_OF_STREAM:
                    # Force transcription of any remaining audio
                    if len(audio_segment) > 0:
                        text = self.transcribe(audio_segment.raw_data)
                        if text:
                            await q.put.aio(text, partition="transcription")
                    await q.put.aio(END_OF_STREAM, partition="transcription")
                    break

                audio_segment, text, force_transcribed = await self.handle_audio_chunk(
                    chunk, audio_segment, chunk_ms, accumulated_ms
                )
                if force_transcribed:
                    accumulated_ms = 0
                else:
                    accumulated_ms += chunk_ms
                if text:
                    await q.put.aio(text, partition="transcription")
        except Exception as e:
            logger.error(f"Error handling queue: {type(e)}: {e}")
            return

    async def handle_audio_chunk(
        self,
        chunk: bytes,
        audio_segment,
        chunk_ms,
        accumulated_ms,
        silence_thresh=-45,  # dB
        min_silence_len=1000,  # ms
        force_transcribe_ms=FORCE_TRANSCRIBE_MS,  # ms
    ):
        from pydub import AudioSegment, silence

        new_audio_segment = AudioSegment(
            data=chunk,
            channels=1,
            sample_width=2,
            frame_rate=TARGET_SAMPLE_RATE,
        )
        audio_segment += new_audio_segment

        silent_windows = silence.detect_silence(
            audio_segment,
            min_silence_len=min_silence_len,
            silence_thresh=silence_thresh,
        )

        # Normal silence-based transcription
        if len(silent_windows) > 0:
            last_window = silent_windows[-1]
            if last_window[0] == 0 and last_window[1] == len(audio_segment):
                audio_segment = AudioSegment.empty()
                return audio_segment, None, False
            segment_to_transcribe = audio_segment[: last_window[1]]
            audio_segment = audio_segment[last_window[1] :]
            try:
                text = self.transcribe(segment_to_transcribe.raw_data)
                return audio_segment, text, True
            except Exception as e:
                logger.error(f"Transcription error: {e}")
                raise e

        # Force transcription if buffer exceeds force_transcribe_ms
        if len(audio_segment) >= force_transcribe_ms:
            try:
                text = self.transcribe(audio_segment.raw_data)
                audio_segment = AudioSegment.empty()
                return audio_segment, text, True
            except Exception as e:
                logger.error(f"Transcription error: {e}")
                raise e

        return audio_segment, None, False

@app.local_entrypoint()
async def main():
    m3u8_url = AUDIO_STREAM_URL
    print(f"🎧 Streaming from: {m3u8_url}")
    with modal.Queue.ephemeral() as q:
        Parakeet().run_with_queue.spawn(q)
        send = asyncio.create_task(send_live_audio(q, m3u8_url))
        recv = asyncio.create_task(receive_text(q))
        await asyncio.gather(send, recv)
    print("✅ Live transcription ended.")

async def send_live_audio(q, m3u8_url):
    for chunk in stream_audio_from_m3u8(m3u8_url, CHUNK_SIZE):
        await q.put.aio(chunk, partition="audio")
        await asyncio.sleep(CHUNK_SIZE / TARGET_SAMPLE_RATE)
    await q.put.aio(END_OF_STREAM, partition="audio")

async def receive_text(q):
    buffer = ""
    sentence_end = re.compile(r'[.!?]\s*$')
    while True:
        message = await q.get.aio(partition="transcription")
        if message == END_OF_STREAM:
            if buffer:
                print(buffer.strip())
            break
        buffer += " " + message.strip()
        if sentence_end.search(buffer):
            print(buffer.strip())
            buffer = ""

class NoStdStreams(object):
    def __init__(self):
        self.devnull = open(os.devnull, "w")

    def __enter__(self):
        self._stdout, self._stderr = sys.stdout, sys.stderr
        self._stdout.flush(), self._stderr.flush()
        sys.stdout, sys.stderr = self.devnull, self.devnull

    def __exit__(self, exc_type, exc_value, traceback):
        sys.stdout, sys.stderr = self._stdout, self._stderr
        self.devnull.close()
