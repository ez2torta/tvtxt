import subprocess

TARGET_SAMPLE_RATE = 16_000
CHUNK_SIZE = 16_000  # 1 second of audio at 16kHz, mono, 16-bit PCM

def stream_audio_from_m3u8(m3u8_url, chunk_size=CHUNK_SIZE):
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

if __name__ == "__main__":
    # Ask the user for the m3u8 URL or file path
    m3u8_url = input("Enter the .m3u8 URL or file path: ").strip()
    print(f"\nReading audio stream from: {m3u8_url}\n")
    for i, chunk in enumerate(stream_audio_from_m3u8(m3u8_url)):
        print(f"Chunk {i+1}: size={len(chunk)} bytes, first 10 bytes={chunk[:10]}")
    print("\nStream ended.")
