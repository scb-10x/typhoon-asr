# Typhoon ASR

A user-friendly Python package for Thai speech recognition using the Typhoon ASR model.

## Installation

```bash
pip install typhoon-asr
```

## Quick Start

### Python API

```python
from typhoon_asr import transcribe

# Basic transcription
result = transcribe("audio.wav")
print(result['text'])

# With word timestamps (estimated)
result = transcribe("audio.wav", with_timestamps=True)
print(result['text'])
for ts in result['timestamps']:
    print(f"[{ts['start']:.2f}s - {ts['end']:.2f}s] {ts['word']}")

# Specify device
result = transcribe("audio.wav", device="cuda")  # or "cpu", "auto"
```

### Command Line

```bash
# Basic usage
typhoon-asr audio.wav

# With timestamps
typhoon-asr audio.wav --with-timestamps

# Specify device
typhoon-asr audio.wav --device cuda
```

## Supported Formats

- `.wav`, `.mp3`, `.m4a`, `.flac`, `.ogg`, `.aac`, `.webm`

## API Reference

### `transcribe(input_file, model_name="scb10x/typhoon-asr-realtime", with_timestamps=False, device="auto")`

**Parameters:**
- `input_file` (str): Path to audio file
- `model_name` (str): HuggingFace model identifier
- `with_timestamps` (bool): Generate estimated word timestamps
- `device` (str): Processing device ("auto", "cpu", "cuda")

**Returns:**
- `dict` with keys:
  - `text`: Transcribed text
  - `timestamps`: List of word timestamps (if enabled)
  - `processing_time`: Processing duration in seconds
  - `audio_duration`: Audio duration in seconds

## Requirements

- Python ≥ 3.8
- CUDA (optional, for GPU acceleration)

## License

Apache Software License 2.0
