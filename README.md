# Typhoon ASR Real-Time

Typhoon ASR Real-Time is a next-generation, open-source Automatic Speech Recognition (ASR) model built for real-world streaming applications in the Thai language. It delivers fast and accurate transcriptions while running efficiently on standard CPUs, enabling anyone to host their own ASR service without expensive hardware or sending sensitive data to third-party clouds.

This repository provides a simple command-line script to demonstrate the performance and features of the Typhoon ASR Real-Time model.

See the blog for more detail: [https://opentyphoon.ai/blog/th/typhoon-asr-real-time-release]()

## Quick Start with Google Colab
For a hands-on demonstration without any local setup, you can run this project directly in Google Colab. The notebook provides a complete environment to transcribe audio files and experiment with the model.

![[alt text](https://colab.research.google.com/assets/colab-badge.svg)(https://colab.research.google.com/drive/1t4tlRTJToYRolTmiN5ZWDR67ymdRnpAz?usp=sharing)]

## Features

*   **Simple Command-Line Interface**: Transcribe Thai audio files directly from your terminal.
*   **Multiple Audio Formats**: Supports a wide range of audio inputs, including `.wav`, `.mp3`, `.m4a`, `.flac`, and more.
*   **Estimated Timestamps**: Generate word-level timestamps for your transcriptions.
*   **Hardware Flexible**: Run inference on either a CPU or a CUDA-enabled GPU.
*   **Streaming Architecture**: Based on a state-of-the-art FastConformer model designed for low-latency, real-time applications.
*   **Language**: Thai

## Requirements

*   Linux / Mac (Windows is not officially supported at the moment)
*   Python 3.10

## Install

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/scb10x/typhoon-asr.git
    cd typhoon-asr
    ```

2.  **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Usage

Use the `typhoon_asr_inference.py` script to transcribe an audio file. The script will automatically handle audio resampling and processing.

**Basic Transcription (CPU):**
```bash
python typhoon_asr_inference.py path/to/your_audio.m4a
```

**Transcription with Estimated Timestamps:**
```bash
python typhoon_asr_inference.py path/to/your_audio.wav --with-timestamps
```

**Transcription on a GPU:**
```bash
python typhoon_asr_inference.py path/to/your_audio.mp3 --device cuda
```

### Arguments

*   `input_file`: (Required) The path to your input audio file.
*   `--with-timestamps`: (Optional) Flag to generate and display estimated word timestamps.
*   `--device`: (Optional) The device to run inference on. Choices: `auto`, `cpu`, `cuda`. Defaults to `auto`.

### Example Output

```
$ python typhoon_asr_inference.py audio/sample_th.wav --with-timestamps

🌪️ Typhoon ASR Real-Time Inference
==================================================
🎵 Processing audio: sample_th.wav
   Original: 48000 Hz, 4.5s
   Resampled: 48000 Hz → 16000 Hz
✅ Processed: processed_sample_th.wav
🌪️ Loading Typhoon ASR Real-Time model...
   Device: CPU
🕐 Running transcription with timestamp estimation...

==================================================
📝 TRANSCRIPTION RESULTS
==================================================
Mode: with timestamps
File: sample_th.wav
Duration: 4.5s
Processing: 1.32s
RTF: 0.293x 🚀 (Real-time capable!)

Transcription:
'ทดสอบการแปลงเสียงเป็นข้อความภาษาไทยแบบเรียลไทม์'

🕐 Word Timestamps (estimated):
---------------------------------------------
 1. [  0.00s -   0.56s] ทดสอบการแปลงเสียงเป็นข้อความภาษาไทยแบบเรียลไทม์

🧹 Cleaned up temporary file: processed_sample_th.wav

✅ Processing complete!
```

## Dependencies

*   [NVIDIA NeMo Toolkit](https://github.com/NVIDIA/NeMo) (`nemo_toolkit[asr]`)
*   [PyTorch](https://pytorch.org/) (`torch`)
*   [Librosa](https://librosa.org/) (`librosa`)
*   [SoundFile](https://pysoundfile.readthedocs.io/) (`soundfile`)

## License

This project is licensed under the Apache 2.0 License. See individual datasets and checkpoints for their respective licenses.
