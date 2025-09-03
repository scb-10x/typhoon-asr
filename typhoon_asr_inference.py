#!/usr/bin/env python3
"""
Typhoon ASR Real-Time Inference Script

A simple script for running Thai speech recognition using Typhoon ASR Real-Time model.
Supports both basic transcription and detailed output with estimated timestamps.

Usage:
    python typhoon_asr_inference.py input_audio.m4a
    python typhoon_asr_inference.py input_audio.wav --with-timestamps
    python typhoon_asr_inference.py input_audio.mp3 --device cuda
"""

import os
import sys
import time
import argparse
from pathlib import Path
import nemo.collections.asr as nemo_asr
import torch
import librosa
import soundfile as sf



def prepare_audio(input_path, output_path=None, target_sr=16000):
    """
    Prepare audio file for Typhoon ASR Real-Time processing

    Args:
        input_path (str): Source audio file path
        output_path (str): Processed output path (auto-generated if None)
        target_sr (int): Target sample rate for the model

    Returns:
        tuple: (success: bool, output_path: str, info: dict)
    """
    input_path = Path(input_path)

    if not input_path.exists():
        return False, None, {"error": f"File not found: {input_path}"}

    supported_formats = ['.wav', '.mp3', '.m4a', '.flac', '.ogg', '.aac', '.webm']
    if input_path.suffix.lower() not in supported_formats:
        return False, None, {
            "error": f"Unsupported format: {input_path.suffix}",
            "supported": supported_formats
        }

    if output_path is None:
        output_path = f"processed_{input_path.stem}.wav"

    print(f"🎵 Processing audio: {input_path.name}")

    # Load and resample audio
    y, sr = librosa.load(str(input_path), sr=None)
    if y is None:
        return False, None, {"error": "Failed to load audio file."}

    duration = len(y) / sr
    print(f"   Original: {sr} Hz, {duration:.1f}s")

    if sr != target_sr:
        y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
        print(f"   Resampled: {sr} Hz → {target_sr} Hz")

    # Normalize audio
    y = y / (max(abs(y)) + 1e-8)

    # Save processed audio
    sf.write(output_path, y, target_sr)
    if not os.path.exists(output_path):
        return False, None, {"error": f"Failed to write processed audio to {output_path}"}
    
    print(f"✅ Processed: {output_path}")

    return True, output_path, {
        "original_sr": sr,
        "target_sr": target_sr,
        "duration": duration,
        "output_path": output_path
    }


def load_typhoon_model(device='auto'):
    """
    Load Typhoon ASR Real-Time model

    Args:
        device (str): Device to use ('auto', 'cpu', 'cuda')

    Returns:
        ASR model object
    """
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(f"🌪️ Loading Typhoon ASR Real-Time model...")
    print(f"   Device: {device.upper()}")

    model = nemo_asr.models.ASRModel.from_pretrained(
        model_name="scb10x/typhoon-asr-realtime",
        map_location=device
    )

    if model is None:
        print(f"❌ Failed to load model.")
        return None
    return model


def basic_transcription(model, audio_file):
    """
    Run basic transcription without timestamps

    Args:
        model: ASR model
        audio_file (str): Path to audio file

    Returns:
        tuple: (transcription: str, processing_time: float)
    """
    print("🎙️ Running basic transcription...")

    start_time = time.time()
    transcriptions = model.transcribe(audio=[audio_file])
    processing_time = time.time() - start_time

    transcription = transcriptions[0] if transcriptions else ""
    return transcription, processing_time


def transcription_with_timestamps(model, audio_file):
    """
    Run transcription with estimated word timestamps

    Args:
        model: ASR model
        audio_file (str): Path to audio file

    Returns:
        tuple: (transcription: str, timestamps: list, processing_time: float)
    """
    print("🕐 Running transcription with timestamp estimation...")

    # Get audio duration
    audio_duration = 0
    if os.path.exists(audio_file):
        audio_info = sf.info(audio_file)
        audio_duration = audio_info.duration

    start_time = time.time()

    # Perform transcription
    result = model.transcribe(audio=[audio_file], return_hypotheses=True)
    
    transcription = ""
    if result and len(result) > 0:
        hypothesis = result[0]
        if hasattr(hypothesis, 'text'):
            transcription = hypothesis.text
        elif isinstance(hypothesis, list) and len(hypothesis) > 0:
            transcription = hypothesis[0].text if hasattr(hypothesis[0], 'text') else str(hypothesis[0])
        else:
            # Fallback for unexpected structure, defaulting to basic transcription
            basic_result = model.transcribe(audio=[audio_file])
            if basic_result:
                transcription = basic_result[0]

    processing_time = time.time() - start_time

    # Generate estimated timestamps
    timestamps = []
    if transcription and audio_duration > 0:
        words = transcription.split()
        if len(words) > 0:
            avg_duration = audio_duration / len(words)

            for i, word in enumerate(words):
                start_t = i * avg_duration
                end_t = start_t + avg_duration
                timestamps.append({
                    'word': word,
                    'start': start_t,
                    'end': end_t
                })

    return transcription, timestamps, processing_time


def main():
    parser = argparse.ArgumentParser(description="Typhoon ASR Real-Time Inference")
    parser.add_argument("input_file", help="Input audio file path")
    parser.add_argument("--with-timestamps", action="store_true",
                       help="Generate estimated word timestamps")
    parser.add_argument("--device", choices=['auto', 'cpu', 'cuda'], default='auto',
                       help="Processing device (default: auto)")

    args = parser.parse_args()

    print("🌪️ Typhoon ASR Real-Time Inference")
    print("=" * 50)

    # Check input file
    if not os.path.exists(args.input_file):
        print(f"❌ Input file not found: {args.input_file}")
        return 1

    # Load model
    model = load_typhoon_model(args.device)
    if model is None:
        return 1

    # Prepare audio
    success, processed_file, info = prepare_audio(
        args.input_file,
        target_sr=16000
    )

    if not success:
        print(f"❌ Audio processing failed: {info.get('error', 'Unknown error')}")
        return 1

    # Calculate performance metrics
    audio_duration = info['duration']

    # Run inference
    if args.with_timestamps:
        transcription, timestamps, processing_time = transcription_with_timestamps(model, processed_file)
        mode = "with timestamps"
    else:
        transcription, processing_time = basic_transcription(model, processed_file)
        timestamps = None
        mode = "basic"

    # Calculate performance
    rtf = processing_time / audio_duration if audio_duration > 0 else 0

    # Display results
    print("\n" + "=" * 50)
    print("📝 TRANSCRIPTION RESULTS")
    print("=" * 50)
    print(f"Mode: {mode}")
    print(f"File: {Path(args.input_file).name}")
    print(f"Duration: {audio_duration:.1f}s")
    print(f"Processing: {processing_time:.2f}s")
    print(f"RTF: {rtf:.3f}x", end="")

    if rtf < 1.0:
        print(" 🚀 (Real-time capable!)")
    else:
        print(" ✅ (Batch processing)")

    print(f"\nTranscription:")
    print(f"'{transcription.text}'")

    # Show timestamps if available
    if timestamps:
        print(f"\n🕐 Word Timestamps (estimated):")
        print("-" * 45)
        for i, ts in enumerate(timestamps[:10], 1):  # Show first 10 words
            print(f"{i:2d}. [{ts['start']:6.2f}s - {ts['end']:6.2f}s] {ts['word']}")

        if len(timestamps) > 10:
            print(f"... and {len(timestamps) - 10} more words")


    # Cleanup processed file if it's temporary
    if processed_file.startswith("processed_") and os.path.exists(processed_file):
        os.remove(processed_file)
        print(f"🧹 Cleaned up temporary file: {processed_file}")

    print("\n✅ Processing complete!")
    return 0


if __name__ == "__main__":
    # The initial try/except for imports handles missing libraries before this point.
    # We can proceed with the main execution.
    sys.exit(main())