#!/usr/bin/env python3
"""
Create NeMo manifest files from audio folder and transcription text.

Supports multiple transcription input formats:
1. Single text file with one transcription per line (matching audio file order)
2. Individual text files with same basename as audio files
3. CSV file with columns: filename, transcription
4. JSON file with filename -> transcription mapping

Usage:
    python create_manifest.py --audio_dir /path/to/audio --transcripts /path/to/transcripts.txt --output manifest.jsonl
    python create_manifest.py --audio_dir /path/to/audio --transcripts /path/to/transcripts.csv --output manifest.jsonl
    python create_manifest.py --audio_dir /path/to/audio --transcripts /path/to/transcripts/ --output manifest.jsonl
"""

import os
import json
import csv
import argparse
import librosa
from pathlib import Path
from typing import Dict, List, Union
import glob


def get_audio_duration(audio_path: str) -> float:
    """Get audio duration using librosa."""
    try:
        duration = librosa.get_duration(path=audio_path)
        return round(duration, 2)
    except Exception as e:
        print(f"Warning: Could not get duration for {audio_path}: {e}")
        return 0.0


def load_transcripts_from_txt(transcript_path: str, audio_files: List[str]) -> Dict[str, str]:
    """Load transcriptions from a single text file (one per line)."""
    transcripts = {}
    with open(transcript_path, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]
    
    if len(lines) != len(audio_files):
        raise ValueError(f"Number of transcriptions ({len(lines)}) doesn't match number of audio files ({len(audio_files)})")
    
    for audio_file, transcript in zip(audio_files, lines):
        filename = os.path.basename(audio_file)
        transcripts[filename] = transcript
    
    return transcripts


def load_transcripts_from_csv(transcript_path: str) -> Dict[str, str]:
    """Load transcriptions from CSV file with columns: filename, transcription."""
    transcripts = {}
    with open(transcript_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Support common column names
            filename = row.get('filename') or row.get('file') or row.get('audio_file')
            text = row.get('transcription') or row.get('text') or row.get('transcript')
            
            if filename and text:
                transcripts[filename] = text.strip()
    
    return transcripts


def load_transcripts_from_json(transcript_path: str) -> Dict[str, str]:
    """Load transcriptions from JSON file."""
    with open(transcript_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Handle different JSON structures
    if isinstance(data, dict):
        return {k: str(v).strip() for k, v in data.items()}
    elif isinstance(data, list):
        # Assume list of dicts with filename/text keys
        transcripts = {}
        for item in data:
            filename = item.get('filename') or item.get('file')
            text = item.get('transcription') or item.get('text')
            if filename and text:
                transcripts[filename] = str(text).strip()
        return transcripts
    else:
        raise ValueError("Unsupported JSON format")


def load_transcripts_from_dir(transcript_dir: str) -> Dict[str, str]:
    """Load transcriptions from directory of text files."""
    transcripts = {}
    transcript_files = glob.glob(os.path.join(transcript_dir, "*.txt"))
    
    for txt_file in transcript_files:
        filename = Path(txt_file).stem  # filename without extension
        with open(txt_file, 'r', encoding='utf-8') as f:
            text = f.read().strip()
            if text:
                transcripts[filename] = text
    
    return transcripts


def load_transcripts(transcript_input: str, audio_files: List[str]) -> Dict[str, str]:
    """Load transcriptions based on input type."""
    transcript_path = Path(transcript_input)
    
    if transcript_path.is_dir():
        return load_transcripts_from_dir(transcript_input)
    elif transcript_path.suffix.lower() == '.txt':
        return load_transcripts_from_txt(transcript_input, audio_files)
    elif transcript_path.suffix.lower() == '.csv':
        return load_transcripts_from_csv(transcript_input)
    elif transcript_path.suffix.lower() == '.json':
        return load_transcripts_from_json(transcript_input)
    else:
        raise ValueError(f"Unsupported transcript format: {transcript_path.suffix}")


def get_audio_files(audio_dir: str) -> List[str]:
    """Get all audio files from directory."""
    audio_extensions = ['.wav', '.mp3', '.flac', '.m4a', '.ogg', '.aac', '.webm']
    audio_files = []
    
    for ext in audio_extensions:
        pattern = os.path.join(audio_dir, f"**/*{ext}")
        audio_files.extend(glob.glob(pattern, recursive=True))
        pattern = os.path.join(audio_dir, f"**/*{ext.upper()}")
        audio_files.extend(glob.glob(pattern, recursive=True))
    
    return sorted(audio_files)


def match_transcripts_to_audio(audio_files: List[str], transcripts: Dict[str, str]) -> List[tuple]:
    """Match transcriptions to audio files."""
    matched_pairs = []
    unmatched_audio = []
    
    for audio_file in audio_files:
        filename = os.path.basename(audio_file)
        stem = Path(filename).stem  # filename without extension
        
        # Try exact filename match first, then stem match
        transcript = transcripts.get(filename) or transcripts.get(stem)
        
        if transcript:
            matched_pairs.append((audio_file, transcript))
        else:
            unmatched_audio.append(audio_file)
    
    if unmatched_audio:
        print(f"Warning: {len(unmatched_audio)} audio files have no matching transcription:")
        for file in unmatched_audio[:5]:  # Show first 5
            print(f"  - {os.path.basename(file)}")
        if len(unmatched_audio) > 5:
            print(f"  ... and {len(unmatched_audio) - 5} more")
    
    return matched_pairs


def create_manifest(audio_dir: str, transcript_input: str, output_path: str):
    """Create NeMo manifest file."""
    print(f"Scanning audio directory: {audio_dir}")
    audio_files = get_audio_files(audio_dir)
    print(f"Found {len(audio_files)} audio files")
    
    print(f"Loading transcriptions from: {transcript_input}")
    transcripts = load_transcripts(transcript_input, audio_files)
    print(f"Loaded {len(transcripts)} transcriptions")
    
    print("Matching transcriptions to audio files...")
    matched_pairs = match_transcripts_to_audio(audio_files, transcripts)
    print(f"Successfully matched {len(matched_pairs)} pairs")
    
    if not matched_pairs:
        print("Error: No audio-transcription pairs found!")
        return
    
    print(f"Creating manifest file: {output_path}")
    with open(output_path, 'w', encoding='utf-8') as f:
        for i, (audio_file, transcript) in enumerate(matched_pairs):
            print(f"Processing {i+1}/{len(matched_pairs)}: {os.path.basename(audio_file)}")
            
            duration = get_audio_duration(audio_file)
            
            manifest_entry = {
                "audio_filepath": os.path.abspath(audio_file),
                "duration": duration,
                "text": transcript,
            }
            
            f.write(json.dumps(manifest_entry, ensure_ascii=False) + '\n')
    
    print(f"✅ Manifest created successfully with {len(matched_pairs)} entries")


def main():
    parser = argparse.ArgumentParser(description="Create NeMo manifest from audio and transcriptions")
    parser.add_argument("--audio_dir", required=True, help="Directory containing audio files")
    parser.add_argument("--transcripts", required=True, help="Transcription input (txt/csv/json file or directory)")
    parser.add_argument("--output", required=True, help="Output manifest file (.jsonl)")
    
    args = parser.parse_args()
    
    # Validate inputs
    if not os.path.exists(args.audio_dir):
        print(f"Error: Audio directory not found: {args.audio_dir}")
        return
    
    if not os.path.exists(args.transcripts):
        print(f"Error: Transcripts input not found: {args.transcripts}")
        return
    
    # Create output directory if needed
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    
    try:
        create_manifest(args.audio_dir, args.transcripts, args.output)
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
