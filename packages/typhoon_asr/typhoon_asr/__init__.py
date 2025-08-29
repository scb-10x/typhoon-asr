import os
import sys
import time
from pathlib import Path
import nemo.collections.asr as nemo_asr
import torch
import librosa
import soundfile as sf

def transcribe(input_file, with_timestamps=False, device='auto'):
    """
    Transcribes a Thai audio file using the Typhoon ASR model.

    Args:
        input_file (str): Path to the input audio file.
        with_timestamps (bool, optional): Whether to generate estimated word timestamps. Defaults to False.
        device (str, optional): The device to run the model on ('auto', 'cpu', 'cuda'). Defaults to 'auto'.

    Returns:
        dict: A dictionary containing the transcription and, if requested, timestamps.
    """

    # --- Helper function: Prepare audio ---
    def prepare_audio(input_path, target_sr=16000):
        input_path = Path(input_path)
        if not input_path.exists():
            raise FileNotFoundError(f"File not found: {input_path}")

        supported_formats = ['.wav', '.mp3', '.m4a', '.flac', '.ogg', '.aac', '.webm']
        if input_path.suffix.lower() not in supported_formats:
            raise ValueError(f"Unsupported format: {input_path.suffix}. Supported formats are: {supported_formats}")

        processed_path = f"processed_{input_path.stem}.wav"

        y, sr = librosa.load(str(input_path), sr=None)
        if y is None:
            raise IOError("Failed to load audio file.")

        duration = len(y) / sr

        if sr != target_sr:
            y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)

        y = y / (max(abs(y)) + 1e-8)
        sf.write(processed_path, y, target_sr)
        
        return processed_path, duration

    # --- Helper function: Load model ---
    def load_typhoon_model(device):
        if device == 'auto':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        print(f"🌪️ Loading Typhoon ASR model on {device.upper()}...")
        model = nemo_asr.models.ASRModel.from_pretrained(
            model_name="scb10x/stt-fastconformer-th-large-v1.5",
            map_location=device
        )
        if model is None:
            raise RuntimeError("Failed to load the ASR model.")
        return model

    model = load_typhoon_model(device)
    processed_file, audio_duration = prepare_audio(input_file)
    
    start_time = time.time()
    
    result_data = {}

    if with_timestamps:
        print("🕐 Running transcription with timestamp estimation...")
        hypotheses = model.transcribe(audio=[processed_file], return_hypotheses=True)
        processing_time = time.time() - start_time
        
        transcription = ""
        if hypotheses and len(hypotheses) > 0 and hasattr(hypotheses[0], 'text'):
            transcription = hypotheses[0].text

        result_data['text'] = transcription
        
        timestamps = []
        if transcription and audio_duration > 0:
            words = transcription.split()
            if len(words) > 0:
                avg_duration = audio_duration / len(words)
                for i, word in enumerate(words):
                    timestamps.append({
                        'word': word,
                        'start': i * avg_duration,
                        'end': (i + 1) * avg_duration
                    })
        result_data['timestamps'] = timestamps

    else:
        print("🎙️ Running basic transcription...")
        transcriptions = model.transcribe(audio=[processed_file])
        processing_time = time.time() - start_time
        result_data['text'] = transcriptions[0] if transcriptions else ""

    if os.path.exists(processed_file):
        os.remove(processed_file)
        
    result_data['processing_time'] = processing_time
    result_data['audio_duration'] = audio_duration

    return result_data