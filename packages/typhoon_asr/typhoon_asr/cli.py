import argparse
from pathlib import Path
from typhoon_asr import transcribe

def main():
    parser = argparse.ArgumentParser(description="Typhoon ASR Real-Time Inference")
    parser.add_argument("input_file", help="Input audio file path")
    parser.add_argument("--with-timestamps", action="store_true", help="Generate estimated word timestamps")
    parser.add_argument("--device", choices=['auto', 'cpu', 'cuda'], default='auto', help="Processing device (default: auto)")

    args = parser.parse_args()

    try:
        result = transcribe(args.input_file, args.with_timestamps, args.device)

        rtf = result['processing_time'] / result['audio_duration'] if result['audio_duration'] > 0 else 0
        
        print("\n" + "=" * 50)
        print("📝 TRANSCRIPTION RESULTS")
        print("=" * 50)
        print(f"File: {Path(args.input_file).name}")
        print(f"Duration: {result['audio_duration']:.1f}s")
        print(f"Processing: {result['processing_time']:.2f}s")
        print(f"RTF: {rtf:.3f}x")
        
        print(f"\nTranscription:")
        print(f"'{result['text']}'")

        if 'timestamps' in result and result['timestamps']:
            print(f"\n🕐 Word Timestamps (estimated):")
            print("-" * 45)
            for ts in result['timestamps']:
                print(f"[{ts['start']:6.2f}s - {ts['end']:6.2f}s] {ts['word']}")

    except Exception as e:
        print(f"❌ An error occurred: {e}")
        return 1
    
    print("\n✅ Processing complete!")
    return 0

if __name__ == "__main__":
    main()