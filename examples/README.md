# Examples

## 1. Running Inference

### Using Jupyter Notebook
Start with `typhoon_asr_inference.ipynb` for interactive inference:

```bash
jupyter notebook typhoon_asr_inference.ipynb
```

The notebook demonstrates:
- Loading the Typhoon ASR model
- Audio preprocessing 
- Basic transcription
- Transcription with word timestamps
- Performance metrics calculation

### Using Command Line
```bash
cd ..
python typhoon_asr_inference.py examples/cv_test.wav
```

## 2. Training Your Own Model

### Step 1: Create Manifest Files
```bash
# From text file (one transcription per line)
python create_manifest.py --audio_dir /path/to/audio --transcripts transcripts.txt --output train.jsonl

# From CSV file with filename,transcription columns
python create_manifest.py --audio_dir /path/to/audio --transcripts data.csv --output val.jsonl

# From directory of individual text files  
python create_manifest.py --audio_dir /path/to/audio --transcripts /path/to/txt_files/ --output test.jsonl
```

### Step 2: Fine-tune Model
```bash
python finetune.py \
  --model_name scb10x/typhoon-asr-realtime \
  --train_manifest train.jsonl \
  --val_manifest val.jsonl \
  --data_dir ./thai_asr_data \
  --epochs 50 \
  --batch_size 16 \
  --lr 1e-3 \
  --change_vocabulary \
  --train_modules decoder
```

## Files

- `typhoon_asr_inference.ipynb` - Interactive notebook for inference
- `cv_test.wav` - Sample Thai audio file from Mozilla Common Voice 17.0
- `create_manifest.py` - Convert audio + transcripts to NeMo manifest format
- `finetune.py` - Fine-tune ASR models for Thai language

## Acknowledgements

The sample audio file `cv_test.wav` is from the [Mozilla Common Voice 17.0](https://commonvoice.mozilla.org/) dataset, licensed under CC0.