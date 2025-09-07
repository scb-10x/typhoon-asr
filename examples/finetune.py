# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
A comprehensive script to fine-tune a pre-trained English ASR model 
(e.g., stt_en_conformer_transducer_large) for the Thai language with W&B logging,
robust checkpointing, and resume capabilities.
"""

import os
import json
import argparse
import time
import torch
import lightning.pytorch as pl
import sentencepiece as spm
from pathlib import Path
from omegaconf import open_dict, OmegaConf
import shutil

import nemo.collections.asr as nemo_asr
from nemo.utils import logging
from nemo.utils.exp_manager import exp_manager


def create_text_corpus(manifest_path: str, output_path: str):
    """Creates a text corpus from a NeMo manifest file."""
    logging.info(f"Creating text corpus from {manifest_path}...")
    try:
        with open(manifest_path, 'r', encoding='utf-8') as manifest_file, \
             open(output_path, 'w', encoding='utf-8') as corpus_file:
            for line in manifest_file:
                try:
                    data = json.loads(line)
                    corpus_file.write(data['text'] + '\n')
                except json.JSONDecodeError:
                    logging.warning(f"Skipping invalid JSON line in manifest: {line.strip()}")
        logging.info(f"Text corpus saved to {output_path}")
    except FileNotFoundError:
        logging.error(f"Manifest file not found at: {manifest_path}")
        raise

def train_tokenizer(corpus_path: str, data_dir: str, vocab_size: int):
    """Trains a SentencePiece BPE tokenizer."""
    data_dir_path = Path(data_dir)
    final_model_path = data_dir_path / "tokenizer.model"
    if final_model_path.exists():
        logging.info("Tokenizer already exists, skipping training.")
        return
    logging.info(f"Starting tokenizer training with vocab size {vocab_size}...")
    temp_model_prefix = data_dir_path / f"tokenizer_th_bpe_v{vocab_size}"
    spm.SentencePieceTrainer.train(
        f'--input={corpus_path} --model_prefix={temp_model_prefix} '
        f'--vocab_size={vocab_size} --character_coverage=1.0 --model_type=bpe '
        f'--user_defined_symbols=<pad>,<bos>,<eos> --hard_vocab_limit=false'
    )
    os.rename(f"{temp_model_prefix}.model", final_model_path)
    os.rename(f"{temp_model_prefix}.vocab", data_dir_path / "vocab.txt")
    logging.info("Tokenizer training complete.")


def main():
    parser = argparse.ArgumentParser(description="Fine-tune NeMo ASR model for Thai.")
    # --- Standard Arguments ---
    parser.add_argument("--model_name", type=str, required=True, help="Pre-trained model from NGC or Hugging Face.")
    parser.add_argument("--train_manifest", type=str, required=True, help="Path to the training manifest file.")
    parser.add_argument("--val_manifest", type=str, help="Path to the validation manifest file.")
    parser.add_argument("--data_dir", type=str, default="./thai_asr_data", help="Directory to store tokenizer and intermediate files.")
    parser.add_argument("--vocab_size", type=int, default=2048, help="Vocabulary size for the new Thai tokenizer.")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs.")
    parser.add_argument("--checkpoint_dir", type=str, help="Directory to store checkpoints.")
    parser.add_argument("--batch_size", type=int, default=16, help="Training batch size.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument('--change_vocabulary', action='store_true', help='Change vocabulary of the model')
    parser.add_argument(
        "--train_modules",
        type=str,
        default="all",
        help=(
            "Comma-separated modules to train: encoder,decoder,joint,all. "
            "Default: decoder"
        ),
    )

    # --- Logging and Validation Control Arguments ---
    parser.add_argument("--log_every_n_steps", type=int, default=100, help="Log training metrics every N steps.")
    parser.add_argument("--val_check_interval", type=int, default=1, help="Run validation every N epochs.")

    # --- W&B Logging Arguments ---
    parser.add_argument("--wandb_project", type=str, default="nemo_asr_finetune", help="Name of the W&B project.")
    parser.add_argument("--wandb_name", type=str, default=None, help="Name of the W&B run. Defaults to a W&B-generated name.")

    # --- Argument for resuming training ---
    parser.add_argument("--resume_from_checkpoint", type=str, default=None, help="Path to a checkpoint to resume training from (e.g., '.../last.ckpt').")

    args = parser.parse_args()
    args.data_dir = os.path.abspath(args.data_dir)
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    # --- File-based synchronization for tokenizer creation ---
    data_dir = Path(args.data_dir)
    done_file = data_dir / "tokenizer.done"

    if local_rank == 0:
        if args.change_vocabulary:
            logging.info("Main process (rank 0) is setting up the tokenizer...")
            data_dir.mkdir(parents=True, exist_ok=True)
            if done_file.exists():
                os.remove(done_file)
            corpus_file_path = data_dir / "corpus.txt"
            create_text_corpus(args.train_manifest, str(corpus_file_path))
            train_tokenizer(str(corpus_file_path), args.data_dir, args.vocab_size)
            with open(done_file, "w") as f:
                f.write("done")
    else:
        logging.info(f"Process {local_rank} is waiting for the tokenizer...")
        while not done_file.exists():
            time.sleep(2)
        time.sleep(1)
        logging.info(f"Process {local_rank} detected tokenizer is ready.")
    ACCUMULATION_STEPS = 1

    trainer = pl.Trainer(
        devices=-1,
        accelerator='gpu',
        max_epochs=args.epochs,
        precision='bf16-mixed',
        strategy='ddp_find_unused_parameters_true',
        logger=False,
        enable_progress_bar=True,
        log_every_n_steps=args.log_every_n_steps,
        check_val_every_n_epoch=args.val_check_interval,
        enable_checkpointing=False,
        accumulate_grad_batches=ACCUMULATION_STEPS
    )

    run_name = args.wandb_name or f"thai-finetune-{time.strftime('%Y%m%d-%H%M%S')}"

    exp_config = {
        "exp_dir": str(Path(args.data_dir) / "experiments"),
        "name": run_name,
        "create_checkpoint_callback": True,
        "checkpoint_callback_params": {
            "filename": 'last-{epoch}',
            "every_n_epochs": 1,
            "save_top_k": 1,
            "save_last": True,
            "always_save_nemo": True,
            "monitor": "val_wer",
            "mode": "min"
        },
        "create_wandb_logger": True,
        "wandb_logger_kwargs": {
            "name": run_name,
            "project": args.wandb_project,
            "entity": "scb10x-ai"
        },
    }
    exp_manager(trainer, OmegaConf.create(exp_config))

    # --- Model Loading and Configuration ---
    logging.info(f"Process {local_rank} loading pre-trained model: {args.model_name}")
    asr_model = nemo_asr.models.EncDecRNNTBPEModel.from_pretrained(args.model_name)
    asr_model.set_trainer(trainer)

    logging.info(f"Process {local_rank} updating model tokenizer and configuration...")
    if args.change_vocabulary:
        asr_model.change_vocabulary(new_tokenizer_dir=args.data_dir, new_tokenizer_type="bpe")

    with open_dict(asr_model.cfg):
        asr_model.cfg.train_ds.manifest_filepath = args.train_manifest
        asr_model.cfg.train_ds.is_tarred = False
        asr_model.cfg.train_ds.tarred_audio_filepaths = None
        asr_model.cfg.train_ds.batch_size = args.batch_size
        asr_model.cfg.train_ds.shuffle = True 
        asr_model.cfg.train_ds.max_duration = 20

        asr_model.cfg.validation_ds.manifest_filepath = args.val_manifest
        asr_model.cfg.validation_ds.is_tarred = False
        asr_model.cfg.validation_ds.tarred_audio_filepaths = None
        asr_model.cfg.validation_ds.batch_size = 4
        asr_model.cfg.validation_ds.max_duration = 20

        asr_model.cfg.optim.name = 'adamw'
        asr_model.cfg.optim.lr = args.lr

        new_sched_config = {'name': 'CosineAnnealing', 'warmup_steps': 0, 'min_lr': 1e-4}
        asr_model.cfg.optim.sched = OmegaConf.create(new_sched_config)

    asr_model.setup_training_data(asr_model.cfg.train_ds)
    if args.val_manifest:
        asr_model.setup_validation_data(asr_model.cfg.validation_ds)
    # Configure which submodules to train via --train_modules
    selected = [m.strip().lower() for m in (args.train_modules or "").split(',') if m.strip()]
    if 'all' in selected:
        asr_model.requires_grad_(True)
        enabled = ['encoder', 'decoder', 'joint']
    else:
        asr_model.requires_grad_(False)
        enabled = []
        if 'encoder' in selected and hasattr(asr_model, 'encoder'):
            asr_model.encoder.requires_grad_(True)
            enabled.append('encoder')
        if 'decoder' in selected and hasattr(asr_model, 'decoder'):
            asr_model.decoder.requires_grad_(True)
            enabled.append('decoder')
        if 'joint' in selected and hasattr(asr_model, 'joint'):
            asr_model.joint.requires_grad_(True)
            enabled.append('joint')

    logging.info(f"Training submodules (requires_grad=True): {enabled if enabled else '[]'}")
    print(asr_model)

    # print training vs total number of parameters
    total_params = sum(p.numel() for p in asr_model.parameters())
    train_params = sum(p.numel() for p in asr_model.parameters() if p.requires_grad)
    print(f"Total number of parameters: {total_params:,}")
    print(f"Number of training parameters: {train_params:,} ({train_params / total_params:.2%})")
    asr_model.setup_optimization()


    logging.info(f"Process {local_rank} starting model fine-tuning...")
    # Pass the checkpoint path to trainer.fit to enable resuming
    trainer.fit(asr_model, ckpt_path=args.resume_from_checkpoint)

    if local_rank == 0:
        # After training is fully complete, save a final .nemo model for inference
        final_model_path = Path(trainer.log_dir) / f"{run_name}_final.nemo"
        asr_model.save_to(str(final_model_path))
        logging.info(f"Fine-tuning complete. Final model for inference saved to: {final_model_path}")
        if args.checkpoint_dir:
            os.makedirs(args.checkpoint_dir, exist_ok=True)
            shutil.copyfile(final_model_path, os.path.join(args.checkpoint_dir, "final.nemo"))
            logging.info(f"Final model copied to checkpoint directory: {args.checkpoint_dir}")
    
if __name__ == '__main__':
    main()