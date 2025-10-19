"""Fine-tune Qwen2-Audio for speech detection using LoRA/QLoRA.

This script fine-tunes the model on clean audio clips to improve
speech detection accuracy, especially in noisy conditions.
"""

import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Optional
import pandas as pd
import torch
from torch.utils.data import Dataset
import soundfile as sf
import librosa
from transformers import (
    Qwen2AudioForConditionalGeneration,
    AutoProcessor,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# Add project root to path
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent
sys.path.insert(0, str(project_root))


@dataclass
class TrainingConfig:
    """Configuration for fine-tuning."""

    # Model
    model_name: str = "Qwen/Qwen2-Audio-7B-Instruct"
    use_4bit: bool = True

    # LoRA
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: list = None  # Will default to attention + MLP

    # Training
    num_epochs: int = 3
    batch_size: int = 2  # Reduced from 4 to avoid OOM
    gradient_accumulation_steps: int = 8  # Increased to maintain effective batch size of 16
    learning_rate: float = 2e-4
    warmup_steps: int = 100
    seed: int = 42  # Random seed for reproducibility

    # Paths
    train_csv: Path = project_root / "data" / "processed" / "normalized_clips" / "train_metadata.csv"
    test_csv: Path = project_root / "data" / "processed" / "normalized_clips" / "test_metadata.csv"
    output_dir: Path = project_root / "checkpoints" / "qwen2_audio_speech_detection_normalized"

    # Prompt
    system_prompt: str = "You classify audio content."
    user_prompt: str = (
        "Choose one:\n"
        "A) SPEECH (human voice)\n"
        "B) NONSPEECH (music/noise/silence/animals)\n\n"
        "Answer with A or B ONLY."
    )


class SpeechDetectionDataset(Dataset):
    """Dataset for speech detection fine-tuning."""

    def __init__(
        self,
        metadata_csv: Path,
        processor,
        system_prompt: str,
        user_prompt: str,
        max_audio_length: int = 30,  # seconds
    ):
        self.df = pd.read_csv(metadata_csv)
        self.processor = processor
        self.system_prompt = system_prompt
        self.user_prompt = user_prompt
        self.max_audio_length = max_audio_length

        print(f"  Loaded {len(self.df)} samples from {metadata_csv.name}")
        print(f"    SPEECH:    {(self.df['ground_truth'] == 'SPEECH').sum()}")
        print(f"    NONSPEECH: {(self.df['ground_truth'] == 'NONSPEECH').sum()}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Load audio
        audio_path = project_root / row['audio_path']
        audio, sr = sf.read(audio_path)

        # Resample if needed
        target_sr = self.processor.feature_extractor.sampling_rate
        if sr != target_sr:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
            sr = target_sr

        # Truncate if too long
        max_samples = int(self.max_audio_length * target_sr)
        if len(audio) > max_samples:
            audio = audio[:max_samples]

        # Ensure audio is float32 and 1D
        if audio.ndim > 1:
            audio = audio.mean(axis=1)  # Convert stereo to mono
        audio = audio.astype('float32')

        # Prepare conversation
        conversation = [
            {"role": "system", "content": self.system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "audio"},
                    {"type": "text", "text": self.user_prompt},
                ],
            },
            {
                "role": "assistant",
                "content": "A" if row['ground_truth'] == 'SPEECH' else "B",
            },
        ]

        # Get answer token for finding the offset
        answer_text = "A" if row['ground_truth'] == 'SPEECH' else "B"

        # Apply chat template
        text = self.processor.apply_chat_template(
            conversation, add_generation_prompt=False, tokenize=False
        )

        # Process inputs
        # IMPORTANT: Pass sampling_rate explicitly to avoid silent errors
        inputs = self.processor(
            text=text,
            audio=[audio],
            sampling_rate=target_sr,  # Must pass explicitly for WhisperFeatureExtractor
            return_tensors="pt",
            padding=True,
        )

        # Remove batch dimension
        inputs = {k: v.squeeze(0) for k, v in inputs.items()}

        # Create labels with masking: only compute loss on the assistant's response token
        # This is much more efficient than computing loss on the entire prompt
        labels = inputs["input_ids"].clone()

        # Find the position of the assistant's response
        # Strategy: Find where the answer token appears near the end
        answer_token_ids = self.processor.tokenizer.encode(answer_text, add_special_tokens=False)

        # Search from the end backwards for the answer token
        input_ids_list = labels.tolist()
        assistant_response_start = -1

        # Look for the answer token in the last 20 tokens (should be at the very end)
        for i in range(len(input_ids_list) - 1, max(len(input_ids_list) - 20, -1), -1):
            if len(answer_token_ids) == 1 and input_ids_list[i] == answer_token_ids[0]:
                assistant_response_start = i
                break
            elif len(answer_token_ids) > 1 and i + len(answer_token_ids) <= len(input_ids_list):
                if input_ids_list[i:i+len(answer_token_ids)] == answer_token_ids:
                    assistant_response_start = i
                    break

        # Mask everything before the assistant's response with -100
        if assistant_response_start > 0:
            labels[:assistant_response_start] = -100

        inputs["labels"] = labels

        return inputs


def collate_fn(batch):
    """Collate function for DataLoader."""
    # Find max lengths
    max_input_len = max(item["input_ids"].shape[0] for item in batch)

    # For audio features, they should already be padded to 3000 by the processor
    # We just need to ensure they're all the same shape
    max_audio_len = 3000  # Fixed size for Qwen2-Audio

    # Check if all items have input_features
    has_audio = all("input_features" in item for item in batch)

    # Pad each item
    padded_batch = {
        "input_ids": [],
        "attention_mask": [],
        "labels": [],
    }

    if has_audio:
        padded_batch["input_features"] = []
        padded_batch["feature_attention_mask"] = []

    for item in batch:
        # Pad text tokens
        input_len = item["input_ids"].shape[0]
        pad_len = max_input_len - input_len

        padded_batch["input_ids"].append(
            torch.nn.functional.pad(item["input_ids"], (0, pad_len), value=0)
        )
        padded_batch["attention_mask"].append(
            torch.nn.functional.pad(item["attention_mask"], (0, pad_len), value=0)
        )
        padded_batch["labels"].append(
            torch.nn.functional.pad(item["labels"], (0, pad_len), value=-100)
        )

        # Handle audio features if present
        if has_audio:
            # Audio features should be [128, 3000]
            audio_features = item["input_features"]
            feature_mask = item["feature_attention_mask"]

            # Pad if needed (should already be 3000 but just in case)
            if audio_features.shape[-1] < max_audio_len:
                pad_len = max_audio_len - audio_features.shape[-1]
                audio_features = torch.nn.functional.pad(audio_features, (0, pad_len), value=0.0)
                feature_mask = torch.nn.functional.pad(feature_mask, (0, pad_len), value=0)

            padded_batch["input_features"].append(audio_features)
            padded_batch["feature_attention_mask"].append(feature_mask)

    # Stack
    return {k: torch.stack(v) for k, v in padded_batch.items()}


def main():
    import argparse
    import random
    import numpy as np

    parser = argparse.ArgumentParser(description="Fine-tune Qwen2-Audio")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output_dir", type=str, help="Custom output directory")
    parser.add_argument("--lora_r", type=int, default=16, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha")
    parser.add_argument("--add_mlp_targets", action="store_true", help="Add MLP layers to LoRA targets")
    args = parser.parse_args()

    config = TrainingConfig()
    config.seed = args.seed
    if args.output_dir:
        config.output_dir = Path(args.output_dir)
    config.lora_r = args.lora_r
    config.lora_alpha = args.lora_alpha

    # Set seeds for reproducibility
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    random.seed(config.seed)
    np.random.seed(config.seed)

    print("=" * 80)
    print("QWEN2-AUDIO FINE-TUNING FOR SPEECH DETECTION")
    print("=" * 80)

    # Check if clean dataset exists
    if not config.train_csv.exists():
        print(f"\n❌ Training data not found: {config.train_csv}")
        print("   Run 'python scripts/create_clean_dataset.py' first!")
        return

    print(f"\nConfiguration:")
    print(f"  Model: {config.model_name}")
    print(f"  4-bit quantization: {config.use_4bit}")
    print(f"  LoRA r={config.lora_r}, alpha={config.lora_alpha}")
    print(f"  Epochs: {config.num_epochs}")
    print(f"  Batch size: {config.batch_size}")
    print(f"  Learning rate: {config.learning_rate}")
    print(f"  Random seed: {config.seed}")

    # Load processor
    print(f"\nLoading processor...")
    processor = AutoProcessor.from_pretrained(config.model_name)

    # Load datasets
    print(f"\nLoading datasets...")
    train_dataset = SpeechDetectionDataset(
        config.train_csv,
        processor,
        config.system_prompt,
        config.user_prompt,
    )

    eval_dataset = SpeechDetectionDataset(
        config.test_csv,
        processor,
        config.system_prompt,
        config.user_prompt,
    )

    # Load model with quantization
    print(f"\nLoading model...")
    if config.use_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
    else:
        bnb_config = None

    model = Qwen2AudioForConditionalGeneration.from_pretrained(
        config.model_name,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float16 if not config.use_4bit else None,
    )

    # Prepare model for training
    print(f"\nPreparing model for k-bit training...")
    model = prepare_model_for_kbit_training(model)

    # Enable gradient checkpointing with use_reentrant=False (recommended by PyTorch 2.4+)
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )

    # Apply LoRA
    print(f"\nApplying LoRA...")

    # Configure target modules
    target_modules = ["q_proj", "v_proj", "k_proj", "o_proj"]
    if args.add_mlp_targets:
        # Add MLP layers for increased capacity
        target_modules.extend(["gate_proj", "up_proj", "down_proj"])
        print(f"  Adding MLP targets: gate_proj, up_proj, down_proj")

    print(f"  Target modules: {target_modules}")

    lora_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        target_modules=target_modules,
        lora_dropout=config.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Training arguments
    config.output_dir.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=str(config.output_dir),
        num_train_epochs=config.num_epochs,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        warmup_steps=config.warmup_steps,
        learning_rate=config.learning_rate,
        fp16=True,
        logging_steps=10,
        save_steps=200,
        eval_strategy="steps",  # Changed from evaluation_strategy
        eval_steps=200,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="none",  # Disable wandb/tensorboard
        save_safetensors=True,  # Use safetensors format (more efficient)
        save_only_model=True,  # Don't save optimizer/scheduler (saves disk space)
    )

    # Trainer
    print(f"\nInitializing trainer...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collate_fn,
    )

    # Train
    print(f"\n" + "=" * 80)
    print("STARTING TRAINING")
    print("=" * 80)

    trainer.train()

    # Save final model
    print(f"\nSaving final model...")
    final_dir = config.output_dir / "final"
    trainer.save_model(str(final_dir))

    print(f"\n✓ Fine-tuning complete!")
    print(f"  Model saved to: {final_dir}")
    print("\nNext steps:")
    print(f"  1. Test the model: python scripts/test_finetuned_model.py")
    print(f"  2. Evaluate on full test set")
    print("=" * 80)


if __name__ == "__main__":
    main()
