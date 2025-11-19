"""Download Qwen2-Audio model and processor to local HuggingFace cache.

This script downloads both the model and processor files to ~/.cache/huggingface/
so they can be used offline with local_files_only=True.
"""

import os
from transformers import Qwen2AudioForConditionalGeneration, AutoProcessor

# Disable hf_transfer to avoid dependency issues
os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = '0'

MODEL_NAME = "Qwen/Qwen2-Audio-7B-Instruct"

print("=" * 80)
print("DOWNLOADING QWEN2-AUDIO MODEL AND PROCESSOR")
print("=" * 80)
print(f"Model: {MODEL_NAME}")
print(f"Cache directory: ~/.cache/huggingface/")
print()

# Download processor (tokenizer, processor config, etc.)
print("Step 1/2: Downloading processor...")
print("-" * 80)
processor = AutoProcessor.from_pretrained(MODEL_NAME)
print("✓ Processor downloaded successfully")
print()

# Download model
print("Step 2/2: Downloading model (~14GB)...")
print("-" * 80)
model = Qwen2AudioForConditionalGeneration.from_pretrained(
    MODEL_NAME,
    device_map="auto",
    torch_dtype="auto",
)
print("✓ Model downloaded successfully")
print()

print("=" * 80)
print("DOWNLOAD COMPLETE")
print("=" * 80)
print("Both model and processor are now cached locally.")
print("You can now run training with local_files_only=True")
print("=" * 80)
