# OPRO Classic: Prompt Optimization with Local LLM

This directory contains the **OPRO Classic** implementation - a restored version of the OPRO optimizer that achieved >90% balanced accuracy on the dev set.

## Overview

OPRO Classic combines:

1. **Local LLM for prompt generation** (e.g., Qwen2.5-7B-Instruct, Llama-3-8B)
2. **Qwen2-Audio-7B-Instruct for evaluation** (base or fine-tuned with LoRA)
3. **Composite reward function**: `R = w_clip·BA_clip + w_cond·BA_cond - w_len·(len/100)`
4. **Robust prompt sanitization** (blocks special tokens, validates format)
5. **Circuit breaker** for failed evaluations

## Key Features

### From `opro_optimizer_local.py`:
- `PromptCandidate` dataclass for tracking prompts
- `LocalLLMGenerator` for running LLMs locally (no API keys)
- `OPROClassicOptimizer` with meta-prompt construction and top-k memory

### From `run_opro_local_8gb_fixed.py`:
- Prompt sanitization (blocks `<|audio_*|>` and other special tokens)
- Strict validation (length, keywords, format)
- Circuit breaker for error handling
- Memory management for 8GB VRAM setups

### From current codebase:
- `Qwen2AudioClassifier` for generation-based evaluation
- Robust label normalization (`normalize_to_binary`)
- Support for LoRA checkpoints

## File Structure

```
scripts/
├── opro_classic_optimize.py     # Main OPRO Classic script (NEW)
├── archive_opro_legacy/         # Archived OPRO v2 scripts
│   ├── opro_post_ft.py
│   ├── opro_post_ft_v2.py
│   └── run_opro_ab.sh
└── archive_obsolete/            # Original OPRO local scripts (reference)
    ├── opro_optimizer_local.py
    └── run_opro_local_8gb_fixed.py

run_opro_classic.sh              # Convenience script for running OPRO Classic
```

## Usage

### Basic Usage (Base Model)

```bash
python scripts/opro_classic_optimize.py \
    --manifest data/processed/conditions_final/conditions_manifest_split.parquet \
    --split dev \
    --output_dir results/opro_classic_base \
    --no_lora \
    --seed 42
```

### With LoRA Fine-Tuning

```bash
python scripts/opro_classic_optimize.py \
    --manifest data/processed/conditions_final/conditions_manifest_split.parquet \
    --split dev \
    --output_dir results/opro_classic_lora \
    --checkpoint checkpoints/qwen_lora_seed42/final \
    --seed 42
```

### Using Convenience Script

```bash
# Base model
./run_opro_classic.sh base 42

# LoRA model
./run_opro_classic.sh lora 42
```

## Command-Line Arguments

### Data
- `--manifest`: Path to manifest Parquet with psychoacoustic conditions (**required**)
- `--split`: Split to use (default: `dev`)

### Output
- `--output_dir`: Output directory for results (**required**)

### Evaluator (Qwen2-Audio)
- `--evaluator_model_name`: Evaluator model (default: `Qwen/Qwen2-Audio-7B-Instruct`)
- `--evaluator_device`: Device for evaluator (default: `cuda`)
- `--no_lora`: Use base model without LoRA
- `--checkpoint`: Path to LoRA checkpoint (required if not `--no_lora`)

### Optimizer LLM
- `--optimizer_llm`: LLM for generating prompts (default: `Qwen/Qwen2.5-7B-Instruct`)
- `--optimizer_device`: Device for optimizer (default: `cuda`)
- `--optimizer_load_in_4bit`: Load optimizer in 4-bit (default: `True`)
- `--optimizer_max_new_tokens`: Max tokens to generate (default: `2000`)
- `--optimizer_temperature`: Sampling temperature (default: `0.7`)

### OPRO Configuration
- `--num_iterations`: Maximum iterations (default: `30`)
- `--candidates_per_iter`: Candidates per iteration (default: `3`)
- `--top_k`: Top-k memory size (default: `10`)
- `--early_stopping`: Early stopping patience (default: `5`)

### Reward Weights
- `--reward_w_ba_clip`: Weight for BA_clip (default: `1.0`)
- `--reward_w_ba_cond`: Weight for BA_conditions (default: `0.25`)
- `--reward_w_length_penalty`: Weight for length penalty (default: `0.05`)

### Initial Prompts
- `--baseline_prompt`: Baseline prompt (default: "Does this audio contain human speech?\nReply with ONLY one word: SPEECH or NON-SPEECH.")
- `--initial_prompts_json`: JSON file with initial prompts (optional)

### Other
- `--seed`: Random seed (default: `42`)

## Output Files

OPRO Classic saves the following files to `--output_dir`:

```
results/opro_classic_base/
├── best_prompt.txt              # Best prompt found
├── best_metrics.json            # Metrics for best prompt
├── best_prompt_summary.json     # Summary of best prompt
├── opro_prompts.jsonl           # Full history (JSONL)
├── opro_memory.json             # Top-k memory (JSON)
└── opro_history.json            # Reward progression
```

## Reward Function

The composite reward balances three objectives:

```python
R = w_clip * BA_clip + w_cond * BA_conditions - w_len * (len(prompt) / 100)
```

Where:
- **BA_clip**: Balanced accuracy at clip level (primary metric)
- **BA_conditions**: Macro-average balanced accuracy across psychoacoustic conditions
- **len(prompt)**: Character count (penalty for verbosity)

Default weights:
- `w_clip = 1.0` (primary objective)
- `w_cond = 0.25` (secondary objective: robustness across conditions)
- `w_len = 0.05` (small penalty for long prompts)

## Memory Management (8GB VRAM)

For systems with 8GB VRAM (e.g., RTX 4070 Laptop):

1. Use smaller optimizer LLM: `--optimizer_llm Qwen/Qwen2.5-3B-Instruct`
2. Enable 4-bit quantization (default)
3. Reduce candidates per iteration: `--candidates_per_iter 2`
4. The script alternates between loading optimizer and evaluator (never both in memory)

Example:
```bash
python scripts/opro_classic_optimize.py \
    --manifest data/processed/conditions_final/conditions_manifest_split.parquet \
    --split dev \
    --output_dir results/opro_classic_8gb \
    --no_lora \
    --optimizer_llm Qwen/Qwen2.5-3B-Instruct \
    --candidates_per_iter 2 \
    --num_iterations 10 \
    --seed 42
```

## Differences from OPRO v2

| Feature | OPRO Classic | OPRO v2 (archived) |
|---------|-------------|-------------------|
| **Prompt Generation** | Local LLM (Qwen2.5, Llama) | Template selection |
| **Evaluation** | `Qwen2AudioClassifier` | Same |
| **Reward** | Composite (BA_clip + BA_cond - length) | Accuracy only |
| **Sanitization** | Yes (blocks special tokens) | Limited |
| **Circuit Breaker** | Yes (error handling) | No |
| **Memory Management** | Aggressive (8GB VRAM) | Standard |
| **Meta-Prompt** | Research-backed | Basic |

## Example Workflow

### 1. Run OPRO on base model
```bash
./run_opro_classic.sh base 42
```

### 2. Inspect results
```bash
cat results/opro_classic_base_seed42/best_prompt.txt
cat results/opro_classic_base_seed42/best_metrics.json
```

### 3. Test best prompt on test set
```bash
python scripts/evaluate_with_generation.py \
    --no-lora \
    --prompt_file results/opro_classic_base_seed42/best_prompt.txt \
    --test_csv data/processed/conditions_final/conditions_manifest_split.parquet \
    --output_csv results/opro_classic_base_seed42/test_predictions.csv
```

### 4. Run OPRO on fine-tuned model
```bash
./run_opro_classic.sh lora 42
```

## Troubleshooting

### OOM (Out of Memory)
- Use smaller optimizer LLM: `Qwen/Qwen2.5-3B-Instruct` or `Qwen/Qwen2.5-1.5B-Instruct`
- Reduce `--candidates_per_iter`
- Reduce `--top_k`

### No Valid Candidates Generated
- Check optimizer LLM temperature (try `0.5` - `0.9`)
- Check `--optimizer_max_new_tokens` (increase if needed)
- Inspect raw LLM output in logs

### Low Reward
- Increase `--num_iterations`
- Try different optimizer LLMs
- Provide better `--initial_prompts_json`

### Slow Evaluation
- Reduce dev set size in manifest
- Use `--split dev` instead of `--split test`

## References

- Original OPRO paper: [Large Language Models as Optimizers](https://arxiv.org/abs/2309.03409)
- Qwen2-Audio: [https://github.com/QwenLM/Qwen2-Audio](https://github.com/QwenLM/Qwen2-Audio)
- Prompt engineering research: [PromptPapers](https://github.com/thunlp/PromptPapers)

## Migration from OPRO v2

If you were using OPRO v2 (`opro_post_ft.py` or `opro_post_ft_v2.py`), here's how to migrate:

1. **Old command:**
   ```bash
   python scripts/opro_post_ft_v2.py --no_lora --train_csv data/dev.csv --output_dir results/opro
   ```

2. **New command:**
   ```bash
   python scripts/opro_classic_optimize.py \
       --manifest data/processed/conditions_final/conditions_manifest_split.parquet \
       --split dev \
       --output_dir results/opro_classic \
       --no_lora
   ```

Key changes:
- Use `--manifest` instead of `--train_csv`
- Specify `--split` (dev/test)
- No need for `--samples_per_iter` (uses full dev set)

## License

Same as parent project (see root LICENSE file).
