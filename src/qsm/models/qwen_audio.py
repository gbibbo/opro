"""Qwen2-Audio wrapper for binary SPEECH/NONSPEECH classification."""

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import librosa
import numpy as np
import soundfile as sf
import torch
from transformers import AutoProcessor, LogitsProcessor, Qwen2AudioForConditionalGeneration, Qwen2AudioProcessor


@dataclass
class PredictionResult:
    """Result from a single prediction."""

    label: Literal["SPEECH", "NONSPEECH", "UNKNOWN"]
    confidence: float
    raw_output: str
    latency_ms: float


class ConstrainedVocabLogitsProcessor(LogitsProcessor):
    """
    Logits processor that constrains decoding to only allowed tokens.

    This forces the model to only output specific words (e.g., SPEECH, NONSPEECH)
    by masking all other tokens with -inf.
    """

    def __init__(self, allowed_token_ids: list[int]):
        """
        Args:
            allowed_token_ids: List of token IDs that are allowed to be generated
        """
        self.allowed_token_ids = set(allowed_token_ids)

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        """
        Mask all tokens except the allowed ones.

        Args:
            input_ids: Input token IDs (batch_size, sequence_length)
            scores: Logits for next token (batch_size, vocab_size)

        Returns:
            Modified scores with disallowed tokens masked to -inf
        """
        # Create mask: True for allowed tokens, False for others
        mask = torch.ones_like(scores, dtype=torch.bool)
        for token_id in self.allowed_token_ids:
            mask[:, token_id] = False

        # Mask disallowed tokens with -inf
        scores = scores.masked_fill(mask, float("-inf"))

        return scores


class Qwen2AudioClassifier:
    """
    Qwen2-Audio classifier for binary speech detection.

    Wraps Qwen2-Audio-7B-Instruct for SPEECH/NONSPEECH classification
    with customizable prompts and automatic response parsing.
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2-Audio-7B-Instruct",
        device: str = "cuda",
        torch_dtype: str = "auto",
        load_in_4bit: bool = False,
        load_in_8bit: bool = False,
        auto_pad: bool = False,  # DISABLED: Processor handles padding with padding=True
        pad_target_ms: int = 2000,  # Kept for backward compatibility
        pad_noise_amplitude: float = 0.0001,  # Kept for backward compatibility
        constrained_decoding: bool = False,  # NEW: Force only SPEECH/NONSPEECH tokens
    ):
        """
        Initialize Qwen2-Audio classifier.

        Args:
            model_name: HuggingFace model identifier
            device: Device to run inference on ("cuda" or "cpu")
            torch_dtype: Model precision ("auto", "float16", or "float32")
            load_in_4bit: Use 4-bit quantization (requires bitsandbytes)
            load_in_8bit: Use 8-bit quantization (requires bitsandbytes)
            auto_pad: Automatically pad short segments with low-amplitude noise (default: True)
            pad_target_ms: Target duration for padding in milliseconds (default: 2000)
            pad_noise_amplitude: Amplitude of padding noise (default: 0.0001)
        """
        self.model_name = model_name
        self.device = device
        self.name = f"qwen2_audio_{model_name.split('/')[-1].lower()}"

        # Padding configuration
        self.auto_pad = auto_pad
        self.pad_target_ms = pad_target_ms
        self.pad_noise_amplitude = pad_noise_amplitude

        # Constrained decoding
        self.constrained_decoding = constrained_decoding

        # Map dtype string to torch dtype
        dtype_map = {
            "auto": "auto",
            "float16": torch.float16,
            "float32": torch.float32,
        }
        self.torch_dtype = dtype_map.get(torch_dtype, "auto")

        print(f"Loading {model_name}...")
        print(f"  Device: {device}")
        print(f"  Dtype: {torch_dtype}")
        if load_in_4bit:
            print("  Quantization: 4-bit")
        elif load_in_8bit:
            print("  Quantization: 8-bit")

        # Load processor and model
        # Use Qwen2AudioProcessor directly to ensure correct audio handling
        self.processor = Qwen2AudioProcessor.from_pretrained(model_name)

        # Prepare model loading kwargs
        model_kwargs = {
            "torch_dtype": self.torch_dtype,
        }

        if load_in_4bit or load_in_8bit:
            # Use quantization
            from transformers import BitsAndBytesConfig

            quantization_config = BitsAndBytesConfig(
                load_in_4bit=load_in_4bit,
                load_in_8bit=load_in_8bit,
                bnb_4bit_compute_dtype=torch.float16 if load_in_4bit else None,
                llm_int8_enable_fp32_cpu_offload=True,  # Enable CPU offloading for 8GB VRAM
            )
            model_kwargs["quantization_config"] = quantization_config
            model_kwargs["device_map"] = "auto"
        else:
            # Standard loading
            model_kwargs["device_map"] = device if device == "cuda" else None

        self.model = Qwen2AudioForConditionalGeneration.from_pretrained(model_name, **model_kwargs)

        if device == "cpu" and not (load_in_4bit or load_in_8bit):
            self.model = self.model.to(device)

        self.model.eval()

        # Default prompt (A/B binary format - optimized for gate evaluation)
        self.system_prompt = "You classify audio content."

        self.user_prompt = (
            "Choose one:\n"
            "A) SPEECH (human voice)\n"
            "B) NONSPEECH (music/noise/silence/animals)\n\n"
            "Answer with A or B ONLY."
        )

        print("Model loaded successfully!")
        if self.auto_pad:
            print(
                f"Auto-padding enabled: <{self.pad_target_ms}ms -> {self.pad_target_ms}ms (noise amplitude: {self.pad_noise_amplitude})"
            )

        # Initialize constrained decoding if enabled
        self.logits_processor = None
        self.prefix_allowed_tokens_fn = None

        if self.constrained_decoding:
            # For A/B format: get ALL single-token variants for "A" and "B"
            # (with/without leading space, newline, etc.)
            tokenizer = self.processor.tokenizer

            # Helper function to get all single-token variants
            def get_single_token_variants(char: str):
                """Get all single-token IDs that decode to variants of the character."""
                variants = [char, f" {char}", f"\n{char}"]
                valid_ids = []
                for variant in variants:
                    ids = tokenizer.encode(variant, add_special_tokens=False)
                    if len(ids) == 1:
                        # Verify it decodes back to something containing the char
                        decoded = tokenizer.decode([ids[0]])
                        if char in decoded.upper():
                            valid_ids.append(ids[0])
                return list(set(valid_ids))  # Remove duplicates

            # Get all valid token IDs for A and B
            ids_A = get_single_token_variants("A")
            ids_B = get_single_token_variants("B")

            # Keep first ID for display
            self.id_A = ids_A[0] if ids_A else None
            self.id_B = ids_B[0] if ids_B else None

            # Get EOS token ID from tokenizer (not hardcoded)
            self.id_eos = tokenizer.eos_token_id
            if self.id_eos is None:
                # Fallback: try to get from model generation config
                self.id_eos = getattr(self.model.generation_config, "eos_token_id", None)

            # Validate that we have all required IDs
            if not ids_A or not ids_B:
                raise ValueError(
                    f"Could not find single tokens for A/B. "
                    f"A: {ids_A}, B: {ids_B}"
                )
            if self.id_eos is None:
                raise ValueError("Could not find EOS token ID")

            # Store ALL variants for use in prefix_allowed_tokens_fn (no bias)
            self.first_step_allowed = ids_A + ids_B

            # Create prefix_allowed_tokens_fn
            def make_prefix_fn(first_allowed, eos_id):
                def prefix_fn(batch_id, input_ids):
                    """Allow only A or B on first step, then only EOS."""
                    # input_ids includes the prompt; we track generation steps
                    # by comparing current length to initial length
                    # This function is called before generating each token
                    # For max_new_tokens=1, we only need first_allowed
                    return first_allowed
                return prefix_fn

            # Will be applied during generate() with the actual input length
            self.prefix_allowed_tokens_fn_maker = make_prefix_fn

            # Debug: print token mappings
            print(f"Constrained decoding enabled (A/B format):")
            print(f"  Tokens for 'A': {ids_A} -> {[repr(tokenizer.decode([tid])) for tid in ids_A]}")
            print(f"  Tokens for 'B': {ids_B} -> {[repr(tokenizer.decode([tid])) for tid in ids_B]}")
            print(f"  EOS token: {self.id_eos}")
            print(f"  Total allowed on first step: {len(self.first_step_allowed)} tokens")

    def _pad_audio_with_noise(
        self,
        audio: np.ndarray,
        sample_rate: int = 16000,
    ) -> np.ndarray:
        """
        Pad audio with low-amplitude noise to reach target duration.

        Places original audio in the center, surrounded by white noise.

        Args:
            audio: Audio array to pad
            sample_rate: Sample rate in Hz (default: 16000)

        Returns:
            Padded audio array
        """
        target_samples = int(self.pad_target_ms * sample_rate / 1000)
        current_samples = len(audio)

        if current_samples >= target_samples:
            # Already at or above target duration, no padding needed
            return audio

        # Calculate padding needed on each side
        total_padding = target_samples - current_samples
        padding_left = total_padding // 2
        padding_right = total_padding - padding_left

        # Generate low-amplitude white noise
        noise_left = np.random.randn(padding_left).astype(np.float32) * self.pad_noise_amplitude
        noise_right = np.random.randn(padding_right).astype(np.float32) * self.pad_noise_amplitude

        # Concatenate: [NOISE_LEFT] + [AUDIO] + [NOISE_RIGHT]
        padded = np.concatenate([noise_left, audio, noise_right])

        return padded

    def predict(self, audio_path: Path | str) -> PredictionResult:
        """
        Predict SPEECH or NONSPEECH for an audio file.

        Args:
            audio_path: Path to audio file (WAV, MP3, etc.)

        Returns:
            PredictionResult with label, confidence, raw output, and latency
        """
        audio_path = Path(audio_path)

        # Load audio (Qwen2-Audio expects 16kHz numpy array)
        audio, sr = sf.read(audio_path)

        # Get target sample rate from processor
        target_sr = self.processor.feature_extractor.sampling_rate

        # Resample if needed
        if sr != target_sr:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
            sr = target_sr

        # Validate audio (no auto-padding - let processor handle it)
        rms = np.sqrt(np.mean(audio ** 2))
        if rms < 1e-3:
            print(f"[WARNING] Very low RMS ({rms:.6f}) - audio may be silent")

        # Debug: Check audio properties
        # print(f"[DEBUG] Audio shape: {audio.shape}, RMS: {rms:.6f}, Duration: {len(audio)/sr:.2f}s")

        # Start timing
        start_time = time.time()

        # Prepare conversation format
        conversation = [
            {"role": "system", "content": self.system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "audio"},  # No audio_url needed when passing audio directly
                    {"type": "text", "text": self.user_prompt},
                ],
            },
        ]

        # Apply chat template
        text = self.processor.apply_chat_template(
            conversation, add_generation_prompt=True, tokenize=False
        )

        # Process inputs
        # IMPORTANT: Use 'audio' (singular) - processor accepts Union[np.ndarray, list[np.ndarray]]
        # Reference: transformers.models.qwen2_audio.processing_qwen2_audio.Qwen2AudioProcessor
        inputs = self.processor(
            text=text,
            audio=[audio],  # Must be list, even for single audio
            sampling_rate=target_sr,  # Explicit sampling_rate to avoid warnings
            return_tensors="pt",
            padding=True,
        )

        # Debug: Check inputs contain audio features
        # print(f"[DEBUG] Input keys: {inputs.keys()}")
        # if 'audio_features' in inputs or 'input_features' in inputs:
        #     feat_key = 'audio_features' if 'audio_features' in inputs else 'input_features'
        #     print(f"[DEBUG] {feat_key} shape: {inputs[feat_key].shape}")

        # Move to device
        inputs = inputs.to(self.device)

        # Debug: Check if we have audio features after moving to device
        # has_audio = 'input_features' in inputs
        # print(f"[DEBUG] Has input_features: {has_audio}")
        # if has_audio:
        #     print(f"[DEBUG] input_features device: {inputs['input_features'].device}")
        #     print(f"[DEBUG] input_features shape: {inputs['input_features'].shape}")
        #     print(f"[DEBUG] input_features mean: {inputs['input_features'].mean():.4f}, std: {inputs['input_features'].std():.4f}")

        # Generate response
        with torch.no_grad():
            # Use single token for A/B format, longer for free-form
            max_tokens = 1 if self.constrained_decoding else 128

            generate_kwargs = {
                **inputs,
                "max_new_tokens": max_tokens,
                "do_sample": False,  # Greedy decoding for consistency
                "pad_token_id": self.processor.tokenizer.eos_token_id,  # Prevent warnings
            }

            # Add prefix_allowed_tokens_fn if constrained decoding is enabled
            if hasattr(self, "prefix_allowed_tokens_fn_maker"):
                generate_kwargs["prefix_allowed_tokens_fn"] = self.prefix_allowed_tokens_fn_maker(
                    self.first_step_allowed,
                    self.id_eos
                )

            outputs = self.model.generate(**generate_kwargs)

        # Decode ONLY the generated tokens (not the input prompt)
        # outputs contains [input_tokens][generated_tokens], we only want the new ones
        input_length = inputs["input_ids"].shape[1]
        generated_tokens = outputs[:, input_length:]

        output_text = self.processor.batch_decode(
            generated_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]

        # Calculate latency
        latency_ms = (time.time() - start_time) * 1000

        # Parse response
        label, confidence = self._parse_response(output_text)

        return PredictionResult(
            label=label,
            confidence=confidence,
            raw_output=output_text,
            latency_ms=latency_ms,
        )

    def _parse_response(self, text: str) -> tuple[str, float]:
        """
        Parse model response to extract label and confidence.

        Supports multiple response formats:
        - Multiple choice (A/B/C/D) - default strategy
        - Binary (SPEECH/NONSPEECH)
        - Natural language

        Args:
            text: Raw model output text (should be clean, only the generated response)

        Returns:
            Tuple of (label, confidence)
        """
        # Clean response - normalize spaces, remove punctuation, uppercase
        import re
        response = text.strip().upper()
        response_normalized = re.sub(r'[^A-Z0-9\s]', '', response)  # Remove punctuation

        # Priority 1: Multiple choice format (A/B/C/D)
        # A = Speech, B/C/D = Nonspeech (Music/Noise/Animals)
        if "A)" in response or response == "A" or response_normalized == "A":
            return "SPEECH", 1.0
        elif any(x in response for x in ["B)", "C)", "D)"]) or response_normalized in ["B", "C", "D"]:
            return "NONSPEECH", 1.0

        # Priority 2: Explicit NONSPEECH/NON-SPEECH/NO SPEECH variants
        # Handle all variations: "NONSPEECH", "NON-SPEECH", "NON SPEECH", "NO SPEECH"
        nonspeech_patterns = [
            "NONSPEECH",
            "NON SPEECH",
            "NOSPEECH",
            "NO SPEECH"
        ]
        if any(pattern in response_normalized for pattern in nonspeech_patterns):
            return "NONSPEECH", 1.0

        # Priority 3: Keywords indicating no speech (noise, silence, music)
        no_speech_keywords = [
            "NOISE",
            "SILENCE",
            "SILENT",
            "MUSIC",
            "BACKGROUND",
            "AMBIENT",
            "QUIET"
        ]
        if any(keyword in response_normalized for keyword in no_speech_keywords):
            return "NONSPEECH", 0.9

        # Priority 4: Explicit SPEECH keyword (check it's not negated)
        if "SPEECH" in response_normalized:
            # Check if it's negated
            # Look for patterns like "NO SPEECH", "NOT SPEECH", "THERE IS NO SPEECH"
            words_before_speech = response_normalized.split("SPEECH")[0]
            if any(neg in words_before_speech for neg in ["NO", "NOT", "NONE", "WITHOUT", "ISNT", "DOESNT"]):
                return "NONSPEECH", 0.9
            else:
                return "SPEECH", 1.0

        # Priority 5: Natural language negations
        negation_patterns = ["THERE IS NO", "THERE ISNT", "DOES NOT CONTAIN", "NO AUDIO", "NOT DETECTED"]
        if any(pattern in response_normalized for pattern in negation_patterns):
            return "NONSPEECH", 0.8

        # Priority 6: Natural language affirmations
        affirmation_patterns = ["THERE IS", "CONTAINS", "PRESENT", "YES", "DETECTED", "VOICE", "HUMAN"]
        if any(pattern in response_normalized for pattern in affirmation_patterns):
            return "SPEECH", 0.8

        # Unable to parse - return UNKNOWN
        return "UNKNOWN", 0.0

    def set_prompt(self, system_prompt: str | None = None, user_prompt: str | None = None):
        """
        Update the system and/or user prompts.

        Args:
            system_prompt: New system prompt (optional)
            user_prompt: New user prompt (optional)
        """
        if system_prompt is not None:
            self.system_prompt = system_prompt
        if user_prompt is not None:
            self.user_prompt = user_prompt
