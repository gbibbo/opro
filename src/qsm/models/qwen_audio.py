"""Qwen2-Audio wrapper for binary SPEECH/NONSPEECH classification."""

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import librosa
import numpy as np
import soundfile as sf
import torch
from transformers import AutoProcessor, LogitsProcessor, Qwen2AudioForConditionalGeneration


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
        auto_pad: bool = True,
        pad_target_ms: int = 2000,  # Optimal: 2000ms provides best performance
        pad_noise_amplitude: float = 0.0001,
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
        self.processor = AutoProcessor.from_pretrained(model_name)

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

        # Default prompt (multiple choice strategy - validated with 100% accuracy on 1000ms segments)
        self.system_prompt = "You classify audio content."

        self.user_prompt = (
            "What best describes this audio?\n"
            "A) Human speech or voice\n"
            "B) Music\n"
            "C) Noise or silence\n"
            "D) Animal sounds\n\n"
            "Answer with ONLY the letter (A, B, C, or D)."
        )

        print("Model loaded successfully!")
        if self.auto_pad:
            print(
                f"Auto-padding enabled: <{self.pad_target_ms}ms -> {self.pad_target_ms}ms (noise amplitude: {self.pad_noise_amplitude})"
            )

        # Initialize constrained decoding if enabled
        self.logits_processor = None
        if self.constrained_decoding:
            # Get token IDs for SPEECH and NONSPEECH
            speech_ids = self.processor.tokenizer.encode("SPEECH", add_special_tokens=False)
            nonspeech_ids = self.processor.tokenizer.encode("NONSPEECH", add_special_tokens=False)

            # Combine all allowed token IDs
            allowed_ids = speech_ids + nonspeech_ids

            # Create logits processor
            from transformers import LogitsProcessorList
            self.logits_processor = LogitsProcessorList([
                ConstrainedVocabLogitsProcessor(allowed_ids)
            ])

            print(f"Constrained decoding enabled: only tokens {allowed_ids} allowed")

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

        # Resample if needed
        if sr != 16000:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
            sr = 16000

        # Apply automatic padding if enabled
        if self.auto_pad:
            audio = self._pad_audio_with_noise(audio, sample_rate=sr)

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
        # IMPORTANT: Parameter is 'audio' (singular), not 'audios' (plural)
        # Must pass numpy array directly, not wrapped in list
        inputs = self.processor(
            text=text,
            audio=audio,  # Singular, numpy array directly
            sampling_rate=sr,  # Explicitly pass to avoid warnings
            return_tensors="pt",
            padding=True,
        )

        # Move to device
        inputs = inputs.to(self.device)

        # Generate response
        with torch.no_grad():
            generate_kwargs = {
                **inputs,
                "max_new_tokens": 128,
                "do_sample": False,  # Greedy decoding for consistency
            }

            # Add logits processor if constrained decoding is enabled
            if self.logits_processor is not None:
                generate_kwargs["logits_processor"] = self.logits_processor

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
        # Clean response
        response = text.strip().upper()

        # Priority 1: Multiple choice format (A/B/C/D)
        # A = Speech, B/C/D = Nonspeech (Music/Noise/Animals)
        if "A)" in response or response == "A":
            return "SPEECH", 1.0
        elif any(x in response for x in ["B)", "C)", "D)"]) or response in ["B", "C", "D"]:
            return "NONSPEECH", 1.0

        # Priority 2: Explicit NONSPEECH or NO-SPEECH keywords
        if "NONSPEECH" in response or "NO-SPEECH" in response:
            return "NONSPEECH", 1.0

        # Priority 3: Explicit SPEECH keyword (but not as part of "no speech")
        if "SPEECH" in response:
            # Check if it's negated
            # Look for patterns like "NO SPEECH", "NOT SPEECH", "THERE IS NO SPEECH"
            words_before_speech = response.split("SPEECH")[0]
            if any(neg in words_before_speech for neg in ["NO", "NOT", "NONE", "WITHOUT"]):
                return "NONSPEECH", 0.9
            else:
                return "SPEECH", 1.0

        # Priority 4: Natural language negations
        negation_patterns = ["THERE IS NO", "THERE ISN'T", "DOES NOT CONTAIN", "NO AUDIO"]
        if any(pattern in response for pattern in negation_patterns):
            return "NONSPEECH", 0.8

        # Priority 5: Natural language affirmations
        affirmation_patterns = ["THERE IS", "CONTAINS", "PRESENT", "YES", "DETECTED"]
        if any(pattern in response for pattern in affirmation_patterns):
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
