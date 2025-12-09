#!/usr/bin/env python3
"""
Output normalization for multi-format speech detection prompts.

Normalizes model outputs from different prompt formats (A/B, MC, labels, open)
to binary SPEECH/NONSPEECH labels with confidence scores.
"""

import re


def normalize_to_binary(
    text: str,
    probs: dict[str, float] | None = None,
    mode: str = "auto",
    mapping: dict[str, str] | None = None,
    verbalizers: list[str] | None = None,
) -> tuple[str | None, float]:
    """
    Normalize model output to binary SPEECH/NONSPEECH label.

    Priority order (highest to lowest):
    1. Exact match with verbalizers (SPEECH/NONSPEECH)
    2. Letter mapping (A/B/C/D) via provided mapping dict
    3. Yes/No responses
    4. Synonyms (voice/talking vs music/noise/silence)
    5. Unknown (returns None)

    Semantic labels win over letters in ambiguous cases (e.g., "B) SPEECH" → SPEECH).

    Args:
        text: Raw model output text
        probs: Dict of token probabilities (optional, for confidence)
        mode: Format mode ("ab", "mc", "labels", "open", "auto")
        mapping: Dict mapping letters to labels (e.g., {"A": "SPEECH", "B": "NONSPEECH"})
        verbalizers: List of valid label strings (e.g., ["SPEECH", "NONSPEECH"])

    Returns:
        (label, confidence): Binary label (SPEECH/NONSPEECH/None) and confidence score

    Examples:
        >>> normalize_to_binary("A", mapping={"A": "SPEECH", "B": "NONSPEECH"})
        ('SPEECH', 1.0)

        >>> normalize_to_binary("NONSPEECH")
        ('NONSPEECH', 1.0)

        >>> normalize_to_binary("I hear music", mode="open")
        ('NONSPEECH', 0.8)
    """
    if not text:
        return None, 0.0

    # Normalize text
    text_clean = text.strip().upper()
    text_lower = text.strip().lower()

    # Default verbalizers
    if verbalizers is None:
        verbalizers = ["SPEECH", "NONSPEECH"]

    # Default confidence
    confidence = 1.0

    # Extract probability if available
    if probs:
        # Try to get confidence from first token probability
        if "p_first_token" in probs:
            confidence = probs["p_first_token"]

    # Priority 1: Exact match with verbalizers (highest priority)
    # Check for NONSPEECH first to avoid substring issues (SPEECH in NON-SPEECH)
    for verb in ["NONSPEECH", "SPEECH"]:
        if verb not in [v.upper() for v in verbalizers]:
            continue

        # Check it's not negated
        if "NOT " + verb in text_clean or "NO " + verb in text_clean:
            continue

        # For NONSPEECH, check multiple formats
        if verb == "NONSPEECH":
            if "NONSPEECH" in text_clean or "NON-SPEECH" in text_clean or "NO SPEECH" in text_clean:
                return "NONSPEECH", confidence

        # For SPEECH, only match if not part of NONSPEECH/NON-SPEECH/NO ... SPEECH
        elif verb == "SPEECH":
            # Check for negation patterns like "NO SPEECH", "NO HUMAN SPEECH", "NO ... SPEECH"
            import re as re_check

            has_negation = (
                "NONSPEECH" in text_clean
                or "NON-SPEECH" in text_clean
                or "NO SPEECH" in text_clean
                or re_check.search(r"\bNO\b.*\bSPEECH\b", text_clean)  # "NO ... SPEECH"
                or text_clean.startswith("NO,")  # "No, there is no..."
            )

            if "SPEECH" in text_clean and not has_negation:
                return "SPEECH", confidence

    # Priority 2: Letter mapping (A/B/C/D)
    if mapping:
        # Extract first letter from response
        letter_match = re.match(r"^([A-D])", text_clean)
        if letter_match:
            letter = letter_match.group(1)
            if letter in mapping:
                label = mapping[letter]
                # Update confidence if we have letter probabilities
                if probs and letter in probs:
                    confidence = probs[letter]
                return label, confidence

    # Priority 3: Yes/No responses (use word boundaries to avoid false matches)
    yes_patterns = ["YES", "SÍ", "AFFIRMATIVE", "TRUE", "CORRECT"]
    no_patterns = [
        "NO",
        "NEGATIVE",
        "FALSE",
        "INCORRECT",
        "ABSENT",
        "NOT PRESENT",
        "IS NOT PRESENT",
    ]

    # Use word boundary matching to avoid false positives (e.g., "SI" in "SILENCE")
    import re as regex_module

    # First check for NO patterns (higher priority to avoid "NO ... PRESENT" matching "PRESENT")
    for pattern in no_patterns:
        if regex_module.search(r"\b" + regex_module.escape(pattern) + r"\b", text_clean):
            return "NONSPEECH", confidence * 0.95

    # Then check for YES patterns (but skip if there's a negation context)
    for pattern in yes_patterns:
        if regex_module.search(r"\b" + regex_module.escape(pattern) + r"\b", text_clean):
            # Double-check for negation context like "NO ... TRUE" or "NOT ... CORRECT"
            has_negation_context = regex_module.search(
                r"\b(NO|NOT)\b.*\b" + regex_module.escape(pattern) + r"\b", text_clean
            )
            if not has_negation_context:
                return "SPEECH", confidence * 0.95  # Slightly lower confidence for yes/no

    # Priority 4: Synonyms and semantic content
    speech_synonyms = [
        "voice",
        "voices",
        "talking",
        "spoken",
        "speaking",
        "speaker",
        "conversation",
        "conversational",
        "words",
        "utterance",
        "vocal",
        "human voice",
        "person talking",
        "dialogue",
        "speech",
        "syllables",
        "phonemes",
        "formants",
        # Laughter and vocalizations
        "laughter",
        "laugh",
        "laughing",
        "giggle",
        "giggling",
        "chuckle",
        "chuckling",
        # Singing (vocal but musical)
        "singing",
        "sing",
    ]

    nonspeech_synonyms = [
        "music",
        "musical",
        "song",
        "melody",
        "instrumental",
        "beep",
        "beeps",
        "tone",
        "tones",
        "pitch",
        "sine wave",
        "noise",
        "noisy",
        "static",
        "hiss",
        "white noise",
        "silence",
        "silent",
        "quiet",
        "nothing",
        "empty",
        "ambient",
        "environmental",
        "background",
        "click",
        "clicks",
        "clock",
        "tick",
        "ticking",
        # Mechanical and vehicle sounds
        "engine",
        "motor",
        "vehicle",
        "car",
        "truck",
        "bus",
        "siren",
        "accelerating",
        "revving",
        "idling",
        "speeding",
        # Other sound effects
        "spray",
        "splash",
        "power tool",
        "effect",
        "sound effect",
        "ringtone",
        "telephone",
    ]

    # Count matches
    speech_score = sum(1 for syn in speech_synonyms if syn in text_lower)
    nonspeech_score = sum(1 for syn in nonspeech_synonyms if syn in text_lower)

    # Tie-breaker: if both have matches, favor speech (human vocalizations are primary)
    # This handles cases like "background noise followed by laughter" → SPEECH
    if speech_score > nonspeech_score:
        return "SPEECH", confidence * 0.8  # Lower confidence for synonym matching
    elif nonspeech_score > speech_score:
        return "NONSPEECH", confidence * 0.8
    elif speech_score > 0 and speech_score == nonspeech_score:
        # Tie: both present, favor speech with lower confidence
        return "SPEECH", confidence * 0.6

    # Priority 5: Unknown/unparseable
    return None, 0.0


def detect_format(text: str) -> str:
    """
    Auto-detect prompt format from text.

    Args:
        text: Prompt text

    Returns:
        Format string: "ab", "mc", "labels", or "open"
    """
    text_upper = text.upper()

    # Check for multiple choice with D option
    if "A)" in text_upper and "D)" in text_upper:
        return "mc"

    # Check for A/B binary
    if ("A)" in text_upper and "B)" in text_upper) or (
        "OPTION A" in text_upper and "OPTION B" in text_upper
    ):
        return "ab"

    # Check for explicit labels
    if "SPEECH" in text_upper and "NONSPEECH" in text_upper:
        return "labels"

    # Default to open
    return "open"


def validate_mapping(mapping: dict[str, str], label_space: list[str]) -> bool:
    """
    Validate that mapping dict maps to valid labels.

    Args:
        mapping: Letter to label mapping
        label_space: Valid label values

    Returns:
        True if valid, False otherwise
    """
    if not mapping:
        return True

    for letter, label in mapping.items():
        if label not in label_space:
            return False

    return True
