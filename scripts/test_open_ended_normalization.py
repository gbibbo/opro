#!/usr/bin/env python3
"""
Test script to verify open-ended prompt response normalization.

This tests that the normalize_to_binary function correctly maps
open-ended responses to SPEECH/NONSPEECH labels.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.qsm.utils.normalize import normalize_to_binary


def test_normalization():
    """Test various open-ended responses."""
    test_cases = [
        # Speech responses
        ("I hear someone talking", "SPEECH"),
        ("There's a human voice", "SPEECH"),
        ("Sounds like speech", "SPEECH"),
        ("A person is speaking", "SPEECH"),
        ("Human conversation", "SPEECH"),
        ("Spoken words", "SPEECH"),
        ("Vocal sounds", "SPEECH"),

        # Laughter and vocalizations (NEW - previously failed)
        ("Laughter.", "SPEECH"),
        ("I heard laughter and the sound of water splashing.", "SPEECH"),
        ("There is a background noise followed by laughter.", "SPEECH"),
        ("I heard giggling", "SPEECH"),

        # Non-speech responses
        ("I hear music", "NONSPEECH"),
        ("It's just noise", "NONSPEECH"),
        ("Silence", "NONSPEECH"),
        ("Background noise", "NONSPEECH"),
        ("Beeping sounds", "NONSPEECH"),
        ("Musical instrument", "NONSPEECH"),
        ("Clock ticking", "NONSPEECH"),

        # Mechanical and vehicle sounds (NEW - previously failed)
        ("I heard an engine accelerating and revving", "NONSPEECH"),
        ("It's a car engine starting and revving.", "NONSPEECH"),
        ("The sound is that of an engine idling loudly.", "NONSPEECH"),
        ("I heard the sound of a motor vehicle on the road", "NONSPEECH"),
        ("It's a sound effect.", "NONSPEECH"),
        ("There is a sound of liquid spraying and gushing.", "NONSPEECH"),

        # Negations (NEW - previously failed critically)
        ("No, there is no human speech in this audio.", "NONSPEECH"),
        ("No, the transcription is empty indicating no speech content.", "NONSPEECH"),
        ("No, the transcription is empty indicating no human speech is present.", "NONSPEECH"),

        # Constrained responses (should still work)
        ("SPEECH", "SPEECH"),
        ("NON-SPEECH", "NONSPEECH"),
        ("A) SPEECH", "SPEECH"),
        ("B", None),  # Ambiguous without mapping

        # Ambiguous (should return None or guess)
        ("I don't know", None),
        ("Unclear", None),
    ]

    print("="*70)
    print("TESTING OPEN-ENDED RESPONSE NORMALIZATION")
    print("="*70)

    passed = 0
    failed = 0

    for response, expected in test_cases:
        label, confidence = normalize_to_binary(response, mode="auto")

        status = "[PASS]" if label == expected else "[FAIL]"
        if label == expected:
            passed += 1
        else:
            failed += 1

        print(f"{status} '{response[:40]}' -> {label} (expected: {expected}, conf: {confidence:.2f})")

    print("="*70)
    print(f"Results: {passed} passed, {failed} failed")
    print("="*70)

    return failed == 0


if __name__ == "__main__":
    success = test_normalization()
    sys.exit(0 if success else 1)
