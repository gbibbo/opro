#!/usr/bin/env python3
"""
Verify and listen to generated audio segments.

This script allows you to:
1. List all generated segments
2. Play random samples to verify labels
3. Check segment statistics
"""

import argparse
import json
import random
import sys
from pathlib import Path

import pandas as pd

try:
    import sounddevice as sd
    import soundfile as sf

    AUDIO_PLAYBACK_AVAILABLE = True
except ImportError:
    AUDIO_PLAYBACK_AVAILABLE = False
    print("Warning: sounddevice not installed. Audio playback not available.")
    print("Install with: pip install sounddevice")


def play_audio(audio_path: Path):
    """Play an audio file."""
    if not AUDIO_PLAYBACK_AVAILABLE:
        print("❌ Audio playback not available (sounddevice not installed)")
        return

    audio, sr = sf.read(audio_path)
    print(f"▶️  Playing: {audio_path.name}")
    print(f"   Duration: {len(audio)/sr*1000:.1f}ms")
    print(f"   Sample rate: {sr}Hz")
    sd.play(audio, sr)
    sd.wait()


def list_segments(segments_dir: Path, label_filter: str = None):
    """List all segments with optional label filter."""
    metadata_path = segments_dir / "segments_metadata.jsonl"

    if not metadata_path.exists():
        print(f"❌ Metadata file not found: {metadata_path}")
        return []

    segments = []
    with open(metadata_path, "r") as f:
        for line in f:
            seg = json.loads(line)
            if label_filter is None or seg["label"] == label_filter:
                segments.append(seg)

    return segments


def main():
    parser = argparse.ArgumentParser(description="Verify audio segments")
    parser.add_argument(
        "--segments-dir",
        type=str,
        required=True,
        help="Directory containing segments",
    )
    parser.add_argument(
        "--label",
        type=str,
        choices=["SPEECH", "NONSPEECH", "ALL"],
        default="ALL",
        help="Filter by label",
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=None,
        help="Filter by duration (ms)",
    )
    parser.add_argument(
        "--play-random",
        type=int,
        default=0,
        help="Play N random samples",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List all segments",
    )

    args = parser.parse_args()

    # Convert to Path and handle both Windows and Unix paths
    segments_dir = Path(args.segments_dir)

    if not segments_dir.exists():
        print(f"❌ Segments directory not found: {segments_dir}")
        print(f"   Absolute path tried: {segments_dir.absolute()}")
        sys.exit(1)

    args.segments_dir = segments_dir

    print("=" * 80)
    print("SEGMENT VERIFICATION")
    print("=" * 80)
    print(f"Directory: {args.segments_dir}")
    print()

    # Load metadata
    label_filter = None if args.label == "ALL" else args.label
    segments = list_segments(args.segments_dir, label_filter)

    if args.duration is not None:
        segments = [s for s in segments if s["duration_ms"] == args.duration]

    print(f"Total segments: {len(segments)}")

    if len(segments) == 0:
        print("❌ No segments found with the specified filters")
        sys.exit(1)

    # Statistics
    df = pd.DataFrame(segments)

    print("\nSegments by duration:")
    duration_counts = df["duration_ms"].value_counts().sort_index()
    for duration_ms, count in duration_counts.items():
        print(f"  {duration_ms:4d}ms: {count:4d} segments")

    print("\nSegments by label:")
    label_counts = df["label"].value_counts()
    for label, count in label_counts.items():
        print(f"  {label:12s}: {count:4d} segments")

    if "condition" in df.columns:
        print("\nSegments by condition:")
        condition_counts = df["condition"].value_counts()
        for condition, count in condition_counts.items():
            if pd.notna(condition):
                print(f"  {str(condition):12s}: {count:4d} segments")

    # List segments
    if args.list:
        print("\n" + "=" * 80)
        print("SEGMENT LIST")
        print("=" * 80)
        for i, seg in enumerate(segments[:20], 1):  # Show first 20
            label_marker = "🗣️ " if seg["label"] == "SPEECH" else "🔇"
            print(
                f"{i:3d}. {label_marker} {Path(seg['audio_path']).name} | "
                f"{seg['start_s']:.2f}-{seg['end_s']:.2f}s | "
                f"{seg['duration_ms']}ms | "
                f"condition={seg.get('condition', 'N/A')}"
            )

        if len(segments) > 20:
            print(f"\n... and {len(segments) - 20} more segments")

    # Play random samples
    if args.play_random > 0:
        print("\n" + "=" * 80)
        print(f"PLAYING {args.play_random} RANDOM SAMPLES")
        print("=" * 80)

        if not AUDIO_PLAYBACK_AVAILABLE:
            print("❌ Audio playback not available")
            print("Install with: pip install sounddevice")
            sys.exit(1)

        samples = random.sample(segments, min(args.play_random, len(segments)))

        for i, seg in enumerate(samples, 1):
            label_marker = "🗣️  SPEECH" if seg["label"] == "SPEECH" else "🔇 NO SPEECH"
            print(f"\n[{i}/{len(samples)}] {label_marker}")
            print(f"File: {Path(seg['audio_path']).name}")
            print(f"Duration: {seg['duration_ms']}ms")
            print(f"Condition: {seg.get('condition', 'N/A')}")
            print(f"Time range: {seg['start_s']:.2f}-{seg['end_s']:.2f}s")

            audio_path = Path(seg["audio_path"])
            if audio_path.exists():
                play_audio(audio_path)
                input("Press Enter to continue to next sample...")
            else:
                print(f"❌ Audio file not found: {audio_path}")

    print("\n" + "=" * 80)
    print("✅ VERIFICATION COMPLETE")
    print("=" * 80)
    print("\nTo play random samples, use:")
    print(f"  python scripts/verify_segments.py --segments-dir {args.segments_dir} --play-random 5")
    print("\nTo filter by label:")
    print(
        f"  python scripts/verify_segments.py --segments-dir {args.segments_dir} --label SPEECH --list"
    )


if __name__ == "__main__":
    main()
