#!/usr/bin/env python3
"""
Interactive audio segment player for validation.

Allows users to:
- Listen to segments
- View waveform/spectrogram
- Filter by dataset, duration, label, condition
- Collect feedback on label accuracy

Usage:
    python scripts/play_segments_interactive.py --segments-dir data/segments/ava_speech/train
    python scripts/play_segments_interactive.py --dataset ava-speech --duration 40 --label SPEECH
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
import soundfile as sf

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def load_segments_metadata(segments_dir: Path) -> pd.DataFrame:
    """Load segments metadata from parquet file."""
    metadata_path = segments_dir / "segments_metadata.parquet"

    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata not found: {metadata_path}")

    df = pd.read_parquet(metadata_path)
    return df


def play_segment(audio_path: Path):
    """Play an audio segment (requires system player)."""
    import platform
    import subprocess
    import os

    system = platform.system()

    try:
        if system == "Windows":
            # Windows: use default player
            subprocess.run(["start", "", str(audio_path)], shell=True, check=True)
        elif system == "Darwin":
            # macOS: use afplay
            subprocess.run(["afplay", str(audio_path)], check=True)
        else:
            # Linux/WSL: Check if running under WSL
            is_wsl = "microsoft" in str(os.uname().release).lower()

            if is_wsl:
                # WSL: Convert path to Windows format and use Windows player
                # Convert /mnt/c/... to C:\...
                path_str = str(audio_path)
                if path_str.startswith("/mnt/"):
                    drive_letter = path_str[5].upper()
                    win_path = f"{drive_letter}:{path_str[6:]}".replace("/", "\\")
                else:
                    win_path = path_str.replace("/", "\\")

                # Use cmd.exe to open with default Windows player
                subprocess.run(
                    ["cmd.exe", "/c", "start", "", win_path],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            else:
                # Regular Linux: try multiple players
                for player in ["aplay", "paplay", "ffplay", "mplayer"]:
                    try:
                        subprocess.run(
                            [player, str(audio_path)],
                            check=True,
                            stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL,
                        )
                        break
                    except (FileNotFoundError, subprocess.CalledProcessError):
                        continue
    except Exception as e:
        print(f"Failed to play audio: {e}")
        print(f"Please manually play: {audio_path}")


def show_segment_info(row: pd.Series, segments_dir: Path):
    """Display segment information."""
    # Normalize path separators for cross-platform compatibility
    import os
    path_str = str(row["audio_path"]).replace("\\", "/")
    audio_path = Path(path_str)
    filename = os.path.basename(path_str)

    print("\n" + "=" * 80)
    print(f"Segment: {filename}")
    print("=" * 80)
    print(f"  Dataset:    {row.get('dataset', 'N/A')}")
    print(f"  URI:        {row['uri']}")
    print(f"  Label:      {row['label']}")
    print(f"  Duration:   {row['duration_ms']}ms")
    print(f"  Time:       {row['start_s']:.2f}s - {row['end_s']:.2f}s")
    print(f"  Condition:  {row.get('condition', 'N/A')}")

    if audio_path.exists():
        # Get audio info
        info = sf.info(audio_path)
        print(f"  Sample rate: {info.samplerate}Hz")
        print(f"  Channels:   {info.channels}")
        print(f"  File size:  {audio_path.stat().st_size / 1024:.1f}KB")
    print(f"  Path:       {audio_path}")
    print("=" * 80)


def interactive_player(
    metadata_df: pd.DataFrame,
    segments_dir: Path,
    auto_play: bool = False,
    filter_dataset: str | None = None,
    filter_duration: int | None = None,
    filter_label: str | None = None,
    filter_condition: str | None = None,
    min_duration: int | None = None,
):
    """
    Interactive player for segment validation.

    Commands:
        p - play current segment
        n - next segment
        b - back to previous segment
        i - show info
        s - skip to segment number
        f - mark as correct (feedback)
        x - mark as incorrect (feedback)
        q - quit
    """
    # Apply filters
    df = metadata_df.copy()

    if filter_dataset:
        df = df[df.get("dataset", "") == filter_dataset]
    if filter_duration:
        df = df[df["duration_ms"] == filter_duration]
    if filter_label:
        df = df[df["label"] == filter_label]
    if filter_condition:
        df = df[df.get("condition", "") == filter_condition]
    if min_duration:
        df = df[df["duration_ms"] >= min_duration]

    if len(df) == 0:
        print("No segments match the filter criteria.")
        return

    print(f"\n{len(df)} segments loaded")
    filter_msg = f"Filters: dataset={filter_dataset}, duration={filter_duration}ms, label={filter_label}, condition={filter_condition}"
    if min_duration:
        filter_msg += f", min_duration>={min_duration}ms"
    print(filter_msg)

    # Feedback tracking
    feedback = {}

    current_idx = 0

    print("\nCommands: p=play, n=next, b=back, i=info, s=skip to #, f=correct, x=incorrect, q=quit")

    while True:
        row = df.iloc[current_idx]

        # Get audio path - normalize path separators for cross-platform
        path_str = str(row["audio_path"]).replace("\\", "/")
        audio_path = Path(path_str)

        # Get just the filename (handle both Windows and Unix paths)
        import os
        filename = os.path.basename(path_str)

        # Show progress
        print(f"\n[{current_idx + 1}/{len(df)}] {filename} - {row['label']}")

        # Auto-play if enabled
        if auto_play:
            if audio_path.exists():
                print("Playing...")
                play_segment(audio_path)

        # Get command
        cmd = input(">>> ").strip().lower()

        if cmd == "q":
            break
        elif cmd == "p":
            if audio_path.exists():
                play_segment(audio_path)
            else:
                print(f"Audio file not found: {audio_path}")
        elif cmd == "n":
            current_idx = (current_idx + 1) % len(df)
        elif cmd == "b":
            current_idx = (current_idx - 1) % len(df)
        elif cmd == "i":
            show_segment_info(row, segments_dir)
        elif cmd == "s":
            try:
                skip_to = int(input("Skip to segment number (1-based): ")) - 1
                if 0 <= skip_to < len(df):
                    current_idx = skip_to
                else:
                    print(f"Invalid segment number. Must be 1-{len(df)}")
            except ValueError:
                print("Invalid input")
        elif cmd == "f":
            feedback[filename] = "correct"
            print("Marked as correct")
            current_idx = (current_idx + 1) % len(df)
        elif cmd == "x":
            feedback[filename] = "incorrect"
            print("Marked as incorrect")
            current_idx = (current_idx + 1) % len(df)
        else:
            print("Unknown command. Use: p, n, b, i, s, f, x, q")

    # Save feedback
    if feedback:
        feedback_path = segments_dir / "validation_feedback.jsonl"
        import json

        with open(feedback_path, "w") as f:
            for filename, status in feedback.items():
                f.write(json.dumps({"filename": filename, "status": status}) + "\n")

        print(f"\nFeedback saved to {feedback_path}")
        print(f"   Correct: {sum(1 for v in feedback.values() if v == 'correct')}")
        print(f"   Incorrect: {sum(1 for v in feedback.values() if v == 'incorrect')}")


def main():
    parser = argparse.ArgumentParser(
        description="Interactive audio segment player for validation"
    )
    parser.add_argument(
        "--segments-dir",
        type=str,
        required=True,
        help="Directory containing segments (e.g., data/segments/ava_speech/train)",
    )
    parser.add_argument(
        "--auto-play",
        action="store_true",
        help="Automatically play each segment",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        help="Filter by dataset (e.g., ava_speech, voxconverse)",
    )
    parser.add_argument(
        "--duration",
        type=int,
        help="Filter by duration in ms (e.g., 100, 500)",
    )
    parser.add_argument(
        "--label",
        type=str,
        choices=["SPEECH", "NONSPEECH"],
        help="Filter by label",
    )
    parser.add_argument(
        "--condition",
        type=str,
        help="Filter by condition (e.g., clean, noise, music)",
    )
    parser.add_argument(
        "--min-duration",
        type=int,
        help="Minimum duration in ms (e.g., 500 to skip short segments)",
    )

    args = parser.parse_args()

    segments_dir = Path(args.segments_dir)

    if not segments_dir.exists():
        print(f"❌ Segments directory not found: {segments_dir}")
        print(f"   Absolute path: {segments_dir.absolute()}")
        sys.exit(1)

    # Load metadata
    print(f"Loading segments from {segments_dir}...")
    metadata_df = load_segments_metadata(segments_dir)

    print(f"Loaded {len(metadata_df)} segments")

    # Start interactive player
    interactive_player(
        metadata_df=metadata_df,
        segments_dir=segments_dir,
        auto_play=args.auto_play,
        filter_dataset=args.dataset,
        filter_duration=args.duration,
        filter_label=args.label,
        filter_condition=args.condition,
        min_duration=args.min_duration,
    )


if __name__ == "__main__":
    main()
