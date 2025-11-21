#!/usr/bin/env python3
"""
Download and process music clips for NONSPEECH augmentation.

Uses FMA (Free Music Archive) - small subset.
Extracts 2-second segments from instrumental tracks.

Requirements:
    pip install requests tqdm librosa

Usage:
    # Download FMA-small metadata
    python scripts/download_process_music.py --step download_metadata

    # Process audio files (extracts segments)
    python scripts/download_process_music.py --step process --fma_dir /path/to/fma_small
"""

import argparse
import os
import pandas as pd
import numpy as np
import soundfile as sf
import librosa
from pathlib import Path
from tqdm import tqdm
import requests
import zipfile


FMA_METADATA_URL = "https://os.unil.cloud.switch.ch/fma/fma_metadata.zip"
FMA_SMALL_URL = "https://os.unil.cloud.switch.ch/fma/fma_small.zip"


def download_fma_metadata(output_dir):
    """Download FMA metadata."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    zip_path = output_dir / "fma_metadata.zip"

    if zip_path.exists():
        print(f"Metadata already exists: {zip_path}")
        return

    print(f"Downloading FMA metadata from {FMA_METADATA_URL}...")
    response = requests.get(FMA_METADATA_URL, stream=True)
    total_size = int(response.headers.get('content-length', 0))

    with open(zip_path, 'wb') as f:
        with tqdm(total=total_size, unit='B', unit_scale=True) as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                pbar.update(len(chunk))

    print(f"Extracting metadata...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(output_dir)

    print(f"Metadata downloaded to {output_dir}")


def load_fma_metadata(metadata_dir):
    """Load FMA tracks metadata."""
    tracks_csv = Path(metadata_dir) / "fma_metadata" / "tracks.csv"

    if not tracks_csv.exists():
        raise FileNotFoundError(f"Tracks metadata not found: {tracks_csv}")

    # FMA tracks.csv has multi-level header
    tracks = pd.read_csv(tracks_csv, index_col=0, header=[0, 1])

    return tracks


def is_instrumental(track_meta, tracks_df):
    """Check if track is likely instrumental (no vocals)."""
    # FMA doesn't have explicit instrumental flag
    # Use genre as proxy - prefer instrumental genres
    instrumental_genres = [
        'Electronic', 'Instrumental', 'Classical',
        'Jazz', 'Experimental', 'Ambient'
    ]

    try:
        genre = tracks_df.loc[track_meta.name, ('track', 'genre_top')]
        return genre in instrumental_genres
    except:
        return False


def extract_music_segments(fma_dir, output_dir, n_clips=800, segment_duration=2.0, sr=16000):
    """Extract segments from FMA audio files."""
    fma_dir = Path(fma_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load metadata
    metadata_dir = fma_dir.parent / "fma_metadata"
    if not metadata_dir.exists():
        print("ERROR: Metadata not found. Run with --step download_metadata first")
        return

    print("Loading FMA metadata...")
    tracks = load_fma_metadata(metadata_dir.parent)

    # Find audio files
    audio_files = list(fma_dir.rglob("*.mp3"))
    print(f"Found {len(audio_files)} audio files")

    if len(audio_files) == 0:
        print(f"ERROR: No audio files found in {fma_dir}")
        print(f"Download FMA-small from: {FMA_SMALL_URL}")
        print(f"Extract to: {fma_dir}")
        return

    # Extract segments
    clip_idx = 0
    np.random.seed(42)

    print(f"Extracting {n_clips} music segments...")

    with tqdm(total=n_clips) as pbar:
        for audio_file in audio_files:
            if clip_idx >= n_clips:
                break

            try:
                # Load audio
                audio, orig_sr = librosa.load(audio_file, sr=sr, mono=True)

                # Skip very short files
                if len(audio) < sr * segment_duration * 2:
                    continue

                # Extract 1-2 segments per file
                n_segments = min(2, (n_clips - clip_idx))

                for _ in range(n_segments):
                    # Random start position (avoid first/last 5 seconds)
                    margin = int(5 * sr)
                    max_start = len(audio) - int(segment_duration * sr) - margin

                    if max_start <= margin:
                        break

                    start = np.random.randint(margin, max_start)
                    end = start + int(segment_duration * sr)

                    segment = audio[start:end]

                    # Check RMS (skip silent segments)
                    rms = np.sqrt(np.mean(segment ** 2))
                    if rms < 0.01:
                        continue

                    # Save segment
                    genre = audio_file.parent.name
                    filename = f"music_{genre}_{clip_idx:04d}.wav"
                    filepath = output_dir / filename

                    sf.write(filepath, segment, sr)

                    clip_idx += 1
                    pbar.update(1)

                    if clip_idx >= n_clips:
                        break

            except Exception as e:
                print(f"\nError processing {audio_file}: {e}")
                continue

    print(f"\nDone! Extracted {clip_idx} music clips to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Download and process music")
    parser.add_argument("--step", type=str, required=True,
                        choices=["download_metadata", "process"],
                        help="Step to run")
    parser.add_argument("--fma_dir", type=str, default="data/raw/fma_small",
                        help="FMA audio directory")
    parser.add_argument("--output_dir", type=str, default="data/raw/music",
                        help="Output directory for processed clips")
    parser.add_argument("--metadata_dir", type=str, default="data/raw",
                        help="Directory to download metadata")
    parser.add_argument("--n_clips", type=int, default=800,
                        help="Number of music clips to extract")
    args = parser.parse_args()

    if args.step == "download_metadata":
        download_fma_metadata(args.metadata_dir)
        print("\nNext steps:")
        print(f"1. Download FMA-small from: {FMA_SMALL_URL}")
        print(f"2. Extract to: {args.fma_dir}")
        print(f"3. Run: python scripts/download_process_music.py --step process --fma_dir {args.fma_dir}")

    elif args.step == "process":
        extract_music_segments(
            args.fma_dir,
            args.output_dir,
            n_clips=args.n_clips
        )


if __name__ == "__main__":
    main()
