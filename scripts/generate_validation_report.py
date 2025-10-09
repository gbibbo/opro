#!/usr/bin/env python3
"""
Generate HTML validation report with embedded audio players.

Creates an interactive HTML page with:
- Summary statistics
- Sample audio players for each category
- Waveform visualizations
- Filterable table

Usage:
    python scripts/generate_validation_report.py
    python scripts/generate_validation_report.py --output validation_report.html
    python scripts/generate_validation_report.py --samples-per-category 10
"""

import argparse
import base64
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import soundfile as sf
from io import BytesIO

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from qsm import CONFIG


def load_all_segments(data_root: Path) -> pd.DataFrame:
    """Load all segment metadata from all datasets."""
    segments_root = data_root / "segments"

    all_metadata = []

    for dataset_dir in segments_root.glob("*"):
        if not dataset_dir.is_dir():
            continue

        for split_dir in dataset_dir.glob("*"):
            if not split_dir.is_dir():
                continue

            metadata_path = split_dir / "segments_metadata.parquet"
            if metadata_path.exists():
                df = pd.read_parquet(metadata_path)
                df["dataset"] = dataset_dir.name
                df["split"] = split_dir.name
                df["segments_dir"] = str(split_dir)
                all_metadata.append(df)

    if not all_metadata:
        raise FileNotFoundError("No segment metadata found")

    combined_df = pd.concat(all_metadata, ignore_index=True)
    return combined_df


def audio_to_base64(audio_path: Path) -> str:
    """Convert audio file to base64 for embedding in HTML."""
    with open(audio_path, "rb") as f:
        audio_data = f.read()

    b64 = base64.b64encode(audio_data).decode("utf-8")
    return f"data:audio/wav;base64,{b64}"


def plot_waveform_to_base64(audio_path: Path) -> str:
    """Generate waveform plot and convert to base64 PNG."""
    data, sr = sf.read(audio_path)

    # Handle mono/stereo
    if data.ndim > 1:
        data = data[:, 0]

    fig, ax = plt.subplots(figsize=(6, 2))
    time = np.arange(len(data)) / sr * 1000  # milliseconds
    ax.plot(time, data, linewidth=0.5, color="steelblue")
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Amplitude")
    ax.set_title(f"Waveform ({len(data) / sr * 1000:.0f}ms)", fontsize=10)
    ax.grid(True, alpha=0.3)

    # Save to base64
    buf = BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format="png", dpi=80)
    plt.close(fig)

    buf.seek(0)
    img_b64 = base64.b64encode(buf.read()).decode("utf-8")
    return f"data:image/png;base64,{img_b64}"


def generate_html_report(
    metadata_df: pd.DataFrame,
    output_path: Path,
    samples_per_category: int = 5,
):
    """Generate interactive HTML validation report."""
    print(f"Generating HTML report with {len(metadata_df)} segments...")

    # Summary statistics
    total_segments = len(metadata_df)
    datasets = metadata_df["dataset"].unique()
    durations = sorted(metadata_df["duration_ms"].unique())
    labels = metadata_df["label"].unique()

    html_parts = []

    # HTML header
    html_parts.append(
        """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Audio Segments Validation Report</title>
        <style>
            body {
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                margin: 20px;
                background-color: #f5f5f5;
            }
            h1 {
                color: #2c3e50;
                border-bottom: 3px solid #3498db;
                padding-bottom: 10px;
            }
            h2 {
                color: #34495e;
                margin-top: 30px;
                border-left: 4px solid #3498db;
                padding-left: 10px;
            }
            .summary {
                background-color: white;
                padding: 20px;
                border-radius: 8px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                margin-bottom: 30px;
            }
            .stat {
                display: inline-block;
                margin: 10px 20px 10px 0;
                padding: 15px 25px;
                background-color: #ecf0f1;
                border-radius: 6px;
                font-size: 18px;
            }
            .stat strong {
                color: #2980b9;
            }
            .segment-card {
                background-color: white;
                padding: 20px;
                margin: 15px 0;
                border-radius: 8px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                border-left: 5px solid #3498db;
            }
            .segment-card.speech {
                border-left-color: #27ae60;
            }
            .segment-card.nonspeech {
                border-left-color: #e74c3c;
            }
            .segment-header {
                font-size: 16px;
                font-weight: bold;
                margin-bottom: 10px;
                color: #2c3e50;
            }
            .segment-info {
                margin: 8px 0;
                color: #7f8c8d;
                font-size: 14px;
            }
            .segment-info span {
                display: inline-block;
                margin-right: 20px;
            }
            audio {
                margin: 15px 0;
                width: 100%;
                max-width: 500px;
            }
            .waveform {
                margin: 15px 0;
            }
            .waveform img {
                max-width: 100%;
                border: 1px solid #ddd;
                border-radius: 4px;
            }
            .filter-controls {
                background-color: white;
                padding: 20px;
                border-radius: 8px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                margin-bottom: 20px;
            }
            .filter-controls select, .filter-controls button {
                margin: 5px 10px 5px 0;
                padding: 8px 15px;
                font-size: 14px;
                border: 1px solid #bdc3c7;
                border-radius: 4px;
            }
            .filter-controls button {
                background-color: #3498db;
                color: white;
                cursor: pointer;
                border: none;
            }
            .filter-controls button:hover {
                background-color: #2980b9;
            }
        </style>
    </head>
    <body>
        <h1>🎵 Audio Segments Validation Report</h1>
    """
    )

    # Summary section
    html_parts.append(
        """
        <div class="summary">
            <h2>📊 Summary</h2>
    """
    )

    html_parts.append(f'<div class="stat"><strong>Total Segments:</strong> {total_segments}</div>')
    html_parts.append(f'<div class="stat"><strong>Datasets:</strong> {", ".join(datasets)}</div>')
    html_parts.append(
        f'<div class="stat"><strong>Durations:</strong> {", ".join([str(d) + "ms" for d in durations])}</div>'
    )

    for label in labels:
        count = (metadata_df["label"] == label).sum()
        html_parts.append(f'<div class="stat"><strong>{label}:</strong> {count}</div>')

    html_parts.append("</div>")  # Close summary

    # Samples section
    html_parts.append('<div id="samples"><h2>🎧 Sample Segments</h2>')

    # Generate samples for each combination
    for dataset in datasets:
        for duration_ms in durations:
            for label in labels:
                # Filter samples
                samples = metadata_df[
                    (metadata_df["dataset"] == dataset)
                    & (metadata_df["duration_ms"] == duration_ms)
                    & (metadata_df["label"] == label)
                ].head(samples_per_category)

                if len(samples) == 0:
                    continue

                html_parts.append(
                    f"""
                    <h3>{dataset.upper()} - {duration_ms}ms - {label}</h3>
                """
                )

                for _, row in samples.iterrows():
                    audio_path = Path(row["audio_path"])

                    if not audio_path.exists():
                        continue

                    # Card class based on label
                    card_class = "speech" if label == "SPEECH" else "nonspeech"

                    # Get filename for display
                    filename = audio_path.name

                    html_parts.append(
                        f"""
                        <div class="segment-card {card_class}">
                            <div class="segment-header">{filename}</div>
                            <div class="segment-info">
                                <span><strong>Label:</strong> {label}</span>
                                <span><strong>Duration:</strong> {duration_ms}ms</span>
                                <span><strong>Time:</strong> {row["start_s"]:.2f}s - {row["end_s"]:.2f}s</span>
                                <span><strong>Condition:</strong> {row.get("condition", "N/A")}</span>
                            </div>
                    """
                    )

                    # Embed audio
                    try:
                        audio_b64 = audio_to_base64(audio_path)
                        html_parts.append(
                            f"""
                            <audio controls>
                                <source src="{audio_b64}" type="audio/wav">
                                Your browser does not support the audio element.
                            </audio>
                        """
                        )

                        # Embed waveform
                        waveform_b64 = plot_waveform_to_base64(audio_path)
                        html_parts.append(
                            f"""
                            <div class="waveform">
                                <img src="{waveform_b64}" alt="Waveform">
                            </div>
                        """
                        )
                    except Exception as e:
                        html_parts.append(f'<p style="color: red;">Error loading audio: {e}</p>')

                    html_parts.append("</div>")  # Close segment-card

    html_parts.append("</div>")  # Close samples

    # HTML footer
    html_parts.append(
        """
    </body>
    </html>
    """
    )

    # Write to file
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("".join(html_parts))

    print(f"Report saved to {output_path}")
    print(f"Open in browser to view: file://{output_path.absolute()}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate HTML validation report for audio segments"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("validation_report.html"),
        help="Output HTML file path (default: validation_report.html)",
    )
    parser.add_argument(
        "--samples-per-category",
        type=int,
        default=5,
        help="Number of samples per (dataset, duration, label) combination (default: 5)",
    )

    args = parser.parse_args()

    # Get data root from config
    data_root = Path(CONFIG["data"]["root"])

    # Load all segments
    print("Loading all segments...")
    metadata_df = load_all_segments(data_root)

    print(f"Loaded {len(metadata_df)} segments from {metadata_df['dataset'].nunique()} datasets")

    # Generate report
    generate_html_report(
        metadata_df=metadata_df,
        output_path=args.output,
        samples_per_category=args.samples_per_category,
    )


if __name__ == "__main__":
    main()
