#!/usr/bin/env python3
"""
Sanity check for audio dataset: verify SR, duration, energy, no NaNs/Infs.

This script performs comprehensive validation of audio files to catch
data quality issues before training.
"""

import argparse
import pandas as pd
import numpy as np
import soundfile as sf
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict


def check_audio_file(audio_path, expected_sr=16000, expected_duration_ms=None):
    """
    Check a single audio file for common issues.

    Returns:
        dict with issues found (empty dict = no issues)
    """
    issues = {}

    try:
        # Load audio
        audio, sr = sf.read(audio_path)

        # Check 1: Sampling rate
        if sr != expected_sr:
            issues['sr_mismatch'] = f"Expected {expected_sr} Hz, got {sr} Hz"

        # Check 2: Duration (if specified)
        if expected_duration_ms is not None:
            duration_ms = len(audio) / sr * 1000
            tolerance = 10  # ms
            if abs(duration_ms - expected_duration_ms) > tolerance:
                issues['duration_mismatch'] = (
                    f"Expected {expected_duration_ms} ms, got {duration_ms:.1f} ms"
                )

        # Check 3: NaN/Inf values
        if np.any(np.isnan(audio)):
            issues['nan_values'] = f"Found {np.sum(np.isnan(audio))} NaN values"

        if np.any(np.isinf(audio)):
            issues['inf_values'] = f"Found {np.sum(np.isinf(audio))} Inf values"

        # Check 4: Energy (silence check)
        rms = np.sqrt(np.mean(audio**2))
        if rms < 1e-6:  # Essentially silent
            issues['near_zero_energy'] = f"RMS = {rms:.2e} (near silent)"

        # Check 5: Clipping (values at exactly ±1.0)
        clipped = np.sum((np.abs(audio) >= 0.999))
        if clipped > len(audio) * 0.01:  # More than 1% clipped
            issues['clipping'] = (
                f"{clipped} samples ({clipped/len(audio)*100:.1f}%) at ±1.0"
            )

        # Check 6: Dynamic range
        peak = np.abs(audio).max()
        if peak < 0.1:  # Very quiet
            issues['low_peak'] = f"Peak amplitude = {peak:.3f} (very quiet)"

        # Return metrics for statistics
        return {
            'issues': issues,
            'sr': sr,
            'duration_ms': len(audio) / sr * 1000,
            'rms': rms,
            'peak': peak,
            'n_samples': len(audio)
        }

    except Exception as e:
        return {
            'issues': {'load_error': str(e)},
            'sr': None,
            'duration_ms': None,
            'rms': None,
            'peak': None,
            'n_samples': None
        }


def main():
    parser = argparse.ArgumentParser(
        description="Sanity check audio files for quality issues"
    )
    parser.add_argument(
        "--metadata_csv",
        type=str,
        required=True,
        help="Path to metadata CSV"
    )
    parser.add_argument(
        "--expected_sr",
        type=int,
        default=16000,
        help="Expected sampling rate (default: 16000 Hz)"
    )
    parser.add_argument(
        "--check_duration",
        action="store_true",
        help="Check that audio duration matches metadata duration_ms column"
    )
    parser.add_argument(
        "--output_report",
        type=str,
        default=None,
        help="Path to save detailed report CSV (optional)"
    )

    args = parser.parse_args()

    # Load metadata
    print(f"Loading metadata from {args.metadata_csv}")
    df = pd.read_csv(args.metadata_csv)
    print(f"Found {len(df)} entries\n")

    # Determine label column
    label_col = 'ground_truth' if 'ground_truth' in df.columns else 'label'

    # Check each audio file
    results = []
    issue_counts = defaultdict(int)
    all_issues = []

    print("Checking audio files...")
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        audio_path = row['audio_path']
        expected_duration = row['duration_ms'] if args.check_duration else None

        result = check_audio_file(
            audio_path,
            expected_sr=args.expected_sr,
            expected_duration_ms=expected_duration
        )

        # Count issues
        for issue_type in result['issues'].keys():
            issue_counts[issue_type] += 1

        if result['issues']:
            all_issues.append({
                'audio_path': audio_path,
                'clip_id': row['clip_id'],
                'label': row[label_col],
                **result['issues']
            })

        results.append({
            'clip_id': row['clip_id'],
            'label': row[label_col],
            'audio_path': audio_path,
            **{k: v for k, v in result.items() if k != 'issues'}
        })

    # Convert results to DataFrame
    results_df = pd.DataFrame(results)

    # Print summary
    print(f"\n{'='*60}")
    print("SANITY CHECK SUMMARY")
    print(f"{'='*60}")

    print(f"\nTotal files checked: {len(df)}")
    print(f"Files with issues: {len(all_issues)} ({len(all_issues)/len(df)*100:.1f}%)")
    print(f"Files OK: {len(df) - len(all_issues)} ({(len(df)-len(all_issues))/len(df)*100:.1f}%)")

    if issue_counts:
        print(f"\nIssue breakdown:")
        for issue_type, count in sorted(issue_counts.items(), key=lambda x: -x[1]):
            print(f"  {issue_type}: {count} files")
    else:
        print("\n✓ No issues found!")

    # Audio statistics
    print(f"\n{'='*60}")
    print("AUDIO STATISTICS")
    print(f"{'='*60}")

    valid_results = results_df[results_df['sr'].notna()]

    if len(valid_results) > 0:
        print(f"\nSampling rates:")
        sr_counts = valid_results['sr'].value_counts()
        for sr, count in sr_counts.items():
            print(f"  {int(sr)} Hz: {count} files ({count/len(valid_results)*100:.1f}%)")

        print(f"\nRMS distribution:")
        print(f"  Min:    {valid_results['rms'].min():.6f}")
        print(f"  25th %: {valid_results['rms'].quantile(0.25):.6f}")
        print(f"  Median: {valid_results['rms'].median():.6f}")
        print(f"  75th %: {valid_results['rms'].quantile(0.75):.6f}")
        print(f"  Max:    {valid_results['rms'].max():.6f}")
        print(f"  Range:  {valid_results['rms'].max()/valid_results['rms'].min():.2f}x")

        print(f"\nPeak amplitude distribution:")
        print(f"  Min:    {valid_results['peak'].min():.6f}")
        print(f"  25th %: {valid_results['peak'].quantile(0.25):.6f}")
        print(f"  Median: {valid_results['peak'].median():.6f}")
        print(f"  75th %: {valid_results['peak'].quantile(0.75):.6f}")
        print(f"  Max:    {valid_results['peak'].max():.6f}")

        # Top outliers by RMS
        print(f"\nTop 5 loudest files (by RMS):")
        top_rms = valid_results.nlargest(5, 'rms')[['clip_id', 'label', 'rms', 'peak']]
        for _, row in top_rms.iterrows():
            print(f"  {row['clip_id']} ({row['label']}): RMS={row['rms']:.6f}, Peak={row['peak']:.6f}")

        print(f"\nTop 5 quietest files (by RMS):")
        bottom_rms = valid_results.nsmallest(5, 'rms')[['clip_id', 'label', 'rms', 'peak']]
        for _, row in bottom_rms.iterrows():
            print(f"  {row['clip_id']} ({row['label']}): RMS={row['rms']:.6f}, Peak={row['peak']:.6f}")

    # Print detailed issues
    if all_issues:
        print(f"\n{'='*60}")
        print("DETAILED ISSUES (first 20)")
        print(f"{'='*60}\n")

        for issue in all_issues[:20]:
            print(f"File: {issue['audio_path']}")
            print(f"  clip_id: {issue['clip_id']}")
            print(f"  label: {issue['label']}")
            for key, value in issue.items():
                if key not in ['audio_path', 'clip_id', 'label']:
                    print(f"  ⚠️  {key}: {value}")
            print()

        if len(all_issues) > 20:
            print(f"... and {len(all_issues) - 20} more issues")

    # Save report if requested
    if args.output_report:
        issues_df = pd.DataFrame(all_issues)
        issues_df.to_csv(args.output_report, index=False)
        print(f"\nDetailed report saved to: {args.output_report}")

    # Exit code
    if issue_counts:
        print(f"\n⚠️  Found {len(all_issues)} files with issues")
        exit(1)
    else:
        print(f"\n✓ All audio files passed sanity checks!")
        exit(0)


if __name__ == "__main__":
    main()
