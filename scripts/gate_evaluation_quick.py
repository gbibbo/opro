"""Quick gate evaluation test with subset of data."""

import sys
from pathlib import Path

# Add project root to path
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent
sys.path.insert(0, str(project_root))

# Import from main gate evaluation
from gate_evaluation import (
    evaluate_prompt,
    analyze_robustness,
    generate_gate_report,
)


def main():
    """Quick test with 50 samples."""

    # Paths
    dev_manifest = project_root / "data" / "processed" / "snr_duration_crossed" / "metadata.csv"
    output_dir = project_root / "results" / "gate_evaluation_quick"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Prompts
    baseline_prompt = "Is this audio clip SPEECH or NON-SPEECH?"
    optimized_prompt = "Based on the audio file, is it SPEECH or NON-SPEECH?"

    print("\n" + "="*80)
    print("QUICK GATE EVALUATION TEST (50 samples)")
    print("="*80)

    # Test with 50 samples
    max_samples = 50

    # Step 1: Baseline
    print("\n\nStep 1/3: Baseline (unconstrained)")
    baseline_df = evaluate_prompt(baseline_prompt, dev_manifest, constrained=False, max_samples=max_samples)
    baseline_df.to_csv(output_dir / "baseline_predictions.csv", index=False)
    baseline_results = analyze_robustness(baseline_df)

    # Step 2: Optimized
    print("\n\nStep 2/3: Optimized (unconstrained)")
    optimized_df = evaluate_prompt(optimized_prompt, dev_manifest, constrained=False, max_samples=max_samples)
    optimized_df.to_csv(output_dir / "optimized_predictions.csv", index=False)
    optimized_results = analyze_robustness(optimized_df)

    # Step 3: Optimized + constrained
    print("\n\nStep 3/3: Optimized (constrained)")
    optimized_constrained_df = evaluate_prompt(optimized_prompt, dev_manifest, constrained=True, max_samples=max_samples)
    optimized_constrained_df.to_csv(output_dir / "optimized_constrained_predictions.csv", index=False)
    optimized_constrained_results = analyze_robustness(optimized_constrained_df)

    # Generate report
    print("\n\nGenerating gate report...")
    gate_pass = generate_gate_report(
        baseline_results,
        optimized_results,
        optimized_constrained_results,
        output_dir / "gate_report.txt"
    )

    print("\n" + "="*80)
    print(f"QUICK TEST COMPLETE - Gate {'PASSED' if gate_pass else 'FAILED'}")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
