#!/bin/bash
#
# OPRO Classic: Run and Monitor (One-liner)
#
# Usage:
#   curl -sSL https://raw.githubusercontent.com/gbibbo/opro/main/slurm/run_and_monitor.sh | bash -s base 42
#
# Or download and run locally:
#   bash run_and_monitor.sh base 42
#

set -e

MODE=${1:-base}
SEED=${2:-42}

clear
echo "╔════════════════════════════════════════════════════════════╗"
echo "║         OPRO CLASSIC - AUTO RUN & MONITOR                 ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""
echo "Mode: $MODE"
echo "Seed: $SEED"
echo ""

# Update repo
echo "📦 Updating repository..."
git pull
echo "✓ Repository updated"
echo ""

# Activate conda (adjust for your cluster)
echo "🔧 Activating conda environment..."
source activate qwen_audio || conda activate qwen_audio
echo "✓ Environment: $CONDA_DEFAULT_ENV"
echo ""

# Submit job
echo "🚀 Submitting SLURM job..."
JOB_ID=$(sbatch --parsable slurm/opro_classic.job $MODE $SEED)
echo "✓ Job submitted: $JOB_ID"
echo ""

# Wait for job to start
echo "⏳ Waiting for job to start..."
sleep 5

# Monitor function
monitor_job() {
    local job_id=$1
    local output_log="logs/opro_classic_${job_id}.out"
    local error_log="logs/opro_classic_${job_id}.err"

    # Wait for log files to be created
    while [ ! -f "$output_log" ]; do
        sleep 2
    done

    echo ""
    echo "╔════════════════════════════════════════════════════════════╗"
    echo "║                   MONITORING JOB $job_id                   ║"
    echo "╚════════════════════════════════════════════════════════════╝"
    echo ""
    echo "📊 Job Status: $(squeue -j $job_id -h -o '%T' 2>/dev/null || echo 'COMPLETED')"
    echo "📁 Output log: $output_log"
    echo "📁 Error log: $error_log"
    echo ""
    echo "════════════════════════════════════════════════════════════"
    echo "                    LIVE OUTPUT BELOW"
    echo "════════════════════════════════════════════════════════════"
    echo ""

    # Follow logs with auto-refresh
    tail -f "$output_log" 2>/dev/null &
    TAIL_PID=$!

    # Monitor job status in background
    while true; do
        STATUS=$(squeue -j $job_id -h -o '%T' 2>/dev/null)
        if [ -z "$STATUS" ]; then
            # Job finished
            sleep 2
            kill $TAIL_PID 2>/dev/null

            echo ""
            echo "════════════════════════════════════════════════════════════"
            echo "                    JOB COMPLETED"
            echo "════════════════════════════════════════════════════════════"
            echo ""

            # Show final status
            if [ -s "$error_log" ]; then
                echo "⚠️  Errors detected:"
                tail -20 "$error_log"
                echo ""
            fi

            # Show results
            OUTPUT_DIR="results/opro_classic_${MODE}_seed${SEED}"
            if [ -f "$OUTPUT_DIR/best_prompt.txt" ]; then
                echo "✅ Best prompt found:"
                echo "────────────────────────────────────────────────────────────"
                cat "$OUTPUT_DIR/best_prompt.txt"
                echo "────────────────────────────────────────────────────────────"
                echo ""

                if [ -f "$OUTPUT_DIR/best_metrics.json" ]; then
                    echo "📊 Metrics:"
                    python3 -c "
import json
with open('$OUTPUT_DIR/best_metrics.json') as f:
    m = json.load(f)
    print(f\"  Reward: {m['reward']:.4f}\")
    print(f\"  BA_clip: {m['ba_clip']:.3f}\")
    print(f\"  BA_conditions: {m['ba_conditions']:.3f}\")
    if 'metrics' in m:
        print(f\"  BA_duration: {m['metrics'].get('ba_duration', 0):.3f}\")
        print(f\"  BA_SNR: {m['metrics'].get('ba_snr', 0):.3f}\")
        print(f\"  BA_filter: {m['metrics'].get('ba_filter', 0):.3f}\")
        print(f\"  BA_reverb: {m['metrics'].get('ba_reverb', 0):.3f}\")
" 2>/dev/null || cat "$OUTPUT_DIR/best_metrics.json"
                fi
                echo ""
                echo "📁 All results in: $OUTPUT_DIR"
            else
                echo "❌ Job failed or no results generated"
                echo "Check error log: $error_log"
            fi

            break
        fi
        sleep 10
    done
}

# Start monitoring
monitor_job $JOB_ID

echo ""
echo "════════════════════════════════════════════════════════════"
echo "Done!"
echo "════════════════════════════════════════════════════════════"
