#!/bin/bash
# Quick script to restart OPRO jobs after fix

set -x

# Cancel the failed job
scancel 2015642

# Pull the fix
cd /mnt/fast/nobackup/users/gb0048/opro
git pull

# Submit both OPRO jobs
BASE_JOB=$(sbatch run_opro_base.sh | awk '{print $4}')
FT_JOB=$(sbatch run_opro_finetuned.sh | awk '{print $4}')

echo ""
echo "Jobs submitted:"
echo "  Base model OPRO: $BASE_JOB"
echo "  Fine-tuned OPRO: $FT_JOB"
echo ""
echo "Monitor with:"
echo "  tail -f /mnt/fast/nobackup/users/gb0048/opro/logs/opro_base_${BASE_JOB}.out"
echo "  tail -f /mnt/fast/nobackup/users/gb0048/opro/logs/opro_base_${BASE_JOB}.err"
echo ""
echo "Or monitor both output and errors together:"
echo "  tail -f /mnt/fast/nobackup/users/gb0048/opro/logs/opro_base_${BASE_JOB}.{out,err}"
