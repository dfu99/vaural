#!/bin/bash
#SBATCH -A gts-yke8
#SBATCH --job-name=vaural-cipipe
#SBATCH --time=01:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=8
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=daniel.fu@emory.edu
#SBATCH --output=logs/vaural/cipipe_%j.log
#SBATCH --error=logs/vaural/cipipe_%j.err

echo "=== VAURAL C_i Full Pipeline Inverse ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Date: $(date)"

cd ~/scratch/vaural
source venv/bin/activate

python -u experiments/ci_full_pipeline.py 2>&1

echo "=== Done: $(date) ==="
