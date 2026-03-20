#!/bin/bash
#SBATCH -A gts-yke8
#SBATCH --job-name=vaural-fixrecv
#SBATCH --time=02:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=8
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=daniel.fu@emory.edu
#SBATCH --output=logs/vaural/fixrecv_%j.log
#SBATCH --error=logs/vaural/fixrecv_%j.err

echo "=== VAURAL Fixed Receiver Experiment ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Date: $(date)"

cd ~/scratch/vaural
source venv/bin/activate

echo "Python: $(which python)"
echo ""

python -u experiments/fixed_receiver.py 2>&1

echo ""
echo "=== Done: $(date) ==="
