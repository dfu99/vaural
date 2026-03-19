#!/bin/bash
#SBATCH -A gts-yke8
#SBATCH --job-name=vaural-ci
#SBATCH --time=02:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=8
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=daniel.fu@emory.edu
#SBATCH --output=logs/vaural/ci_%j.log
#SBATCH --error=logs/vaural/ci_%j.err

echo "=== VAURAL C_i Experiment ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Date: $(date)"
echo ""

cd ~/scratch/vaural

# Create venv if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
    source venv/bin/activate
    pip install --upgrade pip
    pip install torch numpy matplotlib
else
    source venv/bin/activate
fi

echo "Python: $(which python)"
echo "Torch: $(python -c 'import torch; print(torch.__version__)')"
echo ""

# Run experiment
python -u experiments/coordination_quality.py 2>&1

echo ""
echo "=== Done: $(date) ==="
