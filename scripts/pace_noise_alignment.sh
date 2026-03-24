#!/bin/bash
#SBATCH -A gts-yke8
#SBATCH --job-name=vaural-noise
#SBATCH --time=04:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:RTX_6000:1
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=daniel.fu@emory.edu
#SBATCH --output=logs/vaural/noise_%j.log
#SBATCH --error=logs/vaural/noise_%j.err

echo "=== VAURAL Noise Alignment Boundary ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Date: $(date)"

module load cuda

cd ~/scratch/vaural
source venv/bin/activate

echo "Python: $(which python)"
python -c "import torch; print(f'Torch: {torch.__version__}, CUDA: {torch.cuda.is_available()}, Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU\"}')"
echo ""

python -u experiments/noise_alignment_boundary.py 2>&1

echo ""
echo "=== Done: $(date) ==="
