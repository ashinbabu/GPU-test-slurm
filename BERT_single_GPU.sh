#!/bin/bash
#SBATCH --job-name=BERT-1
#SBATCH --partition=gpu-short
#SBATCH --qos=gpu-short
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err

echo "===== Job Info ====="
echo "Job ID: $SLURM_JOB_ID"
echo "Node List: $SLURM_JOB_NODELIST"
echo "GPUs Allocated: $CUDA_VISIBLE_DEVICES"
echo "===================="

# --- Conda Environment Setup ---
module purge
module load cuda/12.8                # Load required CUDA module
# module load miniconda/latest       # Uncomment/Change if needed
source $(conda info --base)/etc/profile.d/conda.sh
conda activate bert         # <-- CHANGE ME to your environment name
# -------------------------------
export LD_PRELOAD=/opt/intel/oneapi/advisor/2025.3/lib64/runtime/libittnotify.so:$LD_PRELOAD
echo "Starting 1-GPU BERT test."
srun /nfs_home/users/vintia/ashin/GPU-demo/bert_test_single_node.py