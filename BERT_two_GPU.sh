#!/bin/bash
#
# ======================================================================
# SLURM Directives: 2-GPU Request
# ======================================================================
#SBATCH --job-name=BERT_2GPU_Bench
#SBATCH --output=logs/slurm-%j.out         # Output file for shell commands
#SBATCH --error=logs/slurm-%j.err          # Error file for shell commands
#SBATCH --nodes=1                          # Request only one node
#SBATCH --ntasks-per-node=1                # One main SLURM task
#SBATCH --cpus-per-task=4                  # CPUs per task (can increase this)
#SBATCH --mem=32G
#SBATCH --time=00:10:00                    # Set a short time limit for the test
#SBATCH --partition=gpu-short              # Your cluster partition
#SBATCH --gres=gpu:2                       # <-- Requests 2 GPUs
#
# ======================================================================
# Setup and Execution
# ======================================================================

echo "Starting job $SLURM_JOB_ID on node $SLURM_NODELIST"

# 1. Activate your Conda Environment
source /nfs_home/software/miniconda/etc/profile.d/conda.sh
conda activate bert

# 2. Extract GPU IDs and Count
# SLURM sets CUDA_VISIBLE_DEVICES (e.g., "0,1") based on --gres=gpu:2
IFS=',' read -r -a GPU_IDS <<< "$CUDA_VISIBLE_DEVICES"
NUM_GPUS=${#GPU_IDS[@]}

echo "--- SLURM Launch Info ---"
echo "GPUs Requested: $NUM_GPUS"
echo "GPU IDs Allocated: $CUDA_VISIBLE_DEVICES"
echo "-------------------------"
echo "Launching $NUM_GPUS independent processes concurrently..."

# 3. Launch each process independently in the background
# This loop runs the script once for GPU 0 and once for GPU 1.
for i in $(seq 0 $((NUM_GPUS - 1))); do
    GPU_ID=${GPU_IDS[$i]}
    
    # CRITICAL: Lock the process to a specific GPU and run in background (&)
    # The output of the Python script is redirected to a unique log file.
    CUDA_VISIBLE_DEVICES=$GPU_ID python /nfs_home/users/vintia/ashin/GPU-demo/bert_test_single_node.py > logs/process_${GPU_ID}.log &
done

# 4. Wait for all background processes to finish
wait

echo "All concurrent processes finished."
echo "Job finished. Total throughput must be calculated by combining the results in logs/process_0.log and logs/process_1.log."