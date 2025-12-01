# 🚀 Documenting BERT Inference Benchmarking on HPC Cluster

## 1. Introduction: What We Tested

This document explains the process used to successfully benchmark the performance of the **BERT** language model on a SLURM-managed High-Performance Computing (HPC) cluster and to measure the scaling efficiency across multiple GPUs.

### 1.1 What is BERT?

**BERT** stands for **Bidirectional Encoder Representations from Transformers**. It is a foundational deep learning model developed by Google for **Natural Language Processing (NLP)**.

* **Bidirectional Context:** Unlike older models, BERT reads text from **both the left and the right** sides simultaneously, enabling a deep, contextual understanding of language.
* **Architecture:** It is built on the **Transformer** architecture, relying on **Self-Attention** mechanisms to weigh word importance. 

[Image of Transformer architecture encoder stack]

* **Goal:** We focused on **Inference Benchmarking**, measuring the speed (**throughput**) at which the pre-trained `bert-large-uncased` model processes data.

---

## 1.2 Environment Setup (Prerequisites)

To replicate this test, the following packages are required in a Conda environment with the correct **CUDA drivers** installed on the compute node.

#### Complete `requirements.txt`

The following content should be saved into a file named **`requirements.txt`** in your working directory.

```text
# requirements.txt for BERT GPU Benchmarking

# PyTorch and related packages (Must match system's CUDA version, e.g., CUDA 11.8)
torch>=2.0.0
torchvision
torchaudio

# Hugging Face Libraries for BERT
transformers>=4.0.0
datasets

# Utility packages
numpy
pandas
tqdm

2. Execution Commands & Workflow 💻

Follow these commands on the HPC cluster's login node to set up the environment and submit the benchmarking job.

Step 1: Create and Activate the Conda Environment

Use the Conda module available on the cluster to create a dedicated environment named bert_bench.
Bash

# Load the necessary modules (may vary by cluster configuration)
module load anaconda
module load cuda/11.8 # Example: use the CUDA version installed on the compute nodes

# 1. Create a new Conda environment
conda create -n bert_bench python=3.10 -y

# 2. Activate the new environment
conda activate bert_bench

Step 2: Install Dependencies

Ensure your requirements.txt file is present, then install the packages.
Bash

# 3. Install the required packages from requirements.txt
pip install -r requirements.txt

Step 3: Prepare and Submit the SLURM Job

Ensure your Python benchmarking script (e.g., benchmark.py) and your SLURM submission script (e.g., run_slurm.sh) are in your working directory.
Bash

# 4. Submit the SLURM job
sbatch run_slurm.sh

Step 4: Monitor and Review Results

After submission, monitor the job queue and check the output files.
Bash

# Check the status of your job (replace <USER_NAME> with your cluster username)
squeue -u <USER_NAME>

# Once complete, review the output logs (e.g., slurm-####.out)
cat slurm-####.out