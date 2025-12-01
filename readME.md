## 1. 📖 Introduction: What We Tested

This document explains the process used to successfully benchmark the performance of the **BERT** language model on a SLURM-managed High-Performance Computing (HPC) cluster and to measure the scaling efficiency across multiple GPUs.

### 1.1 What is BERT?

**BERT** stands for **Bidirectional Encoder Representations from Transformers**. It is a foundational deep learning model developed by Google for **Natural Language Processing (NLP)**.

* **Bidirectional Context:** Unlike older models, BERT reads text from **both the left and the right** sides simultaneously, enabling a deep, contextual understanding of language.
* **Architecture:** It is built on the **Transformer** architecture, relying on **Self-Attention** mechanisms to weigh word importance. 

[Image of Transformer architecture encoder stack]

* **Goal:** We focused on **Inference Benchmarking**, measuring the speed (**throughput**) at which the pre-trained `bert-large-uncased` model processes data.


---

## 2. 🏃 Execution Commands & Workflow

Follow these commands on the HPC cluster's login node to set up the environment and submit the benchmarking job.

### Step 1: Create and Activate the Conda Environment

Use the Conda module available on the cluster to create a dedicated environment named `bert_bench`.

```bash
# Load the necessary modules (may vary by cluster configuration)
module load cuda/11.8 # Example: use the CUDA version installed on the compute nodes

# 1. Create a new Conda environment
conda create -n bert python=3.10 -y

# 2. Activate the new environment
conda activate bert
```
### Step 2: Install Dependencies

```bash
# 3. Install the required packages from requirements.txt
pip install -r requirements.txt
```
### Step3: Prepare and submit the SLURM job

```bash
sbatch BERT_single_GPU.sh
sbatch BERT_two_GPU.sh

```
