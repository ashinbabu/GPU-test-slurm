# 🚀 Documenting BERT Inference Benchmarking on HPC Cluster

## 1. Introduction: What We Tested

This document explains the process used to successfully benchmark the performance of the **BERT** language model on a SLURM-managed High-Performance Computing (HPC) cluster and to measure the scaling efficiency across multiple GPUs.

### 1.1 What is BERT?

**BERT** stands for **Bidirectional Encoder Representations from Transformers**. It is a foundational deep learning model developed by Google for **Natural Language Processing (NLP)**.

* **Bidirectional Context:** Unlike older models, BERT reads text from **both the left and the right** sides simultaneously, enabling a deep, contextual understanding of language.
* **Architecture:** It is built on the 

[Image of Transformer architecture encoder stack]
 **Transformer** architecture, relying on **Self-Attention** mechanisms to weigh word importance.
* **Goal:** We focused on **Inference Benchmarking**, measuring the speed (throughput) at which the pre-trained `bert-large-uncased` model processes data.

### 1.2 Environment Setup (Prerequisites)

To replicate this test, the following packages are required in a Conda environment with the correct **CUDA drivers** installed on the compute node.

#### `requirements.txt`

```text
# requirements.txt for BERT GPU Benchmarking

# PyTorch and related packages (Must match system's CUDA version)
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