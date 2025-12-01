#!/usr/bin/env python
import time
import torch
import os
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# --- Configuration ---
MODEL_NAME = "bert-large-uncased"
MAX_LENGTH = 512
# Choose a large batch size to maximize GPU utilization
BATCH_SIZE = 128 
# Set a number of steps for a short, intensive run
TOTAL_STEPS = 200
# --- End Configuration ---

def run_single_gpu_test():
    # 1. Device Setup
    if not torch.cuda.is_available():
        print("ERROR: CUDA is not available. Running on CPU (will be slow).")
        device = torch.device("cpu")
    else:
        # Use the first available GPU
        device = torch.device("cuda:0")
        print(f"Running test on GPU: {torch.cuda.get_device_name(0)}")
    
    # Set the CUDA_VISIBLE_DEVICES environment variable just in case, though usually 
    # unnecessary for a simple single-GPU run.
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    # 2. Load Model and Tokenizer
    print(f"Loading model {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
    model.to(device)
    model.eval() # Set model to evaluation mode for inference

    # 3. Create Dummy Data
    # Create a list of dummy sentences for tokenization
    dummy_sentences = [
        f"This is a long test sentence for the BERT large model. We want to stress the GPU memory and compute units. Step: {i}"
        for i in range(BATCH_SIZE)
    ]
    
    # Tokenize the data once and move it to the device
    inputs = tokenizer(
        dummy_sentences,
        max_length=MAX_LENGTH,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    ).to(device)
    
    # 4. Intensive Inference Loop
    start_time = time.time()
    
    print(f"Starting {TOTAL_STEPS} inference steps with batch size {BATCH_SIZE}...")
    
    # Use torch.no_grad() for pure inference to save memory and time
    with torch.no_grad():
        for step in range(1, TOTAL_STEPS + 1):
            # Forward pass
            outputs = model(**inputs)
            
            # Simple operation to ensure the output is fully computed
            _ = outputs.logits.sum() 
            
            if step % 50 == 0:
                print(f"Completed step {step}/{TOTAL_STEPS}. Time elapsed: {time.time() - start_time:.2f}s")
            
            # Synchronize the GPU to ensure accurate timing/completion tracking
            torch.cuda.synchronize(device)

    end_time = time.time()
    total_samples = TOTAL_STEPS * BATCH_SIZE
    throughput = total_samples / (end_time - start_time)
    
    print("\n--- Benchmark Results ---")
    print(f"Total time taken: {end_time - start_time:.2f} seconds")
    print(f"Total samples processed: {total_samples}")
    print(f"Throughput: {throughput:.2f} samples/sec")

if __name__ == "__main__":
    run_single_gpu_test()