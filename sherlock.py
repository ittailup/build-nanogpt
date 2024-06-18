import torch, torchaudio
from speech_tokenizer import SpeechTokenizer
import numpy as np
from tqdm import tqdm
import itertools
from pathlib import Path
import os
from huggingface_hub import snapshot_download

snapshot_download(repo_id="ittailup/snac", local_dir='./snac', repo_type='dataset')

device = "cpu"
if torch.cuda.is_available():
    device = "cuda"
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = "mps"
print(f"using device: {device}")

def batch_list(lst, batch_size):
    it = iter(lst)
    return iter(lambda: list(itertools.islice(it, batch_size)), [])

Path('./snacdata').mkdir(parents=True, exist_ok=True)

tokenizer = SpeechTokenizer(device=device)

batch_size = 2
print("batch size:", batch_size)

data_path = "./snac"
file_ext = 'opus'

from datasets import load_dataset
dataset = load_dataset("./snac", split="train", num_proc=30)

current_batch = []
batch_size = 16
batch_num = 1

for snac_tokens in dataset["snac24khz"]:
    array = np.fromstring(snac_tokens, dtype=int, sep=' ')
    current_batch.append(array)
    
    if len(current_batch) == batch_size:
        # flatten the list of arrays and add separator tokens
        flattened_batch = np.concatenate([np.append(arr, 50256) for arr in current_batch])
        np.save(f"./snacdata/snac_train_x{batch_size}_{batch_num}", flattened_batch)
        
        # reset the current batch and increment batch number
        current_batch = []
        batch_num += 1

# handle the last batch if it's not empty
if current_batch:
    flattened_batch = np.concatenate([np.append(arr, 50256) for arr in current_batch])
    np.save(f"./snacdata/snac_train_x{batch_size}_{batch_num}", flattened_batch)