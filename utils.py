import random
import os
import torch
from torch.utils.data import Dataset
import pandas as pd
from tqdm import tqdm


class CustomDataset(Dataset):
    def __init__(self, csv_path, tokenizer):
        self.df = pd.read_csv(csv_path)
        self.tokenizer = tokenizer

        print("Computing max length in dataset...")
        max_len = 0
        for idx in tqdm(range(len(self.df))):
            row = self.df.iloc[idx]
            text = f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request. ### Instruction: {row['instruction']} ### Input: {row['input']} ### Response: {row['output']}"""
            length = len(self.tokenizer.encode(text))
            max_len = max(max_len, length)
        self.max_length = (max_len + 7) // 8 * 8
        print(f"Original max length: {max_len}")
        print(f"Padded max length: {self.max_length}")


    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        text = f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request. ### Instruction: {row['instruction']} ### Input: {row['input']} ### Response: {row['output']}"""
        encoding = self.tokenizer(
            text,
            return_tensors="pt",
            padding="max_length",      
            max_length=self.max_length
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0)
        }


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True