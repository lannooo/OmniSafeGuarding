
import pandas as pd
from .base import SafetyDataset
from typing import Dict, Iterable

SPLITS = {
    '330k_train': 'round0/330k/train.jsonl.xz',
    '330k_test': 'round0/330k/test.jsonl.xz',
    '30k_train': 'round0/30k/train.jsonl.gz',
    '30k_test': 'round0/30k/test.jsonl.gz'
}

class BeaverTails(SafetyDataset):
    def __init__(self, split: str = "30k_test"):
        if split not in SPLITS:
            raise KeyError(f"Unknown split '{split}'. Available: {list(SPLITS.keys())}")
        path = "hf://datasets/PKU-Alignment/BeaverTails/" + SPLITS[split]
        self.df = pd.read_json(path, lines=True)

    def __iter__(self):
        for idx, (_, row) in enumerate(self.df.iterrows()):
            
            yield {
                "id": idx,
                "prompt_1": row["prompt"],
                "prompt_2": row["response"],
            }

    def __len__(self):
        return len(self.df)