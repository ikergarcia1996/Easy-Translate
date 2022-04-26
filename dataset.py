from typing import List, TextIO, Dict, Optional
import torch
from torch.utils.data import IterableDataset
from torch.utils.data.dataset import T_co


def blocks(files, size=65536):
    while True:
        b = files.read(size)
        if not b:
            break
        yield b


def count_lines(input_path: str) -> int:
    with open(input_path, "r", encoding="utf8") as f:
        return sum(bl.count("\n") for bl in blocks(f))


class DatasetReader(IterableDataset):
    def __init__(self, filename, tokenizer, max_length=128):
        self.filename = filename
        self.tokenizer = tokenizer
        self.max_length = max_length

    def preprocess(self, text: str):
        return self.tokenizer(
            text.rstrip().strip(),
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

    def __iter__(self):
        file_itr = open(self.filename, "r")
        mapped_itr = map(self.preprocess, file_itr)
        return mapped_itr


def collate_function(batch: List[T_co]) -> Dict[str, torch.Tensor]:
    return {
        "input_ids": torch.stack([item["input_ids"][0] for item in batch]),
        "attention_mask": torch.stack([item["attention_mask"][0] for item in batch]),
    }


def get_dataloader(
    filename: str, tokenizer: str, batch_size: int, max_length: int
) -> torch.utils.data.DataLoader:
    dataset = DatasetReader(filename, tokenizer, max_length)
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=collate_function,
    )
