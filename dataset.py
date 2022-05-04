from torch.utils.data import IterableDataset


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
        self.current_line = 0

    def preprocess(self, text: str):
        self.current_line += 1
        text = text.rstrip().strip()
        if len(text) == 0:
            print(f"Warning: empty sentence at line {self.current_line}")
        return self.tokenizer(
            text,
            padding=False,
            truncation=True,
            max_length=self.max_length,
            return_tensors=None,
        )

    def __iter__(self):
        file_itr = open(self.filename, "r")
        mapped_itr = map(self.preprocess, file_itr)
        return mapped_itr
