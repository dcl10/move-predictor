import pytorch_lightning as pl
import torch
import os
from torch.utils.data import Dataset, DataLoader, random_split


class MovesDataset(Dataset):
    def __init__(
        self, lines: list, move_to_idx: dict, seq_length: int | None = None
    ) -> None:
        super().__init__()
        self.lines = lines
        self.seq_length = seq_length
        self.move_to_idx = move_to_idx
        self.vocab_size = len(move_to_idx)

        # Convert each line into a list of indices based on the char_to_idx mapping
        self.lines_idx = [self._encode_line(line) for line in lines]

    def _encode_line(self, line: str):
        """Converts a sequence of characters (line) into a list of character indices."""
        return [self.move_to_idx[char] for char in line if char in self.move_to_idx]

    def __len__(self):
        return len(self.lines_idx)

    def __getitem__(self, idx: int):
        """Return a sequence (input) and the next character (target) from a line."""
        line_idx = self.lines_idx[idx]

        # If seq_length is provided, pad or truncate the sequence
        if self.seq_length:
            if len(line_idx) < self.seq_length:
                # Padding with a special index (e.g., -1) to maintain fixed length
                line_idx = line_idx + [self.move_to_idx["<PAD>"]] * (
                    self.seq_length - len(line_idx)
                )
            else:
                line_idx = line_idx[: self.seq_length]

        # Input is the sequence (all except the last char)
        input_seq = line_idx[:-1]
        # Target is the next character (all except the first char)
        target = line_idx[1:]

        return torch.tensor(input_seq), torch.tensor(target)


class MoveDataModule(pl.LightningDataModule):
    def __init__(
        self,
        file_path: str,
        seq_length: int,
        batch_size: int,
        val_split: float = 0.1,
        test_split: float = 0.1,
    ):
        super().__init__()
        self.file_path = file_path
        self.seq_length = seq_length
        self.batch_size = batch_size
        self.val_split = val_split
        self.test_split = test_split
        self.move_to_idx = None
        self.idx_to_move = None
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.lines = []

    def prepare_data(self):
        # Read the text file, and split lines into a list of sequences
        with open(self.file_path, "r") as file:
            lines = file.read().splitlines()

        # Create move-to-index and index-to-character mappings
        moves = set()
        for line in lines:
            l = line.strip()
            new_moves = l.split(" ")
            moves.update(new_moves)

        moves = sorted(list(moves))  # convert `moves` to a sorted list

        self.move_to_idx = {move: idx for idx, move in enumerate(moves)}
        self.move_to_idx["<PAD>"] = len(self.move_to_idx)  # Optional: add padding index
        self.idx_to_move = {idx: move for move, idx in self.move_to_idx.items()}

        # Save the text lines for later use in setup
        self.lines = lines

    def setup(self, stage=None):
        # Create the dataset using the lines of text
        dataset = MovesDataset(self.lines, self.move_to_idx, seq_length=self.seq_length)

        # Split the dataset into training, validation, and test sets
        total_length = len(dataset)
        val_len = int(total_length * self.val_split)
        test_len = int(total_length * self.test_split)
        train_len = total_length - val_len - test_len

        self.train_dataset, self.val_dataset, self.test_dataset = random_split(
            dataset, [train_len, val_len, test_len]
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=_collate_fn,
        )

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size)


def _collate_fn(batch):
    # Pad sequences in a batch so that all have the same length
    inputs, targets = zip(*batch)
    inputs = torch.nn.utils.rnn.pad_sequence(
        inputs, batch_first=True, padding_value=-1
    )  # Padding input
    targets = torch.nn.utils.rnn.pad_sequence(
        targets, batch_first=True, padding_value=-1
    )  # Padding target
    return inputs, targets
