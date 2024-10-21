import torch
from torch.utils.data import Dataset


class MovesDataset(Dataset):
    def __init__(self, lines: list, seq_length: int, move_to_idx: dict) -> None:
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
                line_idx = line_idx + [self.move_to_idx['<PAD>']] * (self.seq_length - len(line_idx))
            else:
                line_idx = line_idx[:self.seq_length]
        
        # Input is the sequence (all except the last char)
        input_seq = line_idx[:-1]
        # Target is the next character (all except the first char)
        target = line_idx[1:]
        
        return torch.tensor(input_seq), torch.tensor(target)
