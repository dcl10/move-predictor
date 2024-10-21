import torch
import torch.nn as nn


class NeuralNetwork(nn.Module):
    def __init__(
        self, vocab_size: int, embedding_dim: int, hidden_dim: int, output_dim: int
    ) -> None:
        super().__init__()

        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        # LSTM layer
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)

        # Fully connected layer
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Convert input to embedding
        embedded = self.embedding(x)

        # Pass embedding to LSTM
        lstm_out, (hidden, cell) = self.lstm(embedded)

        # Get the output of the LSTM
        output = self.fc(lstm_out[:, -1, :])

        return torch.log_softmax(output, dim=1)
