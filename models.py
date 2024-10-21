import torch
import torch.nn as nn
import pytorch_lightning as pl


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


# LightningModule for training the LSTM model
class LSTMModel(pl.LightningModule):
    def __init__(
        self, vocab_size: int, embedding_dim: int, hidden_dim: int, output_dim: int
    ):
        super().__init__()
        self.model = NeuralNetwork(vocab_size, embedding_dim, hidden_dim, output_dim)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        inputs, targets = batch
        logits = self(inputs)
        loss = self.loss_fn(logits, targets[:, -1])
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, targets = batch
        logits = self(inputs)
        loss = self.loss_fn(logits, targets[:, -1])
        self.log("val_loss", loss)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)
