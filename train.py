import pytorch_lightning as pl

from .data import MoveDataModule
from .models import LSTMModel


# Training script
if __name__ == "__main__":
    file_path = "your_text_file.txt"  # Path to your text file
    seq_length = 100  # Length of each sequence
    batch_size = 32  # Batch size for training
    embedding_dim = 64
    hidden_dim = 128
    vocab_size = 50  # This will be determined after loading data
    num_epochs = 10

    # Prepare the data module
    text_data_module = MoveDataModule(file_path, seq_length, batch_size)
    text_data_module.prepare_data()

    # Extract the vocabulary size based on the char_to_idx map
    vocab_size = len(text_data_module.char_to_idx)

    # Define the model
    model = LSTMModel(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        output_dim=vocab_size,
    )

    # Trainer
    trainer = pl.Trainer(max_epochs=num_epochs)

    # Train the model
    trainer.fit(model, datamodule=text_data_module)
