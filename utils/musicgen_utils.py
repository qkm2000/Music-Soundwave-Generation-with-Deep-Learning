import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from tqdm import tqdm


class EarconDataset(Dataset):
    """
    Custom PyTorch Dataset for Earcon Generation

    Args:
        dataframe (pd.DataFrame): DataFrame containing:
            - 'image_vector': CLIP image vector (512-dim)
            - 'earcon_vector': Max-pooled audio vector (512-dim)
            - 'roundness': Numerical representation of pseudoword roundness
    """

    def __init__(self, dataframe):
        """
        Initialize the dataset

        Args:
            dataframe (pd.DataFrame): Input dataframe with required columns
        """
        # Validate required columns
        required_cols = ['image_vector', 'earcon_vector', 'roundness']
        for col in required_cols:
            if col not in dataframe.columns:
                raise ValueError(f"Missing required column: {col}")

        # Convert to numpy arrays for efficient indexing
        self.image_vectors = np.stack(dataframe['image_vector'].values)
        self.earcon_vectors = np.stack(dataframe['earcon_vector'].values)
        self.roundness_values = dataframe['roundness'].values

        # Compute normalization statistics (optional, but recommended)
        self.image_mean = self.image_vectors.mean(axis=0)
        self.image_std = self.image_vectors.std(axis=0)
        self.audio_mean = self.earcon_vectors.mean(axis=0)
        self.audio_std = self.earcon_vectors.std(axis=0)

    def __len__(self):
        """
        Return the total number of samples in the dataset
        """
        return len(self.roundness_values)

    def __getitem__(self, idx):
        """
        Retrieve a single sample from the dataset

        Args:
            idx (int): Index of the sample to retrieve

        Returns:
            dict: Dictionary containing normalized image vector, 
                  roundness value, and audio vector
        """
        # Normalize vectors (optional, but helps with training stability)
        normalized_image_vector = (
            (self.image_vectors[idx] - self.image_mean) /
            (self.image_std + 1e-7)  # Small epsilon to prevent division by zero
        )

        normalized_earcon_vector = (
            (self.earcon_vectors[idx] - self.audio_mean) /
            (self.audio_std + 1e-7)
        )

        return {
            'image_vector': torch.tensor(normalized_image_vector, dtype=torch.float32),
            'roundness': torch.tensor(self.roundness_values[idx], dtype=torch.float32),
            'earcon_vector': torch.tensor(normalized_earcon_vector, dtype=torch.float32)
        }


def create_earcon_dataloaders(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    batch_size: int = 32,
    num_workers: int = 0
) -> tuple:
    """
    Create train, validation, and test DataLoaders

    Args:
        train_df (pd.DataFrame): Training dataframe
        val_df (pd.DataFrame): Validation dataframe
        test_df (pd.DataFrame): Test dataframe
        batch_size (int): Batch size for dataloaders
        num_workers (int): Number of worker processes for data loading

    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    # Create datasets
    train_dataset = EarconDataset(train_df)
    val_dataset = EarconDataset(val_df)
    test_dataset = EarconDataset(test_df)

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, val_loader, test_loader


def train_multimodal_earcon_model(
    model,
    train_dataloader,
    val_dataloader,
    optimizer,
    epochs=10,
    patience=3,
    device='cuda' if torch.cuda.is_available() else 'cpu',
    version="MusicGenModel_0x"
):
    """
    Training loop for the Multimodal Earcon Generator with validation and early stopping.

    Args:
        model: Multimodal Earcon Generator model.
        train_dataloader: DataLoader for training with image vectors, roundness values, and audio vectors.
        val_dataloader: DataLoader for validation with image vectors, roundness values, and audio vectors.
        optimizer: Optimizer for training.
        epochs: Number of training epochs.
        patience: Early stopping patience; training stops if validation loss doesn't improve for this many epochs.
        device: Training device ('cuda' or 'cpu').
    """
    model.to(device)
    model.train()

    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(epochs):
        # Training phase
        model.train()
        best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0
        train_batches = 0
        train_progress = tqdm(
            train_dataloader,
            desc=f"Epoch {epoch+1}/{epochs} - Training",
            leave=False
        )

        for batch in train_progress:
            # Unpack batch
            image_vectors = batch['image_vector'].to(device)
            roundness_values = batch['roundness'].to(device).unsqueeze(-1)  # Ensure 2D tensor
            target_earcon_vector = batch['earcon_vector'].to(device)

            # Zero gradients
            optimizer.zero_grad()

            # Compute loss
            loss, _ = model(
                image_vector=image_vectors,
                roundness_value=roundness_values,
                target_earcon_vector=target_earcon_vector
            )

            # Backpropagate
            loss.backward()
            optimizer.step()

            # Accumulate loss
            train_loss += loss.item()
            train_batches += 1
            train_progress.set_postfix({"Batch Loss": loss.item()})

        avg_train_loss = train_loss / train_batches

        # Validation phase
        model.eval()
        val_loss = 0
        val_batches = 0
        val_progress = tqdm(val_dataloader, desc=f"Epoch {epoch+1}/{epochs} - Validation", leave=False)

        with torch.no_grad():
            for batch in val_progress:
                # Unpack batch
                image_vectors = batch['image_vector'].to(device)
                roundness_values = batch['roundness'].to(device).unsqueeze(-1)  # Ensure 2D tensor
                target_earcon_vector = batch['earcon_vector'].to(device)

                # Compute loss
                loss, _ = model(
                    image_vector=image_vectors,
                    roundness_value=roundness_values,
                    target_earcon_vector=target_earcon_vector
                )

                # Accumulate loss
                val_loss += loss.item()
                val_batches += 1
                val_progress.set_postfix({"Batch Loss": loss.item()})

        avg_val_loss = val_loss / val_batches

        # Print epoch summary
        print(f"Epoch {epoch+1}/{epochs}", end=", ")
        print(f"Training Loss: {avg_train_loss:.4f}", end=", ")
        print(f"Validation Loss: {avg_val_loss:.4f}")

        # Check for early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            # Optionally, save the best model
            torch.save(model.state_dict(), rf"outputs\{version}.pth")
        else:
            patience_counter += 1
            print(f"  No improvement for {patience_counter}/{patience} epochs.")

        if patience_counter >= patience:
            print("Early stopping triggered.")
            break


def generate_earcon(
    model,
    image_vector,
    roundness_value,
    device='cuda' if torch.cuda.is_available() else 'cpu'
):
    """
    Generate an earcon from image vector and roundness value

    Args:
        model:
            Trained Multimodal Earcon Generator
        image_vector:
            Input image vector
        roundness_value:
            Roundness value for the input
        device:
            Inference device

    Returns:
        Generated audio representation
    """
    model.eval()
    with torch.no_grad():
        # Ensure inputs are correct tensor types and shapes
        image_vector = torch.tensor(
            image_vector).float().unsqueeze(0).to(device)
        roundness_value = torch.tensor(
            [[roundness_value]]).float().to(device)

        generated_audio = model(
            image_vector=image_vector,
            roundness_value=roundness_value
        )

    return generated_audio
