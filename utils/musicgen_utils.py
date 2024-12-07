import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from transformers import AdamW, get_linear_schedule_with_warmup
from tqdm import tqdm

from utils.musicgen_model import CustomMusicGenModel, earcon_generation_loss, save_model


class EarconDataset(Dataset):
    """
    Custom PyTorch Dataset for Earcon Generation

    Args:
        dataframe (pd.DataFrame): DataFrame containing:
            - 'image_features': CLIP image vector (512-dim)
            - 'earcon_features': Max-pooled earcon vector (512-dim)
            - 'roundness': Numerical representation of pseudoword roundness
    """

    def __init__(self, dataframe):
        """
        Initialize the dataset

        Args:
            dataframe (pd.DataFrame): Input dataframe with required columns
        """
        # Validate required columns
        required_cols = ['image_features', 'earcon_features', 'roundness']
        for col in required_cols:
            if col not in dataframe.columns:
                raise ValueError(f"Missing required column: {col}")

        # Convert to numpy arrays for efficient indexing
        self.image_featuress = np.stack(dataframe['image_features'].values)
        self.earcon_featuress = np.stack(dataframe['earcon_features'].values)
        self.roundness_values = dataframe['roundness'].values

        # Compute normalization statistics (optional, but recommended)
        self.image_mean = self.image_featuress.mean(axis=0)
        self.image_std = self.image_featuress.std(axis=0)
        self.earcon_mean = self.earcon_featuress.mean(axis=0)
        self.earcon_std = self.earcon_featuress.std(axis=0)

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
                  roundness value, and earcon vector
        """
        # Normalize vectors (optional, but helps with training stability)
        normalized_image_features = (
            (self.image_featuress[idx] - self.image_mean) /
            (self.image_std + 1e-7)  # Small epsilon to prevent division by zero
        )

        normalized_earcon_features = (
            (self.earcon_featuress[idx] - self.earcon_mean) /
            (self.earcon_std + 1e-7)
        )

        return {
            'image_features': torch.tensor(normalized_image_features, dtype=torch.float32),
            'roundness': torch.tensor(self.roundness_values[idx], dtype=torch.float32),
            'earcon_features': torch.tensor(normalized_earcon_features, dtype=torch.float32)
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


def train_earcon_generation_model(
    model,
    train_dataloader,
    val_dataloader,
    optimizer,
    device="cuda" if torch.cuda.is_available() else "cpu",
    num_epochs=10,
    save_path='outputs/',
    version="MusicGenModel_V2_01",
):
    """
    Train the custom MusicGen model for earcon generation

    Args:
        model (CustomMusicGenModel): Model to train
        train_dataloader (DataLoader): Training data loader
        val_dataloader (DataLoader): Validation data loader
        device (torch.device): Device to train on
        learning_rate (float): Learning rate
        num_epochs (int): Number of training epochs
        save_path (str): Path to save best model
        version (str): Model version identifier
    """
    # Prepare optimizer and schedule
    total_steps = len(train_dataloader) * num_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps
    )

    # Move model to device
    model.to(device)

    # Tracking variables
    best_val_loss = float('inf')

    # Training loop
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0

        progress_bar = tqdm(
            train_dataloader, desc=f'Epoch {epoch+1}/{num_epochs}')
        for batch in progress_bar:
            # Move batch to device
            image_features = batch['image_features'].to(device)
            roundness = batch['roundness'].to(device)
            earcon_features = batch['earcon_features'].to(device)

            # Zero gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(
                image_features,
                roundness,
            )

            # Calculate loss
            loss = earcon_generation_loss(
                outputs.hidden_states,
                earcon_features,
                image_features,
                roundness
            )

            # Backward pass
            loss.backward()

            # Clip gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            # Optimizer step
            optimizer.step()
            scheduler.step()

            # Update tracking
            train_loss += loss.item()

            # Update progress bar
            progress_bar.set_postfix({'loss': loss.item()})

        # Validation phase
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_dataloader:
                # Move batch to device
                image_features = batch['image_features'].to(device)
                roundness = batch['roundness'].to(device)
                earcon_features = batch['earcon_features'].to(device)

                # Forward pass
                outputs = model(
                    image_features,
                    roundness,
                )

                # Calculate loss
                loss = earcon_generation_loss(
                    outputs.hidden_states,
                    earcon_features,
                    image_features,
                    roundness
                )

                val_loss += loss.item()

        # Average losses
        avg_train_loss = train_loss / len(train_dataloader)
        avg_val_loss = val_loss / len(val_dataloader)

        print(f'Epoch {epoch+1:>3}/{num_epochs:>3}: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}')

        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            save_model(model, f"{save_path}+{version}.pth")
            print(f'Saved new best model with validation loss: {best_val_loss:.4f}')

    return model
