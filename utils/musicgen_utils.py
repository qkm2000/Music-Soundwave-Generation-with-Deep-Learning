import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import T5Tokenizer
from encodec import EncodecModel


class EarconDataset(Dataset):
    """
    Custom PyTorch Dataset for Earcon Generation
    """

    def __init__(self, dataframe):
        """
        Initialize the dataset

        Args:
            dataframe (pd.DataFrame): Input dataframe with required columns
        """
        # Validate required columns
        required_cols = ['image_features', 'earcon_features', 'roundness', 'image_tag']
        for col in required_cols:
            if col not in dataframe.columns:
                raise ValueError(f"Missing required column: {col}")

        # Preload all data into memory
        self.image_features = np.stack(dataframe['image_features'].values)
        self.earcon_features = np.stack(dataframe['earcon_features'].values)
        self.roundness_values = np.stack(dataframe['roundness'].values)

        # Tokenize and pad 'image_tag' column
        tokenizer = T5Tokenizer.from_pretrained('t5-small')
        self.image_tags = dataframe['image_tag'].apply(
            lambda x: tokenizer.encode(x, max_length=10, padding='max_length', truncation=True)
        ).values
        self.image_tags = np.stack(self.image_tags)

        # Compute normalization statistics and normalize in-place
        # self.image_mean = self.image_features.mean(axis=0)
        # self.image_std = self.image_features.std(axis=0)
        # self.audio_mean = self.earcon_features.mean(axis=0)
        # self.audio_std = self.earcon_features.std(axis=0)

        # self.image_features = (self.image_features - self.image_mean) / (self.image_std + 1e-7)
        # self.earcon_features = (self.earcon_features - self.audio_mean) / (self.audio_std + 1e-7)

        # Convert everything to PyTorch tensors upfront
        self.image_features = torch.tensor(self.image_features, dtype=torch.float32)
        self.earcon_features = torch.tensor(self.earcon_features, dtype=torch.float32)
        self.roundness_values = torch.tensor(self.roundness_values, dtype=torch.float32).unsqueeze(1)
        self.image_tags = torch.tensor(self.image_tags, dtype=torch.float32)

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
        return {
            'image_features': self.image_features[idx],
            'roundness': self.roundness_values[idx],
            'earcon_features': self.earcon_features[idx],
            'image_tag': self.image_tags[idx]
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


def train_musicgen_model(
    model,
    train_dataloader,
    val_dataloader,
    encodec_model=None,  # Encodec model for feature extraction
    test_dataloader=None,
    epochs=10,
    learning_rate=1e-4,
    weight_decay=1e-2,
    patience=3,
    optimizer=None,
    scheduler=None,
    device="cuda" if torch.cuda.is_available() else "cpu",
    accumulation_steps=None,
):
    """
    Train the MusicgenForImageLM model with earcon feature matching.

    Args:
        model (MusicgenForImageLM): The model to train
        train_dataloader (DataLoader): Training data loader
        val_dataloader (DataLoader): Validation data loader
        encodec_model (EncodecModel): Encodec model for audio feature extraction
        test_dataloader (DataLoader, optional): Test data loader
        epochs (int, optional): Number of training epochs. Defaults to 10.
        learning_rate (float, optional): Learning rate for AdamW optimizer. Defaults to 1e-4.
        weight_decay (float, optional): Weight decay for regularization. Defaults to 1e-2.
        patience (int, optional): Patience for early stopping. Defaults to 3.
        device (torch.device, optional): Device to train on. Defaults to cuda if available.
        accumulation_steps (int, optional): Gradient accumulation steps. Defaults to 1.
        feature_loss_weight (float, optional): Weight for feature similarity loss. Defaults to 1.0.

    Returns:
        dict: Training results containing best validation loss and final test metrics
    """
    # Set up accumulation steps
    accumulation_steps = max(1, accumulation_steps)

    # Set up device
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Set up Encodec model
    if encodec_model is None:
        encodec_model = EncodecModel.encodec_model_24khz().to(device)

    # Move models to device
    model.to(device)
    encodec_model.to(device)

    # Prepare optimizer and scheduler
    if optimizer is None:
        optimizer = AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
    # if scheduler is None:
    #     scheduler = ReduceLROnPlateau(
    #         optimizer,
    #         mode='min',
    #         factor=0.5,
    #         patience=1,
    #         verbose=True
    #     )

    # Early stopping setup
    best_val_loss = float('inf')
    epochs_no_improve = 0
    best_model_state = None

    # Training loop
    results = {
        'train_losses': [],
        'val_losses': [],
        'lr_rates': []
    }

    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        optimizer.zero_grad()

        train_progress = tqdm(
            train_dataloader,
            desc=f'Epoch {epoch+1}/{epochs}',
            unit='batch'
        )

        for batch_idx, batch in enumerate(train_progress):
            # Unpack batch
            image_features = batch['image_features'].to(device)
            roundness = batch['roundness'].to(device)
            target_earcon_features = batch['earcon_features'].to(device)

            # Training step
            outputs = model(
                image_features=image_features,
                roundness=roundness,
                labels=target_earcon_features,
            )
            loss = outputs.loss

            # Combine losses
            loss.backward()
            optimizer.step()

            # train_loss += loss.item() * accumulation_steps
            train_loss += loss.item()
            train_progress.set_postfix({
                'loss': train_loss / (batch_idx + 1),
            })

        # Validation phase (similar structure to training phase)
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            count = 1
            val_progress = tqdm(
                val_dataloader,
                desc='Validation',
                unit='batch'
            )
            for batch in val_progress:
                image_features = batch['image_features'].to(device)
                roundness = batch['roundness'].to(device)
                target_earcon_features = batch['earcon_features'].to(device)

                outputs = model(
                    image_features=image_features,
                    roundness=roundness,
                    labels=target_earcon_features,
                )
                loss = outputs.loss

                # Combine losses
                val_loss += loss.item()
                val_progress.set_postfix({
                    'loss': val_loss / count,
                })
                count += 1

        # Normalize losses
        train_loss /= len(train_dataloader)
        val_loss /= len(val_dataloader)

        # Store results
        results['train_losses'].append(train_loss)
        results['val_losses'].append(val_loss)
        results['lr_rates'].append(optimizer.param_groups[0]['lr'])

        # Learning rate scheduling
        if scheduler is not None:
            scheduler.step(val_loss)

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            best_model_state = model.state_dict().copy()
        else:
            epochs_no_improve += 1

        # Print epoch summary
        print(f'Epoch {epoch+1}: Train Loss: {train_loss:.4f}, '
              f'Val Loss: {val_loss:.4f}, '
              f'LR: {optimizer.param_groups[0]["lr"]}')

        # Early stopping check
        if epochs_no_improve >= patience:
            print(f'Early stopping triggered after {epoch+1} epochs')
            break

    # Restore best model
    if best_model_state:
        model.load_state_dict(best_model_state)

    # Test phase (optional)
    if test_dataloader:
        model.eval()
        test_loss = 0.0

        with torch.no_grad():
            test_progress = tqdm(
                test_dataloader,
                desc='Testing',
                unit='batch'
            )
            for batch in test_progress:
                image_features = batch['image_features'].to(device)
                roundness = batch['roundness'].to(device)
                target_earcon_features = batch['earcon_features'].to(device)

                outputs = model(
                    image_features=image_features,
                    roundness=roundness,
                    labels=target_earcon_features,
                )
                loss = outputs.loss

                # Combine losses
                test_loss += loss.item()
                test_progress.set_postfix({
                    'loss': test_loss,
                })

        test_loss /= len(test_dataloader)
        results['test_loss'] = test_loss
        print(f'Test Loss: {test_loss:.4f}')

    return results
