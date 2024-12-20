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
    image_processor,
    train_dataloader,
    val_dataloader,
    encodec_model=None,
    test_dataloader=None,
    epochs=10,
    model_learning_rate=1e-4,
    processor_learning_rate=1e-4,
    weight_decay=1e-5,
    patience=3,
    model_optimizer=None,
    processor_optimizer=None,
    model_scheduler=None,
    processor_scheduler=None,
    device=None,
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
        feature_loss_weight (float, optional): Weight for feature similarity loss. Defaults to 1.0.

    Returns:
        dict: Training results containing best validation loss and final test metrics
    """

    # Set up device
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Set up Encodec model
    model.to(device)
    image_processor.to(device)
    if encodec_model is None:
        encodec_model = EncodecModel.encodec_model_24khz().to(device)

    # Move models to device
    model.to(device)
    encodec_model.to(device)

    # Prepare optimizer and scheduler
    if model_optimizer is None:
        model_optimizer = AdamW(
            model.parameters(),
            lr=model_learning_rate,
            weight_decay=weight_decay
        )
    if processor_optimizer is None:
        processor_optimizer = AdamW(
            image_processor.parameters(),
            lr=processor_learning_rate,
            weight_decay=weight_decay
        )

    # Early stopping setup
    best_val_loss = float('inf')
    epochs_no_improve = 0

    # Training loop
    results = {
        'train_losses': [],
        'val_losses': [],
        'lr_rates': []
    }

    for epoch in range(epochs):
        # Training phase
        model.train()
        image_processor.train()
        train_loss = 0.0

        train_progress = tqdm(
            train_dataloader,
            desc=f'Epoch {epoch+1}/{epochs}',
            unit='batch',
            leave=True
        )

        for batch_idx, batch in enumerate(train_progress):
            # Unpack batch
            image_features = batch['image_features'].to(device)
            roundness = batch['roundness'].to(device)
            target_earcon_features = batch['earcon_features'].to(device)

            processed_images = image_processor(image_features, roundness)

            # print(f"input_ids.requires_grad: {processed_images['input_ids'].requires_grad}")
            # print(f"attention_mask.requires_grad: {processed_images['attention_mask'].requires_grad}")

            # Training step
            outputs = model(
                input_ids=processed_images["input_ids"],
                attention_mask=processed_images["attention_mask"],
                labels=target_earcon_features,
            )
            loss = outputs.loss

            # Backpropagation
            model_optimizer.zero_grad()
            processor_optimizer.zero_grad()
            loss.backward()
            model_optimizer.step()
            processor_optimizer.step()

            # check for gradients
            # for name, param in model.named_parameters():
            #     if param.requires_grad:
            #         if param.grad is None:
            #             print(f"None gradient for {name}")
            # for name, param in image_processor.named_parameters():
            #     if param.requires_grad:
            #         if param.grad is None:
            #             print(f"None gradient for {name}")

            train_loss = train_loss + loss.item()
            train_progress.set_postfix({
                'loss': (train_loss / (batch_idx + 1)),
            })

        # Validation phase (similar structure to training phase)
        model.eval()
        image_processor.eval()
        val_loss = 0.0

        with torch.no_grad():
            val_progress = tqdm(
                val_dataloader,
                desc='Validation',
                unit='batch',
                leave=True
            )
            for batch_idx, batch in enumerate(val_progress):
                # Unpack batch
                image_features = batch['image_features'].to(device)
                roundness = batch['roundness'].to(device)
                target_earcon_features = batch['earcon_features'].to(device)

                processed_images = image_processor(image_features, roundness)

                outputs = model(
                    input_ids=processed_images["input_ids"],
                    attention_mask=processed_images["attention_mask"],
                    labels=target_earcon_features,
                )
                loss = outputs.loss

                # Combine losses
                val_loss += loss.item()
                val_progress.set_postfix({
                    'loss': val_loss / (batch_idx + 1),
                })

        # Normalize losses
        train_loss /= len(train_dataloader)
        val_loss /= len(val_dataloader)

        # Store results
        results['train_losses'].append(train_loss)
        results['val_losses'].append(val_loss)
        results['lr_rates'].append(model_optimizer.param_groups[0]['lr'])

        # Learning rate scheduling
        if model_scheduler is not None:
            model_scheduler.step(val_loss)
        if processor_scheduler is not None:
            processor_scheduler.step(val_loss)

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            best_model_states = {
                'musicgen': model.state_dict().copy(),
                'processor': image_processor.state_dict().copy()
            }
            print('This is the best model thus far, saving...')
        else:
            epochs_no_improve += 1

        # Print epoch summary
        print(
            f'Epoch {epoch+1}: Train Loss: {train_loss:.4f}, '
            f'Val Loss: {val_loss:.4f}, '
            f'Mod LR: {model_optimizer.param_groups[0]["lr"]}, '
            f'Pro LR: {processor_optimizer.param_groups[0]["lr"]}'
        )

        # Early stopping check
        if epochs_no_improve >= patience:
            print(f'Early stopping triggered after {epoch+1} epochs')
            break

    # Restore best model
    if best_model_states:
        model.load_state_dict(best_model_states["musicgen"])
        image_processor.load_state_dict(best_model_states["processor"])

    # Test phase (optional)
    if test_dataloader:
        model.eval()
        image_processor.eval()
        test_loss = 0.0

        with torch.no_grad():
            test_progress = tqdm(
                test_dataloader,
                desc='Testing',
                unit='batch',
                leave=True
            )
            for batch in test_progress:
                image_features = batch['image_features'].to(device)
                roundness = batch['roundness'].to(device)
                target_earcon_features = batch['earcon_features'].to(device)

                processed_images = image_processor(image_features, roundness)

                outputs = model(
                    input_ids=processed_images["input_ids"],
                    attention_mask=processed_images["attention_mask"],
                    labels=target_earcon_features,
                )
                loss = outputs.loss

                test_loss += loss.item()
                test_progress.set_postfix({
                    'loss': test_loss / len(test_dataloader),
                })

        test_loss /= len(test_dataloader)
        results['test_loss'] = test_loss
        print(f'Test Loss: {test_loss:.4f}')

    return results
