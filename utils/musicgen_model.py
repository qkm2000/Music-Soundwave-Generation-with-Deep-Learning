import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from transformers import MusicgenDecoderConfig, MusicgenModel
from encodec import EncodecModel


class MultimodalEarconGenerator(nn.Module):
    def __init__(
        self,
        image_features_dim=512,
        freeze_musicgen_text_encoder=True,
        freeze_musicgen_decoder=True,
        freeze_encodec=True,
        num_projection_layers=3,
        fusion_hidden_dims=[512, 256]
    ):
        super(MultimodalEarconGenerator, self).__init__()

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.sample_rate = 24000
        self.target_length = 512
        self.encodec = EncodecModel.encodec_model_24khz().to(self.device)
        self.loss_fn = MultimodalEarconLoss(sample_rate=self.sample_rate)

        # Use MusicgenForConditionalGeneration
        config = MusicgenDecoderConfig(num_codebooks=1)
        self.musicgen = MusicgenModel(config)

        # freeze model parameters if needed
        self._freeze_model_parameters(
            freeze_musicgen_decoder,
            freeze_musicgen_text_encoder,
            freeze_encodec
        )

        # Dynamic image vector projection with multiple layers
        self.image_projection = self._create_projection_layers(
            image_features_dim,
            output_dim=512,
            num_layers=num_projection_layers
        )

        # Roundness input processing
        roundness_layers = nn.Sequential(
            nn.Linear(1, 64),
            nn.SiLU(),
            nn.BatchNorm1d(64),
            nn.Linear(64, 128),
            nn.LayerNorm(128)
        )

        self.roundness_projection = nn.Sequential(*roundness_layers)

        # Fusion layer with multiple hidden layers
        fusion_layers = []
        current_dim = 512 + 128  # image projection + roundness projection
        for hidden_dim in fusion_hidden_dims:
            fusion_layers.extend([
                nn.Linear(current_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim)
            ])
            current_dim = hidden_dim

        fusion_layers.extend([
            nn.Linear(current_dim, 512),
            nn.ReLU(),
            nn.LayerNorm(512)
        ])

        self.modality_fusion = nn.Sequential(*fusion_layers)

        self.float_to_int_converter = FloatToIntegerLayer(input_dim=512)

        self.musicgen_to_encodec_channels = nn.Linear(1024, 1)

    def forward(
        self,
        image_features,
        roundness_value,
        target_earcon_features=None
    ):
        # Ensure inputs are on the correct device and require gradients
        image_features = image_features.to(self.device)
        roundness_value = roundness_value.to(self.device)

        # Project image vector
        projected_image = self.image_projection(image_features.float())

        # Project roundness value
        projected_roundness = self.roundness_projection(roundness_value.float())

        # Fuse modalities
        fused_input = self.modality_fusion(
            torch.cat([projected_image, projected_roundness], dim=-1)
        )

        # Pass features to MusicGen
        # Convert to int tensor instead of float
        int_fused_input = self.float_to_int_converter(fused_input)
        audio_values = self.musicgen.decoder(int_fused_input)["last_hidden_state"]
        audio_values = self.musicgen_to_encodec_channels(audio_values)
        audio_values = audio_values.permute(0, 2, 1)

        # If training, compute loss
        if target_earcon_features is not None:
            waveform = self.process_audio_tensor(audio_values)

            # Calculate Loss
            loss = self.loss_fn(
                waveform.to(self.device),
                target_earcon_features.to(self.device)
            )

            return loss, audio_values

        return audio_values

    def process_audio_tensor(self, audio_tensor):
        # Ensure the input is a tensor and on the correct device
        if not isinstance(audio_tensor, torch.Tensor):
            audio_tensor = torch.tensor(audio_tensor)
        audio_tensor = audio_tensor.to(self.device)

        # Ensure requires_grad is True
        if not audio_tensor.requires_grad:
            audio_tensor.requires_grad_(True)

        # Process with Encodec
        with torch.set_grad_enabled(True):
            audio_features = self.encodec(audio_tensor)

            target_length = 512
            encoded_frames = self.encodec.encode(audio_features)
            compressed_features = encoded_frames[0][0].to(self.device)  # Take the first codebook
            # print(f"compressed_features shape = {compressed_features.shape}")

            # truncate and pad
            length = compressed_features.shape[2]
            if length > target_length:
                compressed_features = compressed_features[:, :, :target_length].to(self.device)
            else:
                pad = torch.zeros((compressed_features.shape[0], compressed_features.shape[1], target_length - length)).to(self.device)
                compressed_features = torch.cat((compressed_features, pad), dim=2).to(self.device)

            # Ensure the compressed features require gradients
            compressed_features = compressed_features.detach().requires_grad_(True)

        return compressed_features

    def _create_projection_layers(self, input_dim, output_dim, num_layers):
        """Create projection layers with residual connections"""
        layers = []
        current_dim = input_dim

        for _ in range(num_layers - 1):
            next_dim = min(current_dim * 2, output_dim)
            block = nn.Sequential(
                nn.Linear(current_dim, next_dim),
                nn.SiLU(),
                nn.BatchNorm1d(next_dim)
            )

            # Residual connection if dimensions match
            if current_dim == next_dim:
                layers.append(ResidualBlock(block))
            else:
                layers.append(block)

            current_dim = next_dim

        # Final projection
        layers.append(nn.Sequential(
            nn.Linear(current_dim, output_dim),
            nn.LayerNorm(output_dim)
        ))

        return nn.Sequential(*layers)

    def _freeze_model_parameters(
        self,
        freeze_decoder,
        freeze_text_encoder,
        freeze_encodec
    ):
        """Helper method to freeze model parameters"""
        if freeze_decoder:
            for param in self.musicgen.decoder.parameters():
                param.requires_grad = False

        # if freeze_text_encoder:
        #     for param in self.musicgen.text_encoder.parameters():
        #         param.requires_grad = False

        # for param in self.musicgen.audio_encoder.parameters():
        #     param.requires_grad = False

        if freeze_encodec:
            for param in self.encodec.parameters():
                param.requires_grad = False


class FloatToIntegerLayer(nn.Module):
    def __init__(self, input_dim):
        super(FloatToIntegerLayer, self).__init__()
        # Learnable scaling and offset parameters
        self.scale = nn.Parameter(torch.ones(input_dim))
        self.offset = nn.Parameter(torch.zeros(input_dim))

    def forward(self, x):
        # Scale and offset the input
        scaled_x = x * self.scale + self.offset

        # Round to nearest integer and clamp to ensure valid integer range
        int_x = torch.clamp(
            torch.round(scaled_x),
            min=0,
            max=self.get_max_vocab_size()
        )

        return int_x.long()

    def get_max_vocab_size(self):
        # You might want to adjust this based on your specific use case
        # This is typically the vocabulary size of your embedding layer
        return 1024  # Example value, adjust as needed


class ResidualBlock(nn.Module):
    def __init__(self, block):
        super(ResidualBlock, self).__init__()
        self.block = block

    def forward(self, x):
        return x + self.block(x)


class LogTransform(nn.Module):
    """
    Custom module to apply logarithmic transformation with a small epsilon
    to prevent log(0)
    """
    def __init__(self, epsilon=1e-10):
        super(LogTransform, self).__init__()
        self.epsilon = epsilon

    def forward(self, x):
        return torch.log(x + self.epsilon)


class MultimodalEarconLoss(nn.Module):
    def __init__(
        self,
        sample_rate=24000
    ):
        """
        PyTorch-compatible composite loss function for multimodal earcon generation.

        Args:
            mse_weight (float): Weight for Mean Squared Error loss
            spectral_weight (float): Weight for Spectral Loss
            temporal_weight (float): Weight for Temporal Coherence Loss
            sample_rate (int): Audio sample rate
        """
        super(MultimodalEarconLoss, self).__init__()

        # Mel spectrogram transformation
        self.mel_spectrogram = nn.Sequential(
            torchaudio.transforms.MelSpectrogram(
                sample_rate=sample_rate,
                n_mels=64,
                n_fft=512,
                hop_length=512
            ),
            LogTransform()
        )

    def forward(self, generated_audio, target_audio):
        """
        Compute composite loss between generated and target audio.

        Args:
            generated_audio (torch.Tensor): Generated audio tensor
            target_audio (torch.Tensor): Target audio tensor

        Returns:
            torch.Tensor: Composite loss value
        """
        # Ensure input tensors are properly shaped and on the same device
        generated_audio = generated_audio.squeeze(1)  # Remove channel dimension if needed
        target_audio = target_audio.squeeze(1)

        # Ensure tensors are on the same device and have float dtype
        generated_audio = generated_audio.to(torch.float32)
        target_audio = target_audio.to(torch.float32)

        # 1. Mean Squared Error Loss (Time Domain)
        mse_loss = F.mse_loss(generated_audio, target_audio)

        # 2. Mel Spectrogram Spectral Loss
        # Compute log mel spectrograms
        generated_spec = self.mel_spectrogram(generated_audio)
        target_spec = self.mel_spectrogram(target_audio)
        spectral_loss = F.mse_loss(generated_spec, target_spec)

        # 3. Temporal Coherence Loss
        # Penalize large differences between consecutive samples
        temporal_diff = generated_audio[:, 1:] - generated_audio[:, :-1]
        temporal_loss = torch.mean(torch.abs(temporal_diff))

        # 4. Combine losses with learnable weights
        total_loss = (
            mse_loss +
            spectral_loss +
            temporal_loss
        )

        return total_loss


def save_multimodal_model(model, save_dir="outputs", filename='multimodal_earcon'):
    """
    Save the MultimodalEarconGenerator model and MultimodalEarconLoss parameters.

    Args:
        model (MultimodalEarconGenerator): The model to save
        save_dir (str): Directory to save the model and loss parameters
        prefix (str, optional): Prefix for saved file names. Defaults to 'multimodal_earcon'.
    """
    # Create save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)

    # Save model state dict
    model_path = os.path.join(save_dir, f'{filename}.pt')
    torch.save({
        'model_state_dict': model.state_dict(),
        'image_projection_state_dict': model.image_projection.state_dict(),
        'roundness_projection_state_dict': model.roundness_projection.state_dict(),
        'modality_fusion_state_dict': model.modality_fusion.state_dict()
    }, model_path)

    print(f"Model saved to {model_path}")


def load_multimodal_model(
    model,
    model_path,
    strict=True,
    device='cuda' if torch.cuda.is_available() else 'cpu'
):
    """
    Load the MultimodalEarconGenerator model and MultimodalEarconLoss parameters.

    Args:
        model (MultimodalEarconGenerator): The model to load parameters into
        model_path (str): Path to the saved model state dict
        strict (bool, optional): Whether to strictly enforce that the keys 
                                 in the state dict match the model's keys. 
                                 Defaults to True.

    Returns:
        tuple: (updated_model, updated_loss_fn)
    """
    # Load model state dict
    model_checkpoint = torch.load(model_path, map_location=model.device)

    # Load main model parameters
    model.load_state_dict(model_checkpoint.get('model_state_dict', {}), strict=False)

    # Load individual component state dicts if available
    if 'image_projection_state_dict' in model_checkpoint:
        model.image_projection.load_state_dict(
            model_checkpoint['image_projection_state_dict'],
            strict=strict
        )

    if 'roundness_projection_state_dict' in model_checkpoint:
        model.roundness_projection.load_state_dict(
            model_checkpoint['roundness_projection_state_dict'],
            strict=strict
        )

    if 'modality_fusion_state_dict' in model_checkpoint:
        model.modality_fusion.load_state_dict(
            model_checkpoint['modality_fusion_state_dict'],
            strict=strict
        )

    print(f"Model loaded from {model_path}")

    return model.to(device)
