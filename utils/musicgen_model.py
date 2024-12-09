import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from transformers import MusicgenForConditionalGeneration, AutoProcessor
from encodec import EncodecModel


class MultimodalEarconGenerator(nn.Module):
    def __init__(
        self,
        image_features_dim=512,
        earcon_features_dim=512,
        pool_type='max',
        freeze_musicgen=True,
    ):
        super(MultimodalEarconGenerator, self).__init__()

        self.pool_type = pool_type
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.sample_rate = 24000
        self.target_length = 512
        self.encodec = EncodecModel.encodec_model_24khz().to(self.device)
        self.loss_fn = MultimodalEarconLoss(sample_rate=self.sample_rate)

        # Use MusicgenForConditionalGeneration
        self.musicgen = MusicgenForConditionalGeneration.from_pretrained(
            "facebook/musicgen-small")
        self.processor = AutoProcessor.from_pretrained(
            "facebook/musicgen-small")

        # Freeze MusicGen parameters if specified
        if freeze_musicgen:
            for param in self.musicgen.parameters():
                param.requires_grad = False

        # Image vector projection
        self.image_projection = nn.Sequential(
            nn.Linear(image_features_dim, 512),
            nn.ReLU(),
            nn.LayerNorm(512)
        )

        # Roundness input processing
        self.roundness_projection = nn.Sequential(
            nn.Linear(1, 128),
            nn.ReLU(),
            nn.LayerNorm(128)
        )

        # Fusion layer
        self.modality_fusion = nn.Sequential(
            nn.Linear(512 + 128, 512),
            nn.ReLU(),
            nn.LayerNorm(512)
        )

    def forward(
        self,
        image_features,
        image_tag,
        roundness_value,
        target_earcon_features=None
    ):
        # Ensure inputs are on the correct device and require gradients
        image_features = image_features.to(self.device)
        roundness_value = roundness_value.to(self.device)

        # Project image vector
        projected_image = self.image_projection(image_features)

        # Project roundness value
        projected_roundness = self.roundness_projection(roundness_value)

        # Fuse modalities
        fused_input = self.modality_fusion(
            torch.cat([projected_image, projected_roundness], dim=-1)
        )

        # Prepare text input for MusicGen
        text_descriptions = self.convert_fused_input_to_description(fused_input, image_tag, roundness_value)

        # Prepare inputs for MusicGen with explicit padding and truncation
        inputs = self.processor(
            text=text_descriptions,
            padding=True,
            truncation=True,
            return_tensors="pt"
        ).to(self.device)

        audio_values = self.musicgen.generate(
            **inputs,
            max_new_tokens=128,
            do_sample=True,
        )

        # If training, compute loss
        if target_earcon_features is not None:
            # Process the audio tensor directly without saving/loading
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

    def convert_fused_input_to_description(self, fused_input, image_tag, roundness_value):
        """
        Convert the fused input tensor to a text description
        """
        descriptions = []
        for i in range(fused_input.shape[0]):
            # Create a more descriptive text based on the fused input
            mean_val = fused_input[i].mean().item()
            std_val = fused_input[i].std().item()

            # Generate a descriptive text based on the input characteristics
            description = (
                f"A short earcon that suits the following descriptions: "
                f"an intensity level of {mean_val:.2f} "
                f"and variation of {std_val:.2f}, "
                f"with a roundness value of {roundness_value}, "
                f"to match an image description of '{image_tag}'."
            )
            descriptions.append(description)

        return descriptions


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
        mse_weight=1.0,
        spectral_weight=0.5,
        temporal_weight=0.2,
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
                n_mels=128,
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
        # mse_loss = F.mse_loss(generated_audio, target_audio)

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
            # mse_loss +
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
    model_path = os.path.join(save_dir, f'{prefix}.pt')
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
