import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import MusicgenForConditionalGeneration, AutoProcessor, EncodecFeatureExtractor
import scipy
import librosa
import os


class MultimodalEarconGenerator(nn.Module):
    def __init__(
        self,
        image_vector_dim=512,
        earcon_vector_dim=512,
        pool_type='max',
        freeze_musicgen=True,
    ):
        super(MultimodalEarconGenerator, self).__init__()

        self.pool_type = pool_type
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.sample_rate = 24000

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
            nn.Linear(image_vector_dim, 512),
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
        image_vector,
        roundness_value,
        target_earcon_vector=None
    ):
        # Ensure inputs are on the correct device and require gradients
        image_vector = image_vector.to(self.device)
        roundness_value = roundness_value.to(self.device)

        # Project image vector
        projected_image = self.image_projection(image_vector)

        # Project roundness value
        projected_roundness = self.roundness_projection(roundness_value)

        # Fuse modalities
        fused_input = self.modality_fusion(
            torch.cat([projected_image, projected_roundness], dim=-1)
        )

        # Prepare text input for MusicGen
        text_descriptions = self.convert_fused_input_to_description(fused_input)

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
        if target_earcon_vector is not None:
            # Process the audio tensor directly without saving/loading
            waveform = self.process_audio_tensor(audio_values)

            # Calculate Loss
            loss = F.mse_loss(
                waveform.to(self.device),
                target_earcon_vector.to(self.device)
            )

            return loss, audio_values

        return audio_values

    def process_audio_tensor(self, audio_tensor):
        # Process the audio tensor directly
        encodec = EncodecFeatureExtractor(feature_size=1)

        # Ensure the input is a tensor and on the correct device
        if not isinstance(audio_tensor, torch.Tensor):
            audio_tensor = torch.tensor(audio_tensor)
        audio_tensor = audio_tensor.cpu()
        audio_tensor = audio_tensor.squeeze(0)
        audio_tensor = audio_tensor.squeeze(0)

        # Process with Encodec
        audio_features = encodec(
            audio_tensor,
            sampling_rate=24000,
            return_tensors="pt",
            max_length=131072,
            truncation=True
        )

        # Gradient-preserving processing
        padding_mask = audio_features['padding_mask']
        input_values = audio_features['input_values']

        # Ensure gradient computation
        input_values = input_values.requires_grad_(True)

        expanded_mask = padding_mask.unsqueeze(1)
        masked_input_values = input_values * expanded_mask

        # Pooling with gradient preservation
        batch_size, _, original_size = masked_input_values.shape
        target_size = 512
        chunk_size = original_size // target_size
        if original_size % target_size != 0:
            # Trim tensor if original_size isn't divisible by target_size
            masked_input_values = masked_input_values[0, 0, :(target_size * chunk_size)]

        pooled_tensor = masked_input_values.view(
            batch_size,
            target_size,
            chunk_size
        ).max(dim=2).values

        pooled_tensor = pooled_tensor.squeeze(1)
        pooled_tensor.to(self.device)

        return pooled_tensor

    def convert_fused_input_to_description(self, fused_input):
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
                f"A sound with smooth, flowing characteristics, "
                f"with an intensity level of {mean_val:.2f} "
                f"and variation of {std_val:.2f}"
            )
            descriptions.append(description)

        return descriptions


if __name__ == "__main__":
    # Create dummy inputs
    image_vector = torch.randn(1, 512)  # Batch of 1, 512-dim vectors
    roundness_value = torch.randn(1, 1)  # Batch of 1, single roundness values
    earco_vector = torch.randn(1, 512)  # Batch of 1, 512-dim vectors

    # Instantiate model
    model = MultimodalEarconGenerator()

    # Move model to device
    model = model.to(model.device)

    # Test forward pass
    output = model(image_vector, roundness_value, earco_vector)
    # print("Output shape:", output.shape)
    print(output)
