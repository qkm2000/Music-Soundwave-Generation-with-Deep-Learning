import torch
import torch.nn as nn
from transformers import AutoProcessor, MusicgenModel, MusicgenDecoderConfig
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class CustomMusicGenModel(nn.Module):
    def __init__(
        self,
        base_musicgen_model_name='facebook/musicgen-small',
        freeze_musicgen=True,
        embedding_dim=512,  # Increased embedding dimension
        num_attention_heads=8,  # Multi-head attention parameters
        num_layers=2  # Number of attention and fusion layers
    ):
        """
        Enhanced MusicGen model with multi-head attention and increased complexity

        Args:
            base_musicgen_model_name (str): Base MusicGen model to use
            freeze_musicgen (bool): Whether to freeze base model parameters
            embedding_dim (int): Dimension for feature embeddings
            num_attention_heads (int): Number of attention heads
            num_layers (int): Number of layers in fusion networks
        """
        super().__init__()

        # Load base MusicGen model
        config = MusicgenDecoderConfig(
            num_codebooks=1
        )
        self.base_model = MusicgenModel(config)
        self.processor = AutoProcessor.from_pretrained(
            base_musicgen_model_name
        )
        # Freeze base model parameters
        if freeze_musicgen:
            for param in self.base_model.parameters():
                param.requires_grad = False

        # Enhanced image feature processing with multiple layers
        self.image_fusion_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(512 if i == 0 else embedding_dim, embedding_dim),
                nn.LayerNorm(embedding_dim),
                nn.ReLU(),
                nn.Dropout(0.3)
            ) for i in range(num_layers)
        ])

        # Enhanced roundness feature processing
        self.roundness_fusion_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(1 if i == 0 else 64, 64),
                nn.LayerNorm(64),
                nn.ReLU(),
                nn.Dropout(0.2)
            ) for i in range(num_layers)
        ])

        # Multi-head attention layer for processing fused conditioning
        self.multi_head_attention = nn.MultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=num_attention_heads,
            dropout=0.3,
            batch_first=True
        )

        # Additional projection layer after multi-head attention
        self.attention_projection = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

        # Final fusion layer
        self.final_fusion_layer = nn.Sequential(
            nn.Linear(embedding_dim + 64, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

    def forward(
        self,
        image_features,  # CLIP extracted features
        roundness_values,
    ):
        """
        Enhanced forward pass with multi-head attention

        Args:
            image_features (torch.Tensor): Image features from CLIP
            roundness_values (torch.Tensor): Roundness values for images

        Returns:
            MusicGen model outputs
        """
        # Process image features through multiple layers
        processed_image_features = image_features
        for layer in self.image_fusion_layers:
            processed_image_features = layer(processed_image_features)

        # Process roundness values through multiple layers
        processed_roundness = roundness_values.unsqueeze(1).float()
        for layer in self.roundness_fusion_layers:
            processed_roundness = layer(processed_roundness)

        # Combine image features and roundness
        combined_features = torch.cat([
            processed_image_features,
            processed_roundness
        ], dim=-1)

        # Fuse features
        fused_conditioning = self.final_fusion_layer(combined_features)

        # Add multi-head attention processing
        # Create a sequence-like input for attention
        attention_input = fused_conditioning.unsqueeze(1)  # Add sequence dimension

        # Apply multi-head attention
        attn_output, _ = self.multi_head_attention(
            attention_input,
            attention_input,
            attention_input
        )

        # Project attention output
        fused_conditioning = self.attention_projection(attn_output.squeeze(1))

        # Generate MusicGen inputs
        fused_conditioning = self.generate_musicgen_inputs(fused_conditioning)

        # Pass to base MusicGen model
        outputs = self.base_model(
            input_ids=fused_conditioning['input_ids'],
            attention_mask=fused_conditioning['attention_mask'],
        )

        return outputs

    def generate_musicgen_inputs(self, vector):
        """
        Convert conditioning vector to MusicGen input format

        Args:
            vector (torch.Tensor): Conditioning vector

        Returns:
            dict: Inputs for MusicGen model
        """
        # Approximate conversion to match MusicGen input requirements
        # This is a placeholder and might need adjustment based on specific MusicGen input format
        input_ids = vector.to(vector.device).long()
        attention_mask = torch.ones(1, 512, dtype=torch.long, device=vector.device)
        result = {
            'input_ids': input_ids,
            'attention_mask': attention_mask
        }
        return result

    def generate(
        self,
        image_features,
        roundness_values,
        num_return_sequences=1
    ):
        """
        Generate audio using custom inputs with enhanced processing

        Args:
            image_features (torch.Tensor): Image features
            roundness_values (torch.Tensor): Roundness values

        Returns:
            Generated audio sequences
        """
        # Process image features through multiple layers
        processed_image_features = image_features
        for layer in self.image_fusion_layers:
            processed_image_features = layer(processed_image_features)

        # Process roundness values through multiple layers
        processed_roundness = roundness_values.unsqueeze(1).float()
        for layer in self.roundness_fusion_layers:
            processed_roundness = layer(processed_roundness)

        # Combine image features and roundness
        combined_features = torch.cat([
            processed_image_features,
            processed_roundness
        ], dim=-1)

        # Fuse features
        fused_conditioning = self.final_fusion_layer(combined_features)

        # Apply multi-head attention
        attention_input = fused_conditioning.unsqueeze(1)
        attn_output, _ = self.multi_head_attention(
            attention_input,
            attention_input,
            attention_input
        )

        # Project attention output
        fused_conditioning = self.attention_projection(attn_output.squeeze(1))

        # Generate audio
        generated_sequences = self.base_model.generate(
            input_ids=fused_conditioning.unsqueeze(0),
            num_return_sequences=num_return_sequences
        )

        return generated_sequences


# Custom Loss Function (remains the same as in the original code)
def earcon_generation_loss(
    generated_audio,
    target_audio,
    image_features,
    roundness_values
):
    """
    Custom loss function for earcon generation

    Args:
        generated_audio (torch.Tensor): Generated audio sequences
        target_audio (torch.Tensor): Target audio sequences
        image_features (torch.Tensor): Original image features
        roundness_values (torch.Tensor): Roundness values

    Returns:
        torch.Tensor: Computed loss
    """
    # 1. Reconstruction Loss
    reconstruction_loss = F.mse_loss(generated_audio, target_audio)

    # 2. Feature Consistency Loss
    # Ensure generated audio maintains some relationship with input features
    image_consistency_loss = F.cosine_similarity(
        image_features,
        generated_audio.mean(dim=[1, 2])  # Aggregate audio features
    ).mean()

    # 3. Roundness Correlation Loss
    # Encourage some relationship between roundness and audio characteristics
    roundness_correlation_loss = F.mse_loss(
        roundness_values,
        generated_audio.std(dim=[1, 2])
    )

    # Combine losses with weights
    total_loss = (
        reconstruction_loss +
        0.1 * image_consistency_loss +
        0.05 * roundness_correlation_loss
    )

    return total_loss


def save_model(model, path):
    """
    Save model weights

    Args:
        model (CustomMusicGenModel): Model to save
        path (str): Save path
    """
    torch.save(model.state_dict(), path)


def load_model(path, base_model='facebook/musicgen-small'):
    """
    Load model weights

    Args:
        path (str): Path to saved model weights
        base_model (str): Base MusicGen model to use

    Returns:
        CustomMusicGenModel: Loaded model
    """
    model = CustomMusicGenModel(base_model)
    model.load_state_dict(torch.load(path))
    return model
