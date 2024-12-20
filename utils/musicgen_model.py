import os
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Union
from transformers import (
    MusicgenPreTrainedModel,
    MusicgenDecoderConfig,
)
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions
from transformers.generation.streamers import BaseStreamer
from transformers.generation.configuration_utils import GenerationConfig
from transformers.generation.logits_process import LogitsProcessorList
from transformers.generation.stopping_criteria import StoppingCriteriaList
from transformers.models.musicgen.modeling_musicgen import MusicgenDecoder
from encodec import EncodecModel


class MusicgenImageProcessor(nn.Module):
    def __init__(
        self,
        hidden_size: int = 2048,
        vocab_size: int = 2048,
        max_length: int = 128,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        super().__init__()
        self.device = device
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.max_length = max_length

        # Input projection to handle image vector and roundness
        self.image_projection = nn.Sequential(
            nn.Linear(512 + 1, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )

        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=hidden_size, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=hidden_size, out_channels=hidden_size, kernel_size=3, padding=1),
            nn.ReLU()
        )

        # Attention layers
        # self.attention = nn.Sequential(
        #     nn.MultiheadAttention(embed_dim=hidden_size, num_heads=8),
        #     nn.LayerNorm(hidden_size),
        #     nn.Dropout(0.1),
        # )

        # Token prediction layers
        self.token_predictor = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 2),
            nn.ReLU(),
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, vocab_size)
        )

        # Mask predictor
        self.mask_predictor = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1),
            nn.Sigmoid()
        )

        # Trainable threshold for attention mask
        self.threshold = nn.Parameter(torch.tensor(0.4), requires_grad=True)
        self.threshold.data.clamp_(0, 1)

        # Xavier initialization for all weights
        self.reset_parameters()

    def forward(
        self,
        image_features,
        roundness,
    ):
        # Combine image features and roundness
        combined_input = torch.cat([image_features, roundness], dim=-1).to(self.device)

        # Project and reshape for 1D convolutions
        x = self.image_projection(combined_input)
        x = x.unsqueeze(1)
        x = self.cnn(x)

        # Predict tokens
        logits = self.token_predictor(x)  # [batch_size, seq_len, vocab_size]
        gumbel_softmax_output = torch.nn.functional.gumbel_softmax(
            logits, tau=1.0, hard=True, dim=-1
        )  # Hard=True enables discrete token selection
        input_ids = torch.argmax(gumbel_softmax_output, dim=-1).long()
        # input_ids = logits.argmax(dim=-1)  # [batch_size, seq_len]

        # Predict attention mask
        mask_logits = self.mask_predictor(x).squeeze(-1)  # [batch_size, seq_len]
        sigmoid_mask = torch.sigmoid(mask_logits)
        attention_mask = (sigmoid_mask > self.threshold).float().long()

        # Ensure proper sequence length
        if input_ids.shape[1] < self.max_length:
            pad_length = self.max_length - input_ids.shape[1]
            input_ids = torch.nn.functional.pad(input_ids, (0, pad_length), value=0)
            attention_mask = torch.nn.functional.pad(attention_mask, (0, pad_length), value=0)
        else:
            input_ids = input_ids[:, :self.max_length]
            attention_mask = attention_mask[:, :self.max_length]

        # input_ids = input_ids.requires_grad_(True)
        # attention_mask = attention_mask.requires_grad_(True)

        return {
            'input_ids': input_ids,  # [batch_size, max_length]
            'attention_mask': attention_mask  # [batch_size, max_length]
        }

    def reset_parameters(self):
        """Initialize weights and biases."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)  # Xavier init for Linear layers
                if module.bias is not None:
                    nn.init.zeros_(module.bias)  # Zero initialization for biases
            elif isinstance(module, nn.Conv1d):
                nn.init.kaiming_uniform_(module.weight, nonlinearity='relu')  # He init for Conv layers
                if module.bias is not None:
                    nn.init.zeros_(module.bias)  # Zero initialization for biases
            elif isinstance(module, nn.MultiheadAttention):
                nn.init.xavier_uniform_(module.in_proj_weight)  # Xavier init for attention weights
                if module.bias_k is not None:
                    nn.init.zeros_(module.bias_k)  # Initialize key bias to zero
                if module.bias_v is not None:
                    nn.init.zeros_(module.bias_v)  # Initialize value bias to zero


class MusicgenForImageLM(MusicgenPreTrainedModel):
    def __init__(
        self,
        config: MusicgenDecoderConfig = None,
        freeze_encodec: bool = True,
        freeze_musicgen: bool = True,
    ):
        super().__init__(config)

        if config is None:
            config = MusicgenDecoderConfig(
                num_codebooks=1,
            )

        self.model = MusicgenDecoder(config)
        self.encodec = EncodecModel.encodec_model_24khz()
        self.num_codebooks = config.num_codebooks
        self.generation_config.max_length = 128

        # Freeze the Encodec model
        if freeze_encodec:
            for param in self.encodec.parameters():
                param.requires_grad = False

        # Freeze the Musicgen model
        if freeze_musicgen:
            for param in self.model.parameters():
                param.requires_grad = False

        # Keep the LM heads for multi-codebook generation
        self.lm_heads = nn.ModuleList(
            [nn.Linear(config.hidden_size, config.vocab_size, bias=False)
             for _ in range(config.num_codebooks)]
        )

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithCrossAttentions]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Call the base model
        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            head_mask=head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]

        # Apply LM heads to hidden states
        lm_logits = torch.stack([head(hidden_states) for head in self.lm_heads], dim=1)

        # Reshape logits
        lm_logits = lm_logits.reshape(-1, *lm_logits.shape[2:])
        lm_logits = lm_logits.view(lm_logits.size(0), 1, -1)

        # Calculate loss if labels are provided
        loss = None
        if labels is not None:
            length = 512
            # Extract audio features from generated audio
            # Encode audio using Encodec model
            generated_audio_features = self.encodec.encode(lm_logits)[0][0].float()

            # Pad or truncate the generated_audio_features to length 512
            if generated_audio_features.shape[-1] < length:
                padding = length - generated_audio_features.shape[-1]
                generated_audio_features = F.pad(generated_audio_features, (0, padding))
            else:
                generated_audio_features = generated_audio_features[:, :, :length]

            generated_audio_features = generated_audio_features.requires_grad_(True)

            # Calculate feature similarity loss
            # Ensure all tensors require gradients
            generated_audio_features = generated_audio_features.requires_grad_(True)
            labels = labels.requires_grad_(True)

            # Normalize features
            generated_norm = F.normalize(generated_audio_features, p=2, dim=-1)
            target_norm = F.normalize(labels, p=2, dim=-1)

            # Calculate cosine similarity
            cosine_similarity = torch.sum(generated_norm * target_norm, dim=-1)

            # Convert to loss (minimizing negative cosine similarity)
            feature_loss = 1 - cosine_similarity.mean()

            # Calculate mean squared error (MSE) loss
            mse_loss = F.mse_loss(generated_audio_features, labels)
            loss = mse_loss + feature_loss
            # loss = mse_loss
            # print(loss.requires_grad)

        if not return_dict:
            output = (lm_logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutputWithCrossAttentions(
            loss=loss,
            logits=lm_logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            cross_attentions=outputs.cross_attentions,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        attention_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        head_mask=None,
        cross_attn_head_mask=None,
        past_key_values=None,
        use_cache=True,
        delay_pattern_mask=None,
        **kwargs,
    ):
        # First handle delay pattern masking
        if delay_pattern_mask is None:
            input_ids, delay_pattern_mask = self.build_delay_pattern_mask(
                input_ids,
                pad_token_id=self.generation_config.pad_token_id,
                max_length=self.generation_config.max_length,
            )

        # Apply the delay pattern mask
        input_ids = self.apply_delay_pattern_mask(input_ids, delay_pattern_mask)

        # Handle past key values for efficient generation
        if past_key_values is not None:
            input_ids = input_ids[:, -1:]

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "encoder_hidden_states": encoder_hidden_states,
            "encoder_attention_mask": encoder_attention_mask,
            "head_mask": head_mask,
            "cross_attn_head_mask": cross_attn_head_mask,
            "past_key_values": past_key_values,
            "use_cache": use_cache,
        }

    @torch.no_grad()
    def generate(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        generation_config: Optional[GenerationConfig] = None,
        logits_processor: Optional[LogitsProcessorList] = None,
        stopping_criteria: Optional[StoppingCriteriaList] = None,
        synced_gpus: Optional[bool] = None,
        streamer: Optional["BaseStreamer"] = None,
        **kwargs,
    ):
        """
        Generates audio sequences based on input ids and attention mask.

        Parameters:
            input_ids (torch.LongTensor, optional): Input token ids
            attention_mask (torch.LongTensor, optional): Attention mask
            generation_config (GenerationConfig, optional): Configuration for generation
            logits_processor (LogitsProcessorList, optional): Custom logits processors
            stopping_criteria (StoppingCriteriaList, optional): Custom stopping criteria
            synced_gpus (bool, optional): Whether to synchronize GPUs
            streamer (BaseStreamer, optional): Streamer for generated sequences
            **kwargs: Additional generation parameters

        Returns:
            Generated audio sequence
        """
        # 1. Handle `generation_config` and kwargs that might update it, and validate the resulting objects
        if generation_config is None:
            generation_config = self.generation_config

        generation_config = copy.deepcopy(generation_config)
        model_kwargs = generation_config.update(**kwargs)
        generation_config.validate()
        self._validate_model_kwargs(model_kwargs.copy())

        # 2. Set generation parameters if not already defined
        logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
        stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()

        if input_ids is None:
            # Initialize input_ids with start tokens if not provided
            input_ids = torch.full(
                (1, 8),
                generation_config.bos_token_id,
                dtype=torch.long,
                device=self.device
            )

        batch_size = input_ids.shape[0]

        # Prepare special tokens and model kwargs
        self._prepare_special_tokens(
            generation_config, False, device=input_ids.device)
        model_kwargs = {
            "use_cache": generation_config.use_cache,
            "attention_mask": attention_mask,
        }

        # 3. Prepare inputs for generation
        input_ids, model_input_name, model_kwargs = self._prepare_model_inputs(
            input_ids, generation_config.bos_token_id, model_kwargs
        )

        # Build the delay pattern mask for multi-codebook generation
        input_ids, delay_pattern_mask = self.build_delay_pattern_mask(
            input_ids,
            pad_token_id=generation_config._decoder_start_token_tensor,
            max_length=generation_config.max_length,
        )

        if streamer is not None:
            streamer.put(input_ids.cpu())

        # Stash the delay mask
        model_kwargs["delay_pattern_mask"] = delay_pattern_mask

        # 4. Prepare logits processor
        logits_processor = self._get_logits_processor(
            generation_config=generation_config,
            input_ids_seq_length=input_ids.shape[-1],
            encoder_input_ids=input_ids,
            prefix_allowed_tokens_fn=None,
            logits_processor=logits_processor,
            device=input_ids.device,
        )

        # 5. Prepare stopping criteria
        stopping_criteria = self._get_stopping_criteria(
            generation_config=generation_config, stopping_criteria=stopping_criteria
        )

        # 6. Prepare logits warper for sampling
        prepared_logits_warper = (
            self._get_logits_warper(generation_config, device=input_ids.device)
            if generation_config.do_sample
            else None
        )

        # Expand inputs for multiple return sequences
        input_ids, model_kwargs = self._expand_inputs_for_generation(
            input_ids=input_ids,
            expand_size=generation_config.num_return_sequences,
            **model_kwargs,
        )

        # 7. Run generation (sampling)
        outputs = self._sample(
            input_ids,
            logits_processor=logits_processor,
            logits_warper=prepared_logits_warper,
            stopping_criteria=stopping_criteria,
            generation_config=generation_config,
            synced_gpus=synced_gpus,
            streamer=streamer,
            **model_kwargs,
        )

        # 8. Process outputs
        if generation_config.return_dict_in_generate:
            output_ids = outputs.sequences
        else:
            output_ids = outputs

        # Apply the pattern mask to the final ids
        output_ids = self.apply_delay_pattern_mask(
            output_ids, model_kwargs["delay_pattern_mask"])

        # Revert the pattern delay mask by filtering the pad token id
        output_ids = output_ids[output_ids != generation_config._pad_token_tensor].reshape(
            batch_size, self.num_codebooks, -1
        )

        if generation_config.return_dict_in_generate:
            outputs.sequences = output_ids
            return outputs
        else:
            return output_ids

    def build_delay_pattern_mask(self, input_ids: torch.LongTensor, pad_token_id: int, max_length: int = None):
        """Build a delayed pattern mask to the input_ids."""
        # (bsz * num_codebooks, seq_len) -> (bsz, num_codebooks, seq_len)
        input_ids = input_ids.reshape(-1, self.num_codebooks, input_ids.shape[-1])
        bsz, num_codebooks, seq_len = input_ids.shape

        max_length = max_length if max_length is not None else self.generation_config.max_length
        input_ids_shifted = (
            torch.ones((bsz, num_codebooks, max_length), dtype=torch.long, device=input_ids.device) * -1
        )

        channel_codebooks = num_codebooks // 2 if self.config.audio_channels == 2 else num_codebooks
        # we only apply the mask if we have a large enough seq len - otherwise we return as is
        if max_length < 2 * channel_codebooks - 1:
            return input_ids.reshape(bsz * num_codebooks, -1), input_ids_shifted.reshape(bsz * num_codebooks, -1)

        # fill the shifted ids with the prompt entries, offset by the codebook idx
        for codebook in range(channel_codebooks):
            if self.config.audio_channels == 1:
                # mono channel - loop over the codebooks one-by-one
                input_ids_shifted[:, codebook, codebook: seq_len + codebook] = input_ids[:, codebook]
            else:
                # left/right channels are interleaved in the generated codebooks
                input_ids_shifted[:, 2 * codebook, codebook: seq_len + codebook] = input_ids[:, 2 * codebook]
                input_ids_shifted[:, 2 * codebook + 1, codebook: seq_len + codebook] = input_ids[:, 2 * codebook + 1]

        # construct a pattern mask that indicates the positions of padding tokens for each codebook
        delay_pattern = torch.triu(
            torch.ones((channel_codebooks, max_length), dtype=torch.bool), diagonal=max_length - channel_codebooks + 1
        )
        # then fill the lower triangular part (the BOS padding)
        delay_pattern = delay_pattern + \
            torch.tril(torch.ones((channel_codebooks, max_length), dtype=torch.bool))

        if self.config.audio_channels == 2:
            # for left/right channel we need to duplicate every row of the pattern mask in an interleaved fashion
            delay_pattern = delay_pattern.repeat_interleave(2, dim=0)

        mask = ~delay_pattern.to(input_ids.device)
        if pad_token_id is None:
            pad_token_id = 2048
        input_ids = mask * input_ids_shifted + ~mask * pad_token_id

        # find the first position to start generating - this is the first place we have the -1 token
        # and will always be in the first codebook (since it has no codebook offset)
        first_codebook_ids = input_ids[:, 0, :]
        start_ids = (first_codebook_ids == -1).nonzero()[:, 1]
        if len(start_ids) > 0:
            first_start_id = min(start_ids)
        else:
            # we have no tokens that need to be filled - return entire matrix of input ids
            first_start_id = seq_len

        # (bsz * num_codebooks, seq_len) -> (bsz, num_codebooks, seq_len)
        pattern_mask = input_ids.reshape(bsz * num_codebooks, -1)
        input_ids = input_ids[..., :first_start_id].reshape(
            bsz * num_codebooks, -1)
        return input_ids, pattern_mask

    @staticmethod
    def apply_delay_pattern_mask(input_ids, decoder_pad_token_mask):
        """Apply a delay pattern mask to the decoder input ids, only preserving predictions where
        the mask is set to -1, and otherwise setting to the value detailed in the mask."""
        seq_len = input_ids.shape[-1]
        decoder_pad_token_mask = decoder_pad_token_mask[..., :seq_len]
        print(input_ids.shape)
        print(decoder_pad_token_mask.shape)
        input_ids = torch.where(
            decoder_pad_token_mask == -1, input_ids, decoder_pad_token_mask)
        return input_ids


def save_musicgen_image_model(model, save_path="outputs/", filename='MusicGenModel_0x.pt'):
    """
    Save the MusicgenForImageLM model and its configuration.

    Args:
        model (MusicgenForImageLM): The model to save
        save_path (str): Directory path where the model will be saved
        filename (str, optional): Name of the file to save the model. Defaults to 'musicgen_image_model.pth'

    Returns:
        str: Full path to the saved model file
    """
    if not filename.endswith('.pt'):
        filename += '.pt'

    # Ensure the save directory exists
    os.makedirs(save_path, exist_ok=True)

    # Create the full file path
    full_path = os.path.join(save_path, filename)

    # Prepare a dictionary with model state and configuration
    save_dict = {
        'model_state_dict': model.state_dict(),
        'config': model.config.to_dict(),
    }

    # Save the model
    torch.save(save_dict, full_path)

    print(f"Model saved to {full_path}")
    return full_path


def save_musicgen_image_processor(processor, save_path="outputs/", filename='MusicGenImageProcessor_0x.pt'):
    """
    Save the MusicgenImageProcessor model and its configuration.

    Args:
        processor (MusicgenImageProcessor): The processor to save
        save_path (str): Directory path where the processor will be saved
        filename (str, optional): Name of the file to save the processor. Defaults to 'musicgen_image_processor.pth'

    Returns:
        str: Full path to the saved processor file
    """
    if not filename.endswith('.pt'):
        filename += '.pt'

    # Ensure the save directory exists
    os.makedirs(save_path, exist_ok=True)

    # Create the full file path
    full_path = os.path.join(save_path, filename)

    # Save the processor
    torch.save(processor.state_dict(), full_path)

    print(f"Processor saved to {full_path}")
    return full_path


def load_musicgen_image_model(
    load_path="outputs/",
    filename='MusicGenModel_0x.pt'
):
    """
    Load a previously saved MusicgenForImageLM model.

    Args:
        model_class (type): The model class used to instantiate the model (MusicgenForImageLM)
        load_path (str): Directory path where the model is saved
        filename (str, optional): Name of the file to load the model from. Defaults to 'musicgen_image_model.pth'

    Returns:
        MusicgenForImageLM: Loaded and configured model
    """
    if not filename.endswith('.pt'):
        filename += '.pt'

    # Create the full file path
    full_path = os.path.join(load_path, filename)

    # Check if the file exists
    if not os.path.exists(full_path):
        raise FileNotFoundError(f"Model file not found at {full_path}")

    # Load the saved dictionary
    save_dict = torch.load(full_path, map_location=torch.device('cpu'), weights_only=True)

    # Recreate the model configuration
    model_config = save_dict['config']
    model_config = MusicgenDecoderConfig(**model_config)

    # Instantiate a new model with the saved configuration
    model = MusicgenForImageLM(config=model_config)

    # Load the model's state dictionary
    model.load_state_dict(save_dict['model_state_dict'])

    # Set the model to evaluation mode
    model.eval()

    print(f"Model loaded from {full_path}")
    return model.to("cuda" if torch.cuda.is_available() else "cpu")


def load_musicgen_image_processor(
    load_path="outputs/",
    filename='MusicGenImageProcessor_0x.pt'
):
    """
    Load a previously saved MusicgenImageProcessor model.

    Args:
        load_path (str): Directory path where the processor is saved
        filename (str, optional): Name of the file to load the processor from. Defaults to 'musicgen_image_processor.pth'

    Returns:
        MusicgenImageProcessor: Loaded and configured processor
    """
    if not filename.endswith('.pt'):
        filename += '.pt'

    # Create the full file path
    full_path = os.path.join(load_path, filename)

    # Check if the file exists
    if not os.path.exists(full_path):
        raise FileNotFoundError(f"Processor file not found at {full_path}")

    # Instantiate a new processor
    processor = MusicgenImageProcessor()

    # Load the processor's state dictionary
    processor.load_state_dict(torch.load(full_path, weights_only=True))

    # Set the processor to evaluation mode
    processor.eval()

    print(f"Processor loaded from {full_path}")
    return processor.to("cuda" if torch.cuda.is_available() else "cpu")