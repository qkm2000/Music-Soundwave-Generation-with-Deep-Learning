import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from typing import Optional, Tuple, Union
from transformers import (
    MusicgenPreTrainedModel,
    MusicgenDecoderConfig,
    MusicgenModel,
)
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions


class MusicgenForImageLM(MusicgenPreTrainedModel):
    def __init__(self, config: MusicgenDecoderConfig):
        super().__init__(config)

        self.model = MusicgenModel(config)

        self.num_codebooks = config.num_codebooks

        # Modify the input projection to handle image vector and roundness
        self.image_projection = nn.Sequential(
            nn.Linear(512 + 1, config.hidden_size),
            nn.ReLU(),
            nn.Linear(config.hidden_size, config.hidden_size)
        )

        # Keep the LM heads for multi-codebook generation
        self.lm_heads = nn.ModuleList(
            [nn.Linear(config.hidden_size, config.vocab_size, bias=False)
             for _ in range(config.num_codebooks)]
        )

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        image_features: torch.FloatTensor,  # Shape: [batch_size, 512]
        roundness: torch.FloatTensor,     # Shape: [batch_size, 1]
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

        # Combine image vector and roundness
        combined_input = torch.cat([image_features, roundness], dim=1)

        # Project the combined input to the model's hidden size
        projected_input = self.image_projection(combined_input)

        # If no input_ids are provided, initialize with start token
        if input_ids is None:
            input_ids = torch.full(
                (projected_input.size(0), 8),
                self.config.bos_token_id,
                dtype=torch.long,
                device=projected_input.device
            )

        # Prepare encoder hidden states from the projected input
        encoder_hidden_states = projected_input.unsqueeze(1)
        encoder_attention_mask = torch.ones_like(input_ids, dtype=torch.long)

        # Call the base model with the image-derived encoder states
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
        lm_logits = torch.stack([head(hidden_states)
                                for head in self.lm_heads], dim=1)

        loss = None
        if labels is not None:
            # Loss computation similar to the original implementation
            logits = lm_logits[:, :, -labels.shape[1]:]

            loss_fct = CrossEntropyLoss()
            loss = torch.zeros([], device=self.device)

            # Mask pad tokens
            labels = labels.masked_fill(labels == self.config.pad_token_id, -100)

            # Per codebook cross-entropy
            for codebook in range(self.config.num_codebooks):
                codebook_logits = logits[:, codebook].contiguous().view(-1, logits.shape[-1])
                codebook_labels = labels[..., codebook].contiguous().view(-1)
                loss += loss_fct(codebook_logits, codebook_labels)

            loss = loss / self.config.num_codebooks

        # Reshape logits
        lm_logits = lm_logits.reshape(-1, *lm_logits.shape[2:])
        lm_logits = lm_logits.view(lm_logits.size(0), 1, -1)

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

    # Most other methods from the original class can remain the same
    # You may want to modify prepare_inputs_for_generation and generate methods
    # to handle the new input format (image vector and roundness)
    def prepare_inputs_for_generation(
        self,
        input_ids,
        image_features=None,
        roundness=None,
        **kwargs
    ):
        # Modify to incorporate image vector and roundness
        base_kwargs = super().prepare_inputs_for_generation(input_ids, **kwargs)

        if image_features is not None:
            base_kwargs['encoder_hidden_states'] = self.image_projection(
                torch.cat([image_features, roundness], dim=1)
            ).unsqueeze(1)
            base_kwargs['encoder_attention_mask'] = torch.ones_like(
                input_ids, dtype=torch.long)

        return base_kwargs

    def generate(
        self,
        image_features: Optional[torch.Tensor] = None,
        roundness: Optional[torch.Tensor] = None,
        **kwargs
    ):
        # Validate input
        if image_features is None or roundness is None:
            raise ValueError(
                "Both image_features and roundness must be provided for generation")

        # Ensure correct shapes
        if image_features.dim() == 1:
            image_features = image_features.unsqueeze(0)
        if roundness.dim() == 0:
            roundness = roundness.unsqueeze(0)

        # Project image vector and roundness
        projected_input = self.image_projection(
            torch.cat([image_features, roundness], dim=1))

        # Initialize input_ids if not provided
        inputs = kwargs.get('inputs')
        if inputs is None:
            inputs = torch.full(
                (image_features.size(0), 1),
                self.generation_config.bos_token_id,
                dtype=torch.long,
                device=image_features.device
            )

        # Call the parent generate method with modified inputs
        kwargs['encoder_hidden_states'] = projected_input.unsqueeze(1)
        kwargs['encoder_attention_mask'] = torch.ones_like(
            inputs, dtype=torch.long)
        kwargs['inputs'] = inputs

        return super().generate(**kwargs)
