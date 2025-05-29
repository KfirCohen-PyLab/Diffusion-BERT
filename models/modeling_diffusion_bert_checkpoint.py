import torch
import torch.nn as nn
from transformers import BertPreTrainedModel, BertModel
from transformers.modeling_outputs import MaskedLMOutput
from transformers.generation.utils import GenerationMixin

class DiffusionBertForMaskedLM(BertPreTrainedModel, GenerationMixin):
    def __init__(self, config):
        super().__init__(config)
        self.bert = BertModel(config)
        
        # Time embedding components - matches checkpoint architecture exactly
        self.time_embed = nn.Sequential(
            nn.Linear(1, 1536),  # Changed to match checkpoint
            nn.ReLU(),
            nn.Linear(1536, 1536),
            nn.ReLU(),
            nn.Linear(1536, 768)  # Final projection to model dimension
        )

        # Denoising network - matches checkpoint architecture exactly
        self.denoise_net = nn.Sequential(
            nn.Linear(768, 1536),  # Input projection
            nn.LayerNorm(1536),
            nn.ReLU(),
            nn.Linear(1536, 1536),  # Middle layer
            nn.LayerNorm(1536),
            nn.ReLU(),
            nn.Linear(1536, 768),  # Output projection
            nn.LayerNorm(768),
            nn.ReLU()
        )

        # Refinement network - matches checkpoint
        self.refinement = nn.Sequential(
            nn.Linear(768, 768),
            nn.LayerNorm(768),
            nn.ReLU(),
            nn.Linear(768, 768)
        )

        # Initialize weights
        self.init_weights()
        
        # Register position IDs buffer
        position_ids = torch.arange(config.max_position_embeddings).expand((1, -1))
        self.register_buffer('position_ids', position_ids)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        timestep=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]

        # Apply time embedding if timestep is provided
        if timestep is not None:
            # Convert timestep to embedding
            t_emb = timestep.float().view(-1, 1)
            
            # Get time embedding
            time_embed = self.time_embed(t_emb)
            time_embed = time_embed.unsqueeze(1).expand(-1, sequence_output.shape[1], -1)
            
            # Apply denoising and refinement
            sequence_output = sequence_output + time_embed
            sequence_output = self.denoise_net(sequence_output)
            sequence_output = self.refinement(sequence_output)

        # Get prediction scores (using BERT's vocab transform)
        prediction_scores = torch.matmul(sequence_output, self.bert.embeddings.word_embeddings.weight.transpose(0, 1))

        masked_lm_loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))

        if not return_dict:
            output = (prediction_scores,) + outputs[2:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        return MaskedLMOutput(
            loss=masked_lm_loss,
            logits=prediction_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        ) 