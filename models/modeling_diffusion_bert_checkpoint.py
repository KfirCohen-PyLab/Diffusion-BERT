import torch
import torch.nn as nn
from transformers import BertPreTrainedModel, BertModel
from transformers.modeling_outputs import MaskedLMOutput
from transformers.generation.utils import GenerationMixin

class DiffusionBertForMaskedLM(BertPreTrainedModel, GenerationMixin):
    def __init__(self, config):
        super().__init__(config)
        self.bert = BertModel(config)
        
        # Time embedding components - matches checkpoint architecture
        self.time_embed = nn.Sequential(
            nn.Linear(1, config.hidden_size),  # Changed from hidden_size to 1
            nn.ReLU(),
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.ReLU()  # Removed final linear layer
        )

        # Denoising network - matches checkpoint architecture
        self.denoise_net = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size * 2),
            nn.LayerNorm(config.hidden_size * 2),
            nn.ReLU(),
            nn.Linear(config.hidden_size * 2, config.hidden_size),
            nn.LayerNorm(config.hidden_size),
            nn.ReLU(),
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.LayerNorm(config.hidden_size),
            nn.ReLU()
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
            t_emb = timestep.float().view(-1, 1) / sequence_output.shape[1]
            
            # Get time embedding
            time_embed = self.time_embed(t_emb)
            time_embed = time_embed.unsqueeze(1).expand(-1, sequence_output.shape[1], -1)
            
            # Apply denoising
            sequence_output = sequence_output + time_embed
            sequence_output = self.denoise_net(sequence_output)

        # Get prediction scores (using BERT's vocab transform)
        prediction_scores = self.bert.embeddings.word_embeddings(input_ids).transpose(1, 2)
        prediction_scores = torch.matmul(sequence_output, prediction_scores)

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