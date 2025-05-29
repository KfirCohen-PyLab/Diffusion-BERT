import torch
import torch.nn as nn
from transformers import BertPreTrainedModel, BertModel
from transformers.modeling_outputs import MaskedLMOutput
from transformers.generation.utils import GenerationMixin

class DiffusionBertForMaskedLM(BertPreTrainedModel, GenerationMixin):
    def __init__(self, config):
        super().__init__(config)
        self.bert = BertModel(config)
        
        # Prediction head
        self.cls = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.GELU(),
            nn.LayerNorm(config.hidden_size)
        )
        
        # Decoder (separate from cls to match checkpoint architecture)
        self.predictions = nn.ModuleDict({
            "transform": nn.ModuleDict({
                "dense": nn.Linear(config.hidden_size, config.hidden_size),
                "LayerNorm": nn.LayerNorm(config.hidden_size)
            }),
            "decoder": nn.Linear(config.hidden_size, config.vocab_size, bias=True)
        })
        self.predictions.bias = nn.Parameter(torch.zeros(config.vocab_size))

        # Time embedding components
        self.time_embed = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size * 2),
            nn.ReLU(),
            nn.Linear(config.hidden_size * 2, config.hidden_size),
            nn.ReLU(),
            nn.Linear(config.hidden_size, config.hidden_size)
        )

        # Refinement network
        self.refinement = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.LayerNorm(config.hidden_size),
            nn.ReLU(),
            nn.Linear(config.hidden_size, config.hidden_size)
        )

        # Denoising network
        self.denoise_net = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size * 2),
            nn.LayerNorm(config.hidden_size * 2),
            nn.ReLU(),
            nn.Linear(config.hidden_size * 2, config.hidden_size * 2),
            nn.LayerNorm(config.hidden_size * 2),
            nn.ReLU(),
            nn.Linear(config.hidden_size * 2, config.hidden_size),
            nn.LayerNorm(config.hidden_size),
            nn.ReLU()
        )

        # Initialize weights
        self.init_weights()
        
        # Register position IDs buffer
        position_ids = torch.arange(config.max_position_embeddings).expand((1, -1))
        self.register_buffer('position_ids', position_ids)

    def get_output_embeddings(self):
        return self.predictions.decoder

    def set_output_embeddings(self, new_embeddings):
        self.predictions.decoder = new_embeddings

    def prepare_inputs_for_generation(self, input_ids, attention_mask=None, **kwargs):
        input_shape = input_ids.shape
        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=input_ids.device)
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            **kwargs
        }

    def _reorder_cache(self, past, beam_idx):
        reordered_past = ()
        for layer_past in past:
            reordered_past += (tuple(past_state.index_select(0, beam_idx) for past_state in layer_past),)
        return reordered_past

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

        # Get BERT outputs
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )

        # Get sequence output
        sequence_output = outputs[0]

        # Apply time embedding if timestep is provided
        if timestep is not None:
            time_emb = self.time_embed(timestep.unsqueeze(-1).float())
            sequence_output = sequence_output + time_emb.unsqueeze(1)

        # Apply denoising network
        hidden_states = self.denoise_net(sequence_output)
        
        # Apply refinement
        hidden_states = self.refinement(hidden_states)
        
        # Apply prediction head
        hidden_states = self.cls(hidden_states)
        
        # Apply final transformation
        hidden_states = self.predictions.transform.dense(hidden_states)
        hidden_states = torch.nn.functional.gelu(hidden_states)
        hidden_states = self.predictions.transform.LayerNorm(hidden_states)
        
        # Get prediction scores
        prediction_scores = self.predictions.decoder(hidden_states)
        prediction_scores = prediction_scores + self.predictions.bias

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