import torch
import torch.nn as nn
from transformers import BertPreTrainedModel, BertModel
from transformers.modeling_outputs import MaskedLMOutput
from transformers.generation.utils import GenerationMixin
from .configuration_diffusion_bert import DiffusionBertConfig

class DiffusionBertForMaskedLM(BertPreTrainedModel):
    config_class = DiffusionBertConfig
    _keys_to_ignore_on_load_missing = [r"position_ids", r"predictions.decoder.bias"]
    _keys_to_ignore_on_load_unexpected = [r"pooler"]

    def __init__(self, config):
        super().__init__(config)

        if config.is_decoder:
            logger.warning(
                "If you want to use `DiffusionBertForMaskedLM` make sure `config.is_decoder=False` for "
                "bi-directional self-attention."
            )

        # Initialize BERT model without pooler
        self.bert = BertModel(config, add_pooling_layer=False)
        
        # Time embedding components - dimensions from checkpoint
        self.time_embed = nn.Sequential(
            nn.Linear(1, 768),  # [768, 1]
            nn.ReLU(),
            nn.Linear(768, 768),  # [768, 768]
            nn.ReLU(),
            nn.Linear(768, 768)  # [768, 768]
        )

        # Denoising network - dimensions from checkpoint
        self.denoise_net = nn.Sequential(
            nn.Linear(768, 1536),  # [1536, 768]
            nn.LayerNorm(1536),
            nn.ReLU(),
            nn.Linear(1536, 1536),  # [1536, 1536]
            nn.LayerNorm(1536),
            nn.ReLU(),
            nn.Linear(1536, 768),  # [768, 1536]
            nn.LayerNorm(768),
            nn.ReLU()
        )

        # Refinement network
        self.refinement = nn.Sequential(
            nn.Linear(768, 768),
            nn.LayerNorm(768),
            nn.ReLU(),
            nn.Linear(768, 768)
        )

        # MLM head
        self.cls = BertOnlyMLMHead(config)

        # Initialize weights and apply final processing
        self.post_init()

    def get_output_embeddings(self):
        return self.cls.predictions.decoder

    def set_output_embeddings(self, new_embeddings):
        self.cls.predictions.decoder = new_embeddings

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

        prediction_scores = self.cls(sequence_output)

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

class BertOnlyMLMHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.predictions = BertLMPredictionHead(config)

    def forward(self, sequence_output):
        prediction_scores = self.predictions(sequence_output)
        return prediction_scores

class BertLMPredictionHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.transform = BertPredictionHeadTransform(config)

        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.bias = nn.Parameter(torch.zeros(config.vocab_size))

        # Need a link between the two variables so that the bias is correctly resized with `resize_token_embeddings`
        self.decoder.bias = self.bias

    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states)
        return hidden_states

class BertPredictionHeadTransform(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        if isinstance(config.hidden_act, str):
            self.transform_act_fn = ACT2FN[config.hidden_act]
        else:
            self.transform_act_fn = config.hidden_act
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states 