import logging
from pathlib import Path
from typing import Union

import torch
from transformers import GPT2LMHeadModel, GPT2PreTrainedModel, top_k_top_p_filtering

from generation.knn_transformers.knnlm import KEY_TYPE, KNNWrapper

logger = logging.getLogger(__name__)
logger.setLevel(20)


class DExpertsWrapper(KNNWrapper):
    STOP_TOKEN = "<|endoftext|>"

    def __init__(
        self,
        antiexpert_model: Union[str, Path, GPT2PreTrainedModel] = None,
        expert_model: Union[str, Path, GPT2PreTrainedModel] = None,
        alpha: int = 2.0,
        filter_p: float = 0.9,
    ):
        self.alpha = alpha
        self.filter_p = filter_p
        self.knn_keytype = KEY_TYPE.last_ffn_input
        self.hook_handles = []

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.antiexpert = None
        if antiexpert_model:
            self.antiexpert = GPT2LMHeadModel.from_pretrained(antiexpert_model).to(
                self.device
            )
            self.antiexpert.eval()

        self.expert = None
        if expert_model:
            self.expert = GPT2LMHeadModel.from_pretrained(expert_model).to(self.device)
            self.expert.eval()

    def break_into(self, model):
        """Break into model to enable ensemble of experts."""
        self.model = model
        model.broken_into = True

        self.is_encoder_decoder = model.config.is_encoder_decoder

        # Inject our pre_forward_hook to capture the labels at every forward pass
        self.original_forward_func = model.forward
        model.forward = self.pre_forward_hook

        # Inject our main function after the model's final layer
        # Main function = post_forward_hook
        # It's where the ensemble of LM, expert and anti-expert logits happens.
        final_layer = KNNWrapper.get_model_last_layer(model.config.model_type)(model)
        self.register_hook(final_layer, self.post_forward_hook)
        self.vocab_size = final_layer.out_features

    def pre_forward_hook(
        self, input_ids=None, attention_mask=None, labels=None, **kwargs
    ):
        """Access pre-forward pass to generate base LM, expert and anti-expert logits."""
        self.labels = labels

        self.expert_logits = None
        if self.expert is not None:
            self.expert_logits = self.expert(
                input_ids, labels=labels, attention_mask=attention_mask, **kwargs
            ).logits

        self.antiexpert_logits = None
        if self.antiexpert is not None:
            self.antiexpert_logits = self.antiexpert(
                input_ids, labels=labels, attention_mask=attention_mask, **kwargs
            ).logits

        return self.original_forward_func(
            input_ids=input_ids, labels=labels, attention_mask=attention_mask, **kwargs
        )

    def post_forward_hook(self, module, input, output):
        """Ensemble base LM, expert and anti-expert logits after forward pass."""
        batch, time_dim, vocab_size = output.shape
        shift = 0 if self.is_encoder_decoder else 1
        lm_logits = output
        expert_logits, antiexpert_logits = self.expert_logits, self.antiexpert_logits

        lm_logits = torch.nn.functional.log_softmax(
            lm_logits, dim=-1
        )  # (batch, time, vocab)
        if self.filter_p:
            for i, logits in enumerate(lm_logits):
                lm_logits[i] = top_k_top_p_filtering(logits, top_p=self.filter_p)

        if self.labels is None:
            nonpad_mask = torch.cat(
                [
                    torch.zeros([batch, time_dim - 1], dtype=torch.bool),
                    torch.ones([batch, 1], dtype=torch.bool),
                ],
                axis=-1,
            ).to(self.device)
        else:
            nonpad_mask = torch.cat(
                [
                    self.labels[:, shift:] != -100,
                    torch.zeros([self.labels.shape[0], shift], dtype=torch.bool).to(
                        self.device
                    ),
                ],
                axis=-1,
            )

        lm_logits = lm_logits[nonpad_mask]
        if expert_logits is not None:
            expert_logits = expert_logits[nonpad_mask]
        if antiexpert_logits is not None:
            antiexpert_logits = antiexpert_logits[nonpad_mask]

        new_scores = self.ensemble(
            lm_logits,
            *(
                antiexpert_logits if antiexpert_logits is not None else lm_logits,
                expert_logits if expert_logits is not None else lm_logits,
            ),
            lmbda=self.alpha,
            patch=False,
        )
        output[nonpad_mask] = new_scores

        return output
