import logging
from enum import Enum, auto

import numpy as np
import torch
from torch import nn
from transformers import AutoTokenizer, top_k_top_p_filtering

from generation.knn_transformers.datastore import DIST, Datastore

logger = logging.getLogger(__name__)
logger.setLevel(20)


class KEY_TYPE(Enum):
    last_ffn_input = auto()
    last_ffn_output = auto()

    @staticmethod
    def from_string(s):
        try:
            return KEY_TYPE[s.lower()]
        except KeyError:
            raise ValueError()


class METHODS(Enum):
    interpolate = auto()
    interpolate_discourage = auto()
    ensemble = auto()

    @staticmethod
    def from_string(s):
        try:
            return METHODS[s.lower()]
        except KeyError:
            raise ValueError()


class KNNWrapper(object):
    def __init__(
        self,
        dstore_dir,
        dimension,
        flat_index=False,
        knn_sim_func=None,
        knn_keytype=None,
        no_load_keys=False,
        move_dstore_to_mem=False,
        knn_gpu=True,
        recompute_dists=False,
        k=1024,
        other_k=None,
        lmbda=0.25,
        knn_temp=1.0,
        probe=32,
        filter_p=0,
        method="interpolate",
        other_dstore_dir=None,
        ensemble_order=("subtract", "add"),
        debug=False,
    ):
        self.dstore_dir = dstore_dir
        self.other_dstore_dir = other_dstore_dir
        self.dimension = dimension
        self.flat_index = flat_index
        self.lmbda = lmbda
        self.k = k
        self.other_k = other_k or k
        self.knn_temperature = knn_temp
        self.probe = probe
        self.filter_p = filter_p
        self.debug = debug
        self._tokenizer = None  # Used only for debugging purposes

        self.knn_sim_func = DIST.l2 if knn_sim_func is None else knn_sim_func
        self.knn_keytype = (
            KEY_TYPE.last_ffn_input if knn_keytype is None else knn_keytype
        )
        self.no_load_keys = no_load_keys
        self.recompute_dists = recompute_dists
        self.move_dstore_to_mem = move_dstore_to_mem
        self.knn_gpu = (
            knn_gpu and torch.cuda.is_available() and torch.cuda.device_count() > 0
        )

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.prompt_input_ids = None
        self.prompt_attention_mask = None
        self.model = None
        self.vocab_size = None
        self.activation_capturer = None
        self.is_encoder_decoder = None
        self.hook_handles = []

        method_to_method_func = {
            METHODS.interpolate: KNNWrapper.interpolate,
            METHODS.interpolate_discourage: KNNWrapper.interpolate_discourage,
            METHODS.ensemble: KNNWrapper.ensemble,
        }
        self.method = METHODS.from_string(method)
        self.method_func = method_to_method_func[self.method]
        self.ds_ensemble_order = ensemble_order

        if self.method == METHODS.ensemble:
            logger.info(
                f"`dstore_dir` should point to datastore that will be {self.ds_ensemble_order[0]}ed. "
                f"`other_dstore_dir` should point to datastore that will be {self.ds_ensemble_order[1]}ed. "
                "If `other_dstore_dir` is left empty, base language model logits will be used."
            )

    def setup_datastore(self):
        self.datastore = Datastore(
            dstore_dir=self.dstore_dir,
            dimension=self.dimension,
            model_type=self.model.config.model_type,
            device=self.device,
            knn_gpu=self.knn_gpu,
            knn_sim_func=self.knn_sim_func,
            move_dstore_to_mem=self.move_dstore_to_mem,
            probe=self.probe,
            no_load_keys=self.no_load_keys,
            flat_index=self.flat_index,
        ).setup_faiss()
        logger.info(f"dstore loaded. `dstore_size`: {self.datastore.dstore_size}")

        self.other_datastore = None
        if self.other_dstore_dir is not None:
            self.other_datastore = Datastore(
                dstore_dir=self.other_dstore_dir,
                dimension=self.dimension,
                model_type=self.model.config.model_type,
                device=self.device,
                knn_gpu=self.knn_gpu,
                knn_sim_func=self.knn_sim_func,
                move_dstore_to_mem=self.move_dstore_to_mem,
                probe=self.probe,
                no_load_keys=self.no_load_keys,
                flat_index=self.flat_index,
            ).setup_faiss()
            logger.info(
                f"Using `other_dstore_dir`. Size: {self.other_datastore.dstore_size}"
            )
        else:
            logger.info(f"Not using `other_dstore_dir`.")

    def break_into(self, model):
        self.model = model
        model.broken_into = True
        self.setup_datastore()

        self.is_encoder_decoder = model.config.is_encoder_decoder

        # Inject our pre_forward_hook to capture the labels at every forward pass
        self.original_forward_func = model.forward
        model.forward = self.pre_forward_hook

        # Inject our activation_capturer to capture the activations at every forward pass
        layer_to_capture_fn, capture_input = KNNWrapper.model_layer_to_capture[
            model.config.model_type
        ][self.knn_keytype]
        layer_to_capture = layer_to_capture_fn(model)
        self.activation_capturer = ActivationCapturer(
            layer_to_capture, capture_input=capture_input
        )
        self.register_hook(layer_to_capture, self.activation_capturer)

        # Inject our main function after the model's final layer
        final_layer = KNNWrapper.get_model_last_layer(model.config.model_type)(model)
        self.register_hook(final_layer, self.post_forward_hook)
        self.vocab_size = final_layer.out_features

        if self.debug:
            self._tokenizer = AutoTokenizer.from_pretrained(model.config._name_or_path)

    def pre_forward_hook(
        self, input_ids=None, attention_mask=None, labels=None, **kwargs
    ):
        self.labels = labels
        self.input_ids = input_ids
        return self.original_forward_func(
            input_ids=input_ids, labels=labels, attention_mask=attention_mask, **kwargs
        )

    def post_forward_hook(self, module, input, output):
        batch, time_dim, vocab_size = output.shape
        shift = 0 if self.is_encoder_decoder else 1
        lm_logits = output
        lm_logits = torch.nn.functional.log_softmax(
            lm_logits, dim=-1
        )  # (batch, time, vocab)

        # From DExperts - adding this reduced perplexity a bit.
        if self.filter_p:
            for i, logits in enumerate(lm_logits):
                lm_logits[i] = top_k_top_p_filtering(logits, top_p=self.filter_p)

        queries = self.activation_capturer.captured  # (batch, time, dim)

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
        queries = queries[nonpad_mask]  # (nonpad, dim)

        self._first_gen = True if nonpad_mask.shape[1] != 1 else False

        new_scores = self.modify_probabilities(lm_logits, queries)
        output[nonpad_mask] = new_scores

        return output

    def modify_probabilities(self, lm_logits, queries):
        dists, knns = self.datastore.get_knns(
            queries, k=self.k, recompute_dists=self.recompute_dists
        )  # (nonpad batch * time, k)
        neg_dists = -dists
        knn_log_probs, _ = self.datastore.knns_to_log_prob(
            knns,
            neg_dists,
            self.vocab_size,
            knn_temperature=self.knn_temperature,
        )

        if self.debug and self._first_gen:
            self.show_retrieved_context(
                description="dstore",
                knns=knns,
                vals=self.datastore.vals,
                show_test_context=True,
            )

        if self.other_datastore is not None:
            if self.method == METHODS.ensemble:
                dists, knns = self.other_datastore.get_knns(
                    queries, k=self.other_k, recompute_dists=self.recompute_dists
                )  # (nonpad batch * time, k)
                neg_dists = -dists
                other_knn_log_probs, _ = self.other_datastore.knns_to_log_prob(
                    knns,
                    neg_dists,
                    self.vocab_size,
                    knn_temperature=self.knn_temperature,
                )

                if self.debug and self._first_gen:
                    self.show_retrieved_context(
                        description="other_dstore",
                        knns=knns,
                        vals=self.other_datastore.vals,
                        show_test_context=False,
                    )

                knn_log_probs = (knn_log_probs, other_knn_log_probs)
            else:
                raise NotImplementedError(
                    f"Other datastore is not supported for method {self.method}."
                )
        else:
            knn_log_probs = (knn_log_probs,)

        interpolated_scores = self.method_func(
            lm_logits,
            *knn_log_probs,
            lmbda=self.lmbda,
            ensemble_order=self.ds_ensemble_order,
        )  # (nonpad, vocab)

        return interpolated_scores

    def show_retrieved_context(
        self,
        description,
        knns,
        vals,
        show_context_tokens=30,
        num_neighbors=3,
        batch_idx=0,
        show_inference_context=True,
    ):
        """Show sample of retrieved contexts from datastore on each forward pass.

        When ensembling next-token probabilities, we'll retrieve the N closest neighbors
        from each datastore. In this function you can print `num_neighbors` contexts
        up to `show_context_tokens` length.

        Args:
            description (_type_): Description to identify from which datastore
                the context came from.
            knns (_type_): Retrieved neighbors indexes.
            vals (_type_): Retrieved next-tokens.
            show_context_tokens (int, optional): Number of tokens to show for
                each retrieved context. Defaults to 30.
            num_neighbors (int, optional): Number of nearest neighbors to show.
                Defaults to 3.
            batch_idx (int, optional): Which sample from the batch to show
                neighbors for. Defaults to 0.
            show_inference_context (bool, optional): To show the current
                inference context or not. Defaults to True.
        """
        neighbors_to_investigate = knns[batch_idx, :num_neighbors]
        context_tokens = torch.stack(
            [
                vals[idx - show_context_tokens : idx]
                if idx >= show_context_tokens
                else None
                for idx in neighbors_to_investigate
            ]
        ).squeeze(-1)

        context = self._tokenizer.batch_decode(context_tokens)
        vals = self._tokenizer.batch_decode(vals[knns][batch_idx, :num_neighbors])

        if show_inference_context:
            inference_context = self._tokenizer.decode(self.input_ids[batch_idx])
            print(f"==== Test Context: {inference_context}")

        print(f"---- {description} - Contexts and Next Tokens: ")
        for i, (c, v) in enumerate(zip(context, vals)):
            print(f"{i+1}) {c} // {v}")

    def register_hook(self, layer, func, pre=False):
        handle = (
            layer.register_forward_pre_hook(func)
            if pre
            else layer.register_forward_hook(func)
        )
        self.hook_handles.append(handle)

    def break_out(self):
        for h in self.hook_handles:
            h.remove()
        if self.model is not None and self.model.broken_into is not None:
            self.model.forward = self.original_forward_func
            self.model.broken_into = None

    def get_metrics(self):
        return {}

    @staticmethod
    def interpolate(lm_log_probs, knn_log_probs, lmbda, **kwargs):
        return torch.logaddexp(
            lm_log_probs + np.log(1 - lmbda), knn_log_probs + np.log(lmbda)
        )

    @staticmethod
    def interpolate_discourage(lm_log_probs, knn_log_probs, lmbda, **kwargs):
        return torch.log(
            torch.nn.functional.relu(
                torch.exp(np.log(1 + lmbda) + lm_log_probs)
                - torch.exp(np.log(lmbda) + knn_log_probs)
            )
        )

    @staticmethod
    def ensemble(
        lm_log_probs,
        *knn_log_probs,
        lmbda=2.0,
        ensemble_order=("subtract", "add"),
        patch=True,
        **kwargs,
    ):
        def patch_log_probs(log_probs):
            val = log_probs[log_probs > -50].min()
            log_probs[log_probs <= -50] = val
            return log_probs

        assert isinstance(knn_log_probs, tuple)
        assert isinstance(knn_log_probs[0], torch.Tensor)

        # Use the lm log probs as the second set of knn log probs if only one set is provided.
        if len(knn_log_probs) < 2:
            if ensemble_order == ("subtract", "add"):
                knn_log_probs = (knn_log_probs[0], lm_log_probs)
            elif ensemble_order == ("add", "subtract"):
                knn_log_probs = (lm_log_probs, knn_log_probs[0])
            else:
                raise ValueError(f"Invalid ensemble order: {ensemble_order}")

        knn_log_probs_subtract, knn_log_probs_sum = knn_log_probs

        if patch:
            knn_log_probs_subtract = patch_log_probs(knn_log_probs_subtract)
            knn_log_probs_sum = patch_log_probs(knn_log_probs_sum)

        return torch.nn.functional.log_softmax(
            lm_log_probs
            + torch.tensor(lmbda) * (knn_log_probs_sum - knn_log_probs_subtract),
            dim=-1,
        )

    @staticmethod
    def get_model_last_layer(model_type):
        # works for gpt2, marian, t5. If a model does not have an ".lm_head" layer,
        # add an "if model_type is ..." statement here, and return the output embedding layer
        if model_type == "gpt_neox":
            return lambda model: model.embed_out
        return lambda model: model.lm_head

    @staticmethod
    def get_model_embedding_layer(model_type):
        if model_type.startswith("gpt2"):
            return lambda model: model.transformer.wte

    # YOU CAN ADD MORE MODELS HERE!
    # For every model name and key type, returns a lambda that returns the relevant layer in the model,
    # and whether the input of that layer should be captured (True) or the output (False)
    model_layer_to_capture = {
        "bart": {
            KEY_TYPE.last_ffn_input: (
                lambda model: model.base_model.decoder.layers[-1].fc1,
                True,
            ),
            KEY_TYPE.last_ffn_output: (
                lambda model: model.base_model.decoder.layers[-1],
                False,
            ),
        },
        "gpt2": {
            KEY_TYPE.last_ffn_input: (lambda model: model.base_model.h[-1].mlp, True),
            KEY_TYPE.last_ffn_output: (lambda model: model.base_model.h[-1], False),
        },
        "marian": {
            KEY_TYPE.last_ffn_input: (
                lambda model: model.base_model.decoder.layers[-1].fc1,
                True,
            ),
            KEY_TYPE.last_ffn_output: (
                lambda model: model.base_model.decoder.layers[-1],
                False,
            ),
        },
        "t5": {
            KEY_TYPE.last_ffn_input: (
                lambda model: model.base_model.decoder.block[-1]
                .layer[2]
                .DenseReluDense,
                True,
            ),
            KEY_TYPE.last_ffn_output: (
                lambda model: model.base_model.decoder.block[-1].layer[2],
                False,
            ),
        },
        "gpt_neox": {
            KEY_TYPE.last_ffn_input: (
                lambda model: model.gpt_neox.layers[-1].mlp,
                True,
            ),
            KEY_TYPE.last_ffn_output: (lambda model: model.gpt_neox.layers[-1], False),
        },
        "opt": {
            KEY_TYPE.last_ffn_input: (
                lambda model: model.model.decoder.layers[-1],
                True,
            ),
            KEY_TYPE.last_ffn_output: (
                lambda model: model.model.decoder.layers[-1],
                False,
            ),
        },
    }


class KNNSaver(object):
    def __init__(
        self,
        dstore_size,
        dstore_dir,
        dimension,
        knn_keytype=None,
        flat_index=False,
        continue_writing=False,
    ):
        self.dstore_size = dstore_size
        self.dstore_dir = dstore_dir
        self.dimension = dimension
        self.flat_index = flat_index
        self.knn_keytype = (
            KEY_TYPE.last_ffn_input if knn_keytype is None else knn_keytype
        )
        self.continue_writing = continue_writing

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = None
        self.activation_capturer = None
        self.is_encoder_decoder = None
        self.dstore_idx = 0
        self.labels = None
        self.hook_handles = []

        logger.info(f"keytype being saved: {self.knn_keytype}")
        logger.info("Saving fp16")

    def break_into(self, model):
        self.model = model
        model.broken_into = True
        self.is_encoder_decoder = model.config.is_encoder_decoder

        # Inject our activation_capturer to capture the activations at every forward pass
        layer_to_capture_fn, capture_input = KNNWrapper.model_layer_to_capture[
            model.config.model_type
        ][self.knn_keytype]
        layer_to_capture = layer_to_capture_fn(model)
        self.activation_capturer = ActivationCapturer(
            layer_to_capture, capture_input=capture_input
        )
        self.register_hook(layer_to_capture, self.activation_capturer)

        # Inject our pre_forward_hook to capture the labels at every forward pass
        self.original_forward_func = model.forward
        model.forward = self.pre_forward_hook

        # Inject our main function after the model's final layer
        final_layer = KNNWrapper.get_model_last_layer(model.config.model_type)(model)
        self.register_hook(final_layer, self.post_forward_hook)

        self.datastore = Datastore(
            dstore_dir=self.dstore_dir,
            dimension=self.dimension,
            model_type=self.model.config.model_type,
            device=self.device,
            flat_index=self.flat_index,
            dstore_size=self.dstore_size,
            continue_writing=self.continue_writing,
        ).load_keys_and_vals()

        # Update values after datastore loading
        logger.info(
            f"dstore_size initial/actual: {self.dstore_size}/{self.datastore.dstore_size}"
        )
        self.dstore_size = self.datastore.dstore_size
        self.dstore_idx = self.datastore.previous_dstore_size or 0
        logger.info(f"dstore_idx current: {self.dstore_idx}")

    def build_index(self):
        self.datastore.build_index()

    def pre_forward_hook(
        self, input_ids=None, attention_mask=None, labels=None, **kwargs
    ):
        if labels is None:
            raise ValueError(
                "labels must be provided when saving a datastore. Are you using --predict_with_generate by mistake? If so, disable it"
            )
        self.labels = labels
        return self.original_forward_func(
            input_ids=input_ids, labels=labels, attention_mask=attention_mask, **kwargs
        )

    def post_forward_hook(self, module, input, output):
        shift = 0 if self.is_encoder_decoder else 1
        captured_keys = self.activation_capturer.captured
        if shift == 1:
            captured_keys = captured_keys[:, :-shift]
        captured_keys = captured_keys.flatten(0, 1)  # (batch * time, dim)
        captured_values = self.labels[:, shift:].flatten(0, 1)  # (batch * time)

        nonpad_mask = captured_values != -100
        keys = captured_keys[nonpad_mask]
        values = captured_values[nonpad_mask]

        batch_time_size = keys.shape[0]
        # if shape[0] == args.tokens_per_sample:
        if self.dstore_idx + batch_time_size > self.dstore_size:
            batch_time_size = max(self.dstore_size - self.dstore_idx, 0)
            keys = keys[:batch_time_size]
            values = values[:batch_time_size]
        try:
            self.datastore.dstore_keys[
                self.dstore_idx : (batch_time_size + self.dstore_idx)
            ] = (keys.cpu().numpy().astype(np.float16))
            self.datastore.dstore_vals[
                self.dstore_idx : (batch_time_size + self.dstore_idx)
            ] = (values.unsqueeze(-1).cpu().numpy().astype(np.int32))
        except ValueError as ex:
            logger.error(
                f"Error saving datastore with mode {self.datastore.dstore_keys.mode}, did you try to save an already existing datastore?"
            )
            logger.error(
                f"Delete the files {self.datastore.dstore_keys.filename} and {self.datastore.dstore_vals.filename} and try again"
            )
            raise ex

        self.dstore_idx += batch_time_size

        if self.dstore_idx >= self.dstore_size:
            logger.info(f"Datastore is full: {self.dstore_idx}/{self.dstore_size}")

        return output

    def register_hook(self, layer, func, pre=False):
        handle = (
            layer.register_forward_pre_hook(func)
            if pre
            else layer.register_forward_hook(func)
        )
        self.hook_handles.append(handle)

    def break_out(self):
        for h in self.hook_handles:
            h.remove()
        if self.model is not None and self.model.broken_into is not None:
            self.model.forward = self.original_forward_func
            self.model.broken_into = None

    def get_metrics(self):
        return {}


class ActivationCapturer(nn.Module):
    def __init__(self, layer, capture_input=False):
        super().__init__()
        self.layer = layer
        self.capture_input = capture_input

        self.captured = None

    def forward(self, module, input, output):
        if self.capture_input:
            self.captured = input[0].detach()
        else:
            self.captured = output.detach()
