# Copyright 2025 NVIDIA CORPORATION & AFFILIATES
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0
# Modified from Dream repos: https://github.com/HKUNLP/Dream

import warnings
import copy
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.distributions as dists
from torch.nn import functional as F
from transformers import __version__
from transformers.generation.configuration_utils import (
    GenerationConfig
)
from transformers.utils import (
    ModelOutput,
    is_torchdynamo_compiling,
    logging,
)

logger = logging.get_logger(__name__)

def get_num_transfer_tokens(mask_index, steps):
    '''
    In the reverse process, the interval [0, 1] is uniformly discretized into steps intervals.
    Furthermore, because LLaDA employs a linear noise schedule (as defined in Eq. (8)),
    the expected number of tokens transitioned at each step should be consistent.

    This function is designed to precompute the number of tokens that need to be transitioned at each step.
    '''
    mask_num = mask_index.sum(dim=1, keepdim=True)

    base = mask_num // steps
    remainder = mask_num % steps

    num_transfer_tokens = torch.zeros(mask_num.size(0), steps, device=mask_index.device, dtype=torch.int64) + base

    for i in range(mask_num.size(0)):
        num_transfer_tokens[i, :remainder[i]] += 1

    return num_transfer_tokens

def top_p_logits(logits, top_p=None):
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
    sorted_indices_to_remove = cumulative_probs > top_p

    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0

    mask = torch.zeros_like(logits, dtype=torch.bool, device=logits.device)
    mask = mask.scatter_(-1, sorted_indices, sorted_indices_to_remove)
    logits = logits.masked_fill(mask, torch.finfo(logits.dtype).min)
    return logits

def top_k_logits(logits, top_k=None):
    top_k = min(top_k, logits.size(-1))

    indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
    logits = logits.masked_fill(indices_to_remove, torch.finfo(logits.dtype).min)
    return logits

def sample_tokens(logits, temperature=0.0, top_p=None, top_k=None, margin_confidence=False, neg_entropy=False):

    if temperature > 0:
        logits = logits / temperature
    if top_p is not None and top_p < 1:
        logits = top_p_logits(logits, top_p)
    if top_k is not None:
        logits = top_k_logits(logits, top_k)
    probs = torch.softmax(logits, dim=-1)

    if temperature > 0:
        try:
            x0 = dists.Categorical(probs=probs).sample()
            confidence = torch.gather(probs, -1, x0.unsqueeze(-1)).squeeze(-1)
        except:
            confidence, x0 = probs.max(dim=-1)
    else:
        confidence, x0 = probs.max(dim=-1)

    if margin_confidence:
        sorted_probs, _ = torch.sort(probs, dim=-1, descending=True)

        top1_probs = sorted_probs[:, 0]
        top2_probs = sorted_probs[:, 1]

        confidence = top1_probs - top2_probs

    if neg_entropy:
        epsilon = 1e-10
        log_probs = torch.log(probs + epsilon)
        confidence = torch.sum(probs * log_probs, dim=-1)

    return confidence, x0

def shift_logits_for_generation(logits: torch.Tensor) -> torch.Tensor:
    return torch.cat([logits[:, :1], logits[:, :-1]], dim=1)

def get_transfer_index(logits, temperature, top_p, top_k, mask_index, x, num_transfer_tokens, threshold=None):
    confidence, x0 = sample_tokens(logits, temperature=temperature, top_p=top_p, top_k=top_k)
    x0 = torch.where(mask_index, x0, x)
    masked_confidence = torch.where(mask_index, confidence, -torch.inf)

    transfer_index = torch.zeros_like(mask_index, dtype=torch.bool, device=x.device)
    if threshold is not None:
        num_transfer_tokens = mask_index.sum(dim=1, keepdim=True)

    for batch_idx in range(masked_confidence.shape[0]):
        if int(num_transfer_tokens[batch_idx]) == 0:
            continue
        _, select_index = torch.topk(masked_confidence[batch_idx], k=int(num_transfer_tokens[batch_idx]))
        transfer_index[batch_idx, select_index] = True
        if threshold is not None:
            for k in range(1, int(num_transfer_tokens[batch_idx])):
                if masked_confidence[batch_idx, select_index[k]] < threshold:
                    transfer_index[batch_idx, select_index[k]] = False
    return x0, transfer_index, masked_confidence

def get_pre_transfer_index(logits, temperature, top_p, top_k, mask_index, x, threshold=None):
    confidence, x0 = sample_tokens(logits, temperature=temperature, top_p=top_p, top_k=top_k)
    x0 = torch.where(mask_index, x0, x)
    masked_confidence = torch.where(mask_index, confidence, -torch.inf)
    transfer_index = torch.zeros_like(mask_index, dtype=torch.bool, device=x.device)

    num_transfer_tokens = mask_index.sum(dim=1, keepdim=True)
    for batch_idx in range(masked_confidence.shape[0]):
        if int(num_transfer_tokens[batch_idx]) == 0:
            continue
        _, select_index = torch.topk(masked_confidence[batch_idx], k=int(num_transfer_tokens[batch_idx]))
        transfer_index[batch_idx, select_index] = True
        if threshold is not None:
            for k in range(int(num_transfer_tokens[batch_idx])):
                if masked_confidence[batch_idx, select_index[k]] < threshold:
                    transfer_index[batch_idx, select_index[k]] = False
    return x0, transfer_index

@dataclass
class DreamModelOutput(ModelOutput):
    sequences: torch.LongTensor = None
    history: Optional[Tuple[torch.FloatTensor]] = None
    nfe: Optional[int] = None
    conf: Optional[float] = None

class DreamGenerationConfig(GenerationConfig):
    def __init__(self, **kwargs):
        self.temperature: float = kwargs.pop("temperature", 0.0)
        self.top_p: Optional[float] = kwargs.pop("top_p", None)
        self.top_k: Optional[int] = kwargs.pop("top_k", None)
        self.max_length = kwargs.pop("max_length", 20)
        self.max_new_tokens = kwargs.pop("max_new_tokens", None)

        self.eps: float = kwargs.pop("eps", 1e-3)
        self.steps: int = kwargs.pop("steps", 512)
        self.alg: str = kwargs.pop("alg", 'origin')
        self.alg_temp: Optional[float] = kwargs.pop("alg_temp", None)

        self.num_return_sequences: int = kwargs.pop("num_return_sequences", 1)
        self.return_dict_in_generate: bool = kwargs.pop("return_dict_in_generate", False)
        self.output_history: bool = kwargs.pop("output_history", False)

        self.mask_token_id = kwargs.pop("mask_token_id", None)
        self.pad_token_id = kwargs.pop("pad_token_id", None)
        self.bos_token_id = kwargs.pop("bos_token_id", None)
        self.eos_token_id = kwargs.pop("eos_token_id", None)

        self.generation_kwargs = kwargs.pop("generation_kwargs", {})

        self._from_model_config = kwargs.pop("_from_model_config", False)
        self._commit_hash = kwargs.pop("_commit_hash", None)
        self.transformers_version = kwargs.pop("transformers_version", __version__)

        if not self._from_model_config:

            for key, value in kwargs.items():
                try:
                    setattr(self, key, value)
                except AttributeError as err:
                    logger.error(f"Can't set {key} with value {value} for {self}")
                    raise err

        self.validate(is_init=True)

    def validate(self, is_init=False):
        pass

class DreamGenerationMixin:
    @staticmethod
    def _expand_inputs_for_generation(
        expand_size: int = 1,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None
    ) -> Tuple[torch.LongTensor, Dict[str, Any]]:
        """Expands tensors from [batch_size, ...] to [batch_size * expand_size, ...]"""

        if expand_size == 1:
            return input_ids, attention_mask
        if input_ids is not None:
            input_ids = input_ids.repeat_interleave(expand_size, dim=0)
        if attention_mask is not None:
            attention_mask = attention_mask.repeat_interleave(expand_size, dim=0)
        return input_ids, attention_mask

    def _validate_generated_length(self, generation_config, input_ids_length, has_default_max_length):
        """Performs validation related to the resulting generated length"""

        if is_torchdynamo_compiling():
            return

        if has_default_max_length and generation_config.max_new_tokens is None and generation_config.max_length == 20:

            warnings.warn(
                f"Using the model-agnostic default `max_length` (={generation_config.max_length}) to control the "
                "generation length. We recommend setting `max_new_tokens` to control the maximum length of the "
                "generation.",
                UserWarning,
            )
        if input_ids_length >= generation_config.max_length:
            input_ids_string = "input_ids"
            raise ValueError(
                f"Input length of {input_ids_string} is {input_ids_length}, but `max_length` is set to"
                f" {generation_config.max_length}. This can lead to unexpected behavior. You should consider"
                " increasing `max_length` or, better yet, setting `max_new_tokens`."
            )

    def _prepare_generated_length(
        self,
        generation_config,
        has_default_max_length,
        input_ids_length,
    ):
        """Prepared max and min length in generation configs to avoid clashes between similar attributes"""

        if generation_config.max_new_tokens is not None:
            if not has_default_max_length and generation_config.max_length is not None:
                logger.warning(
                    f"Both `max_new_tokens` (={generation_config.max_new_tokens}) and `max_length`(="
                    f"{generation_config.max_length}) seem to have been set. `max_new_tokens` will take precedence. "
                    "Please refer to the documentation for more information. "
                    "(https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)"
                )
            generation_config.max_length = generation_config.max_new_tokens + input_ids_length

        elif has_default_max_length:
            if generation_config.max_length == DreamGenerationConfig().max_length:
                generation_config.max_length = generation_config.max_length + input_ids_length
                max_position_embeddings = getattr(self.config, "max_position_embeddings", None)
                if max_position_embeddings is not None:
                    generation_config.max_length = min(generation_config.max_length, max_position_embeddings)

        return generation_config

    def _prepare_generation_config(
        self, generation_config: Optional[DreamGenerationConfig], **kwargs: Dict
    ) -> DreamGenerationConfig:
        """
        Prepares the base generation config, then applies any generation configuration options from kwargs. This
        function handles retrocompatibility with respect to configuration files.
        """

        using_model_generation_config = False
        if generation_config is None:
            generation_config = DreamGenerationConfig.from_model_config(self.config)
            using_model_generation_config = True

        if not is_torchdynamo_compiling():
            generation_config = copy.deepcopy(generation_config)
            _kwargs = generation_config.update(**kwargs)

            if not using_model_generation_config:
                if generation_config.bos_token_id is None:
                    generation_config.bos_token_id = self.generation_config.bos_token_id
                if generation_config.eos_token_id is None:
                    generation_config.eos_token_id = self.generation_config.eos_token_id
                if generation_config.pad_token_id is None:
                    generation_config.pad_token_id = self.generation_config.pad_token_id
                if generation_config.mask_token_id is None:
                    generation_config.mask_token_id = self.generation_config.mask_token_id

        return generation_config

    def _prepare_special_tokens(
        self,
        generation_config: DreamGenerationConfig,
        device: Optional[Union[torch.device, str]] = None,
    ):
        """
        Prepares the special tokens for generation, overwriting the generation config with their processed versions
        converted to tensor.
        Note that `generation_config` is changed in place and stops being serializable after this method is called.
        That is no problem if called within `generate` (`generation_config` is a local copy that doesn't leave the
        function). However, if called outside `generate`, consider creating a copy of `generation_config` first.
        """

        def _tensor_or_none(token, device=None):
            if token is None:
                return token

            device = device if device is not None else self.device
            if isinstance(token, torch.Tensor):
                return token.to(device)
            return torch.tensor(token, device=device, dtype=torch.long)

        bos_token_tensor = _tensor_or_none(generation_config.bos_token_id, device=device)
        eos_token_tensor = _tensor_or_none(generation_config.eos_token_id, device=device)
        pad_token_tensor = _tensor_or_none(generation_config.pad_token_id, device=device)
        mask_token_tensor = _tensor_or_none(generation_config.mask_token_id, device=device)

        if eos_token_tensor is not None and eos_token_tensor.ndim == 0:
            eos_token_tensor = eos_token_tensor.unsqueeze(0)

        if pad_token_tensor is None and eos_token_tensor is not None:
            pad_token_tensor = eos_token_tensor[0]
            logger.warning(f"Setting `pad_token_id` to `eos_token_id`:{pad_token_tensor} for open-end generation.")

        generation_config._bos_token_tensor = bos_token_tensor
        generation_config._eos_token_tensor = eos_token_tensor
        generation_config._pad_token_tensor = pad_token_tensor
        generation_config._mask_token_tensor = mask_token_tensor

    @torch.no_grad()
    def diffusion_generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        generation_config: Optional[DreamGenerationConfig] = None,
        **kwargs,
    ) -> Union[DreamModelOutput, torch.LongTensor]:

        generation_config = self._prepare_generation_config(generation_config, **kwargs)

        assert inputs is not None
        input_ids = inputs
        device = input_ids.device
        attention_mask = kwargs.pop("attention_mask", None)
        self._prepare_special_tokens(generation_config, device=device)

        input_ids_length = input_ids.shape[-1]
        has_default_max_length = kwargs.get("max_length") is None and generation_config.max_length is not None
        generation_config = self._prepare_generated_length(
            generation_config=generation_config,
            has_default_max_length=has_default_max_length,
            input_ids_length=input_ids_length,
        )

        self._validate_generated_length(generation_config, input_ids_length, has_default_max_length)

        if not is_torchdynamo_compiling() and self.device.type != input_ids.device.type:
            warnings.warn(
                "You are calling .generate() with the `input_ids` being on a device type different"
                f" than your model's device. `input_ids` is on {input_ids.device.type}, whereas the model"
                f" is on {self.device.type}. You may experience unexpected behaviors or slower generation."
                " Please make sure that you have put `input_ids` to the"
                f" correct device by calling for example input_ids = input_ids.to('{self.device.type}') before"
                " running `.generate()`.",
                UserWarning,
            )
        if (
            hasattr(generation_config, "pad_token_id") and
            torch.any(input_ids == generation_config.pad_token_id) and
            attention_mask is None
        ):
            warnings.warn(
                "Padding was detected but no attention mask is passed here. For correct "
                "generation results, please set `attention_mask` when batch-padding inputs.",
                UserWarning,
            )

        input_ids, attention_mask = self._expand_inputs_for_generation(
            expand_size=generation_config.num_return_sequences,
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        threshold = kwargs.get("threshold", 0.9)
        threshold_early = kwargs.get("threshold_early", None)
        block_length = kwargs.get("block_length", 32)
        dual_cache = kwargs.get("dual_cache", False)
        use_cluster = kwargs.get("use_cluster", True)

        result = self._sample(
            input_ids,
            attention_mask=attention_mask,
            generation_config=generation_config,
            threshold=threshold,
            threshold_early=threshold_early,
            block_length=block_length,
            dual_cache=dual_cache,
            use_cluster=use_cluster,
        )
        return result

    def _sample(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.LongTensor],
        generation_config: DreamGenerationConfig,
        threshold: Optional[float] = 0.9,
        threshold_early: Optional[float] = None,
        block_length: Optional[int] = 32,
        dual_cache: bool = False,
        use_cluster: bool = True,
    ) -> Union[DreamModelOutput, torch.LongTensor]:

        output_history = generation_config.output_history
        return_dict_in_generate = generation_config.return_dict_in_generate
        max_length = generation_config.max_length
        mask_token_id = generation_config.mask_token_id
        steps = generation_config.steps
        temperature = generation_config.temperature
        top_p = generation_config.top_p
        top_k = generation_config.top_k
        alg = generation_config.alg
        alg_temp = generation_config.alg_temp
        nfe = 0
        conf_sum = 0

        histories = [] if (return_dict_in_generate and output_history) else None

        x = F.pad(input_ids, (0, max_length - input_ids.shape[1]), value=mask_token_id)
        gen_length = max_length - input_ids.shape[1]

        if block_length is None:
            block_length = gen_length

        assert gen_length % block_length == 0, f"gen_length ({gen_length}) must be divisible by block_length ({block_length})"
        num_blocks = gen_length // block_length

        assert steps % num_blocks == 0, f"steps ({steps}) must be divisible by num_blocks ({num_blocks})"
        steps_per_block = steps // num_blocks
        timesteps = torch.linspace(1, generation_config.eps, steps_per_block + 1, device=x.device)

        if attention_mask is not None and torch.any(attention_mask == 0.0):

            attention_mask = F.pad(attention_mask, (0, max_length - attention_mask.shape[1]), value=1.0)
            tok_idx = attention_mask.long().cumsum(-1) - 1
            tok_idx.masked_fill_(attention_mask == 0, 1)

            attention_mask = torch.logical_and(
                attention_mask.unsqueeze(1).unsqueeze(-2),
                attention_mask.unsqueeze(1).unsqueeze(-1),
            )
        else:
            tok_idx = None
            attention_mask = "full"

        past_key_values = None
        past_hidden_states = None
        clustered_hidden_states = None

        if hasattr(self, "reset_dual_cache_debug_state"):
            self.reset_dual_cache_debug_state()

        for num_block in range(num_blocks):
            current_block_start = input_ids.shape[1] + num_block * block_length
            current_block_end = current_block_start + block_length
            if dual_cache:
                if not use_cluster:
                    model_output = self(x, attention_mask, tok_idx, use_cache=True, use_cluster=False)
                    past_key_values = model_output.past_key_values
                    logits = shift_logits_for_generation(model_output.logits)
                    confidence, x0 = sample_tokens(logits, temperature=temperature, top_p=top_p, top_k=top_k)
                    x[:, current_block_start] = x0[:, current_block_start]
                    nfe += 1

                    replace_position = torch.zeros_like(x, dtype=torch.bool)
                    replace_position[:, current_block_start:current_block_end] = 1

                    i = 1
                    while True:
                        mask_index = (x[:, current_block_start:current_block_end] == mask_token_id)
                        if mask_index.sum().item() == 0:
                            break

                        current_attention_mask = (
                            attention_mask
                            if attention_mask == "full"
                            else attention_mask[:, :, :, current_block_start:]
                        )

                        model_output = self(
                            x[:, current_block_start:current_block_end],
                            current_attention_mask,
                            tok_idx[:, current_block_start:current_block_end] if tok_idx is not None else None,
                            past_key_values=past_key_values,
                            use_cache=True,
                            dual_cache=True,
                            replace_position=replace_position,
                            use_cluster=False,
                        )
                        logits = shift_logits_for_generation(model_output.logits)
                        nfe += 1

                        if alg == 'confidence_threshold':
                            mask_logits = logits[mask_index]
                            confidence, x0 = sample_tokens(mask_logits, temperature=temperature, top_p=top_p, top_k=top_k)
                            x_ = torch.zeros_like(
                                x[:, current_block_start:current_block_end],
                                device=self.device,
                                dtype=torch.long,
                            ) + mask_token_id
                            full_confidence = torch.full_like(
                                x[:, current_block_start:current_block_end],
                                -torch.inf,
                                device=self.device,
                                dtype=logits.dtype,
                            )

                            x_[mask_index] = x0.clone()
                            full_confidence[mask_index] = confidence
                            full_confidence[:, block_length:] = -torch.inf

                            current_transfer_tokens = (x[:, current_block_start:current_block_end] == mask_token_id).sum()
                            selected_confidence, select_index = torch.topk(full_confidence, current_transfer_tokens)
                            transfer_index = torch.zeros_like(x_, device=x.device, dtype=torch.bool)

                            select_index = select_index.to(x.device)
                            transfer_index[0, select_index[0]] = True
                            for k in range(1, current_transfer_tokens):
                                if selected_confidence[0, k] < threshold:
                                    transfer_index[0, select_index[0, k]] = False
                            x[:, current_block_start:current_block_end][transfer_index] = x_[transfer_index]
                        else:
                            if i == steps_per_block:
                                break
                            t = timesteps[i]
                            s = timesteps[i + 1]
                            mask_index[:, block_length:] = False
                            mask_logits = logits[mask_index]
                            confidence, x0 = sample_tokens(
                                mask_logits,
                                temperature,
                                top_p=top_p,
                                top_k=top_k,
                                neg_entropy=True,
                            )
                            num_mask_token = mask_index.sum() / mask_index.shape[0]
                            number_transfer_tokens = (
                                int(num_mask_token * (1 - s / t))
                                if i < steps_per_block - 1
                                else int(num_mask_token)
                            )
                            full_confidence = torch.full_like(
                                x[:, current_block_start:current_block_end],
                                -torch.inf,
                                device=self.device,
                                dtype=logits.dtype,
                            )
                            full_confidence[mask_index] = confidence
                            full_confidence[:, block_length:] = -torch.inf

                            if number_transfer_tokens > 0:
                                if alg_temp is None or alg_temp == 0:
                                    _, transfer_index = torch.topk(full_confidence, number_transfer_tokens)
                                else:
                                    full_confidence = full_confidence / alg_temp
                                    full_confidence = F.softmax(full_confidence, dim=-1)
                                    transfer_index = torch.multinomial(
                                        full_confidence, num_samples=number_transfer_tokens
                                    )
                                x_ = torch.zeros_like(
                                    x[:, current_block_start:current_block_end],
                                    device=self.device,
                                    dtype=torch.long,
                                ) + mask_token_id
                                x_[mask_index] = x0.clone()
                                row_indices = torch.arange(x.size(0), device=self.device).unsqueeze(1).expand_as(transfer_index)
                                x[:, current_block_start:current_block_end][row_indices, transfer_index] = x_[
                                    row_indices, transfer_index
                                ]
                        i += 1
                    continue

                dec_tok_number_until_update = 0
                replace_position = torch.zeros_like(x, dtype=torch.bool)
                replace_position[:, current_block_start:current_block_end] = 1
                block_mask_index = (x[:, current_block_start:current_block_end] == mask_token_id)
                num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps_per_block)
                supports_efficient_window = True

                if block_mask_index.sum().item() == 0:
                    continue

                if supports_efficient_window and num_block % 4 == 3 and past_key_values is not None:
                    block_prefix_window = max(0, current_block_start - 2 * block_length)
                    block_suffix_window = min(x.shape[1], current_block_end + block_length)
                    x_window = x[:, block_prefix_window:block_suffix_window]
                    efficient_window_position = torch.zeros_like(x, dtype=torch.bool)
                    efficient_window_position[:, block_prefix_window:block_suffix_window] = 1
                    current_attention_mask = (
                        attention_mask
                        if attention_mask == "full"
                        else attention_mask[:, :, block_prefix_window:block_suffix_window, :]
                    )
                    model_output = self(
                        x_window,
                        current_attention_mask,
                        tok_idx[:, block_prefix_window:block_suffix_window] if tok_idx is not None else None,
                        past_key_values=past_key_values,
                        use_cache=True,
                        dual_cache=True,
                        replace_position=replace_position,
                        cluster=False,
                        efficient=True,
                        efficient_window_position=efficient_window_position,
                        get_token_drift_score=True,
                    )
                    local_block_start = current_block_start - block_prefix_window
                    local_block_end = current_block_end - block_prefix_window
                    block_logits = shift_logits_for_generation(model_output.logits)[:, local_block_start:local_block_end, :]
                    past_key_values = model_output.past_key_values
                    if getattr(model_output, "past_hidden_states", None) is not None:
                        if past_hidden_states is None:
                            past_hidden_states = model_output.past_hidden_states
                        else:
                            for layer_idx, layer_hidden in enumerate(model_output.past_hidden_states):
                                past_hidden_states[layer_idx][:, block_prefix_window:block_suffix_window, :] = layer_hidden
                    clustered_hidden_states = getattr(model_output, "clustered_hidden_states", None)
                else:
                    model_output = self(
                        x,
                        attention_mask,
                        tok_idx,
                        use_cache=True,
                        dual_cache=True,
                        replace_position=replace_position,
                        cluster=False,
                        get_token_drift_score=True,
                    )
                    past_key_values = model_output.past_key_values
                    past_hidden_states = getattr(model_output, "past_hidden_states", None)
                    clustered_hidden_states = getattr(model_output, "clustered_hidden_states", None)
                    block_logits = shift_logits_for_generation(model_output.logits)[:, current_block_start:current_block_end, :]

                nfe += 1
                mask_index_block = (x[:, current_block_start:current_block_end] == mask_token_id)
                x_block = x[:, current_block_start:current_block_end]
                x0_block, transfer_index, _ = get_transfer_index(
                    block_logits,
                    temperature,
                    top_p,
                    top_k,
                    mask_index_block,
                    x_block,
                    num_transfer_tokens[:, 0] if threshold is None else None,
                    threshold,
                )
                x[:, current_block_start:current_block_end][transfer_index] = x0_block[transfer_index]

                i = 1
                drift_window = []
                while True:
                    if (x[:, current_block_start:current_block_end] == mask_token_id).sum().item() == 0:
                        break

                    nfe += 1
                    dec_tok_number_until_update += transfer_index.sum().item()
                    mask_index = (x[:, current_block_start:current_block_end] == mask_token_id)
                    cluster = transfer_index.sum().item() > 1 or dec_tok_number_until_update > 2
                    if cluster:
                        dec_tok_number_until_update = 0
                    efficient = False
                    topk = 4

                    current_attention_mask = (
                        attention_mask
                        if attention_mask == "full"
                        else attention_mask[:, :, current_block_start:current_block_end, :]
                    )
                    model_output = self(
                        x[:, current_block_start:current_block_end],
                        current_attention_mask,
                        tok_idx[:, current_block_start:current_block_end] if tok_idx is not None else None,
                        past_key_values=past_key_values,
                        use_cache=True,
                        dual_cache=True,
                        replace_position=replace_position,
                        topk=topk,
                        cluster=cluster,
                        efficient=efficient,
                        past_hidden_states=past_hidden_states,
                        clustered_hidden_states=clustered_hidden_states,
                        get_token_drift_score=True,
                    )
                    logits = shift_logits_for_generation(model_output.logits)
                    past_key_values = model_output.past_key_values
                    if cluster:
                        past_hidden_states = getattr(model_output, "past_hidden_states", past_hidden_states)
                        clustered_hidden_states = getattr(model_output, "clustered_hidden_states", clustered_hidden_states)
                    logits_sel = getattr(model_output, "logits_sel", None)
                    sel_indices = getattr(model_output, "sel_indices", None)
                    drift_scores = getattr(model_output, "drift_scores", None)

                    delta = None
                    if drift_scores is not None:
                        d = drift_scores[-1] if isinstance(drift_scores, (list, tuple)) else drift_scores
                        if d is not None:
                            if d.dim() == 2:
                                d = d[0]
                            d_block = d[current_block_start:current_block_end] if d.size(0) > block_length else d
                            if len(drift_window) > 0:
                                prev_stack = torch.stack(drift_window, dim=0)
                                delta = d_block - prev_stack.mean(dim=0)
                            else:
                                delta = d_block
                            drift_window.append(d_block.detach())
                            if len(drift_window) > 5:
                                drift_window.pop(0)

                    x0, transfer_index, conf = get_transfer_index(
                        logits,
                        temperature,
                        top_p,
                        top_k,
                        mask_index,
                        x[:, current_block_start:current_block_end],
                        num_transfer_tokens[:, i] if threshold is None else None,
                        threshold,
                    )

                    if threshold_early is not None and delta is not None:
                        conf_block = conf
                        diff = torch.clamp(0.9 - conf_block, min=0.0)
                        diff = 10 * (diff ** 2)
                        dynamic_gate = diff
                        early_conf_mask = (conf_block > threshold_early) & (conf_block < threshold)
                        drift_mask = (delta.unsqueeze(0) >= dynamic_gate) & (delta.unsqueeze(0) < 2.0)
                        early_commit_block = early_conf_mask & drift_mask & mask_index
                        transfer_index |= early_commit_block

                    x[:, current_block_start:current_block_end][transfer_index] = x0[transfer_index]

                    if logits_sel is not None and sel_indices is not None and cluster:
                        pred_indices = sel_indices + 1
                        suffix_mask = (pred_indices >= current_block_end) & (pred_indices < x.shape[1])
                        if suffix_mask.any():
                            suffix_sel_indices = pred_indices[suffix_mask]
                            suffix_logits = logits_sel[:, suffix_mask, :]
                            suffix_actual_mask = (x[:, suffix_sel_indices] == mask_token_id)
                            if suffix_actual_mask.any():
                                x_suffix_new, transfer_idx_local = get_pre_transfer_index(
                                    suffix_logits,
                                    temperature,
                                    top_p,
                                    top_k,
                                    suffix_actual_mask,
                                    x[:, suffix_sel_indices],
                                    threshold=threshold,
                                )
                                x[:, suffix_sel_indices][transfer_idx_local] = x_suffix_new[transfer_idx_local]
                    i += 1
                continue

            x = generation_tokens_hook_func(None, x, None)
            i = 0

            while True:
                remaining_masks = (x[:, current_block_start:current_block_end] == mask_token_id).sum()
                if remaining_masks == 0:
                    break
                mask_index = (x == mask_token_id)
                mask_index[:, :current_block_start] = False
                mask_index[:, current_block_end:] = False

                logits = self(x, attention_mask, tok_idx).logits
                logits = shift_logits_for_generation(logits)
                nfe += 1

                logits = generation_logits_hook_func(i, x, logits)

                mask_logits = logits[mask_index]
                if not alg == 'confidence_threshold':
                    t = timesteps[i]
                    s = timesteps[i + 1]

                if alg == 'origin':
                    p_transfer = 1 - s / t if i < steps - 1 else 1
                    x0 = torch.zeros_like(x[mask_index], device=self.device, dtype=torch.long) + mask_token_id
                    transfer_index_t_s = torch.rand(*x0.shape, device=self.device) < p_transfer
                    _, x0[transfer_index_t_s]= sample_tokens(mask_logits[transfer_index_t_s], temperature=temperature, top_p=top_p, top_k=top_k)
                    x[mask_index] = x0.clone()
                elif alg == 'confidence_threshold':
                    confidence, x0 = sample_tokens(mask_logits, temperature=temperature, top_p=top_p, top_k=top_k)
                    x_ = torch.zeros_like(x, device=self.device, dtype=torch.long) + mask_token_id
                    x_[mask_index] = x0.clone()
                    full_confidence = torch.full_like(x, -torch.inf, device=self.device, dtype=logits.dtype)
                    full_confidence[mask_index] = confidence
                    full_confidence[:, :current_block_start] = -torch.inf
                    full_confidence[:, current_block_end:] = -torch.inf
                    avg_confidence_unmasked = torch.mean(full_confidence[mask_index]).item()
                    conf_sum += avg_confidence_unmasked
                    current_transfer_tokens = remaining_masks.item()

                    if current_transfer_tokens > 0:
                        full_confidence_flat = full_confidence.view(-1)
                        selected_confidence, select_index = torch.topk(
                            full_confidence_flat, min(current_transfer_tokens, full_confidence_flat.numel())
                        )
                        transfer_index = torch.zeros_like(full_confidence_flat, device=x.device, dtype=torch.bool)
                        num_selected = 0
                        select_index = select_index.to(x.device)
                        for k in range(len(selected_confidence)):
                            if selected_confidence[k] >= threshold:
                                transfer_index[select_index[k]] = True
                                num_selected += 1
                        if num_selected == 0 and len(selected_confidence) > 0:
                            transfer_index[select_index[0]] = True
                        transfer_index = transfer_index.view_as(x)
                        x[transfer_index] = x_[transfer_index]
                else:
                    if i == steps_per_block:
                        break
                    t = timesteps[i]
                    s = timesteps[i + 1]
                    mask_index[:, block_length:] = False
                    mask_logits = logits[mask_index]
                    confidence, x0 = sample_tokens(mask_logits, temperature, top_p=top_p, top_k=top_k, neg_entropy=True)
                    num_mask_token = mask_index.sum() / mask_index.shape[0]
                    number_transfer_tokens = int(num_mask_token * (1 - s / t)) if i < steps_per_block - 1 else int(num_mask_token)
                    full_confidence = torch.full_like(x[:, current_block_start:], -torch.inf, device=self.device, dtype=logits.dtype)
                    full_confidence[mask_index[:, current_block_start:]] = confidence

                    if number_transfer_tokens > 0:
                        if alg_temp is None or alg_temp == 0:
                            _, transfer_index = torch.topk(full_confidence, number_transfer_tokens)
                        else:
                            full_confidence = full_confidence / alg_temp
                            full_confidence = F.softmax(full_confidence, dim=-1)
                            transfer_index = torch.multinomial(full_confidence, num_samples=number_transfer_tokens)
                        x_ = torch.zeros_like(x[:, current_block_start:], device=self.device, dtype=torch.long) + mask_token_id
                        x_[mask_index[:, current_block_start:]] = x0.clone()
                        row_indices = torch.arange(x.size(0), device=self.device).unsqueeze(1).expand_as(transfer_index)
                        x[:, current_block_start:][row_indices, transfer_index] = x_[row_indices, transfer_index]
                i += 1

        if return_dict_in_generate:
            return DreamModelOutput(
                sequences=x,
                history=histories,
                nfe=nfe,
                conf=float('nan'),
            )
        else:
            return x,nfe
