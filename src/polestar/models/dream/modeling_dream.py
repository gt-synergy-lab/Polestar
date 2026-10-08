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

"""PyTorch Dream model."""

import math
import os
from typing import List, Optional, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import nn

from transformers.activations import ACT2FN
from transformers.cache_utils import Cache, DynamicCache
from transformers.modeling_outputs import (
    BaseModelOutput,
    MaskedLMOutput,
)
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import (
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    is_flash_attn_2_available,
    is_flash_attn_greater_or_equal_2_10,
    logging,
)
from transformers import PretrainedConfig
from .configuration_dream import DreamConfig
from .generation import DreamGenerationMixin, DreamGenerationConfig

if is_flash_attn_2_available():
    from transformers.modeling_flash_attention_utils import _flash_attention_forward
from .clustering import cluster_past_hidden_states

logger = logging.get_logger(__name__)

_CHECKPOINT_FOR_DOC = "Dream-7B"
_CONFIG_FOR_DOC = "DreamConfig"

class BaseModelOutputWithPast(BaseModelOutput):
    def __init__(
        self,
        last_hidden_state: torch.FloatTensor,
        hidden_states: Optional[Tuple[torch.FloatTensor]] = None,
        attentions: Optional[Tuple[torch.FloatTensor]] = None,
        past_key_values: Optional[Tuple[torch.FloatTensor]] = None,
        past_hidden_states=None,
        clustered_hidden_states=None,
        update_packet=None,
        drift_scores=None,
        logits_sel: Optional[torch.FloatTensor] = None,
        sel_indices: Optional[torch.LongTensor] = None,
        ):
        super().__init__(last_hidden_state, hidden_states, attentions)
        self.past_key_values = past_key_values

        self.past_hidden_states = past_hidden_states
        self.clustered_hidden_states = clustered_hidden_states
        self.update_packet = update_packet
        self.drift_scores = drift_scores
        self.logits_sel = logits_sel
        self.sel_indices = sel_indices

class MaskedLMOutputWithPastKeyValues(MaskedLMOutput):
    def __init__(
        self,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,

        past_hidden_states: Optional[List[torch.Tensor]] = None,
        clustered_hidden_states: Optional[List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]] = None,
        update_packet: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        drift_scores: Optional[List[torch.Tensor]] = None,
        logits_sel: Optional[torch.FloatTensor] = None,
        sel_indices: Optional[torch.LongTensor] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.past_key_values = past_key_values

        self.past_hidden_states = past_hidden_states
        self.clustered_hidden_states = clustered_hidden_states
        self.update_packet = update_packet
        self.drift_scores = drift_scores
        self.logits_sel = logits_sel
        self.sel_indices = sel_indices

# Copied from transformers.models.llama.modeling_llama.LlamaRMSNorm with Llama->Dream
class DreamRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        DreamRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"

# Copied from transformers.models.llama.modeling_llama.LlamaRotaryEmbedding with Llama->Dream
class DreamRotaryEmbedding(nn.Module):
    def __init__(
        self,
        dim=None,
        max_position_embeddings=2048,
        base=10000,
        device=None,
        scaling_factor=1.0,
        rope_type="default",
        config: Optional[DreamConfig] = None,
    ):
        super().__init__()

        self.rope_kwargs = {}
        if config is None:
            logger.warning_once(
                "`DreamRotaryEmbedding` can now be fully parameterized by passing the model config through the "
                "`config` argument. All other arguments will be removed in v4.46"
            )
            self.rope_kwargs = {
                "rope_type": rope_type,
                "factor": scaling_factor,
                "dim": dim,
                "base": base,
                "max_position_embeddings": max_position_embeddings,
            }
            self.rope_type = rope_type
            self.max_seq_len_cached = max_position_embeddings
            self.original_max_seq_len = max_position_embeddings
        else:

            if config.rope_scaling is not None:
                self.rope_type = config.rope_scaling.get("rope_type", config.rope_scaling.get("type"))
            else:
                self.rope_type = "default"
            self.max_seq_len_cached = config.max_position_embeddings
            self.original_max_seq_len = config.max_position_embeddings

        self.config = config
        self.rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]

        inv_freq, self.attention_scaling = self.rope_init_fn(self.config, device, **self.rope_kwargs)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.original_inv_freq = self.inv_freq

    def reset_parameters(self):
        inv_freq, self.attention_scaling = self.rope_init_fn(self.config, self.inv_freq.device, **self.rope_kwargs)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.original_inv_freq = self.inv_freq

    def _dynamic_frequency_update(self, position_ids, device):
        """
        dynamic RoPE layers should recompute `inv_freq` in the following situations:
        1 - growing beyond the cached sequence length (allow scaling)
        2 - the current sequence length is in the original scale (avoid losing precision with small sequences)
        """
        seq_len = torch.max(position_ids) + 1
        if seq_len > self.max_seq_len_cached:
            inv_freq, self.attention_scaling = self.rope_init_fn(
                self.config, device, seq_len=seq_len, **self.rope_kwargs
            )
            self.register_buffer("inv_freq", inv_freq, persistent=False)
            self.max_seq_len_cached = seq_len

        if seq_len < self.original_max_seq_len and self.max_seq_len_cached > self.original_max_seq_len:
            self.register_buffer("inv_freq", self.original_inv_freq, persistent=False)
            self.max_seq_len_cached = self.original_max_seq_len

    @torch.no_grad()
    def forward(self, x, position_ids):
        if "dynamic" in self.rope_type:
            self._dynamic_frequency_update(position_ids, device=x.device)

        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        position_ids_expanded = position_ids[:, None, :].float()

        device_type = x.device.type
        device_type = device_type if isinstance(device_type, str) and device_type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()

        cos = cos * self.attention_scaling
        sin = sin * self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)

# Copied from transformers.models.llama.modeling_llama.rotate_half
def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

# Copied from transformers.models.llama.modeling_llama.apply_rotary_pos_emb
def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1, block_end_index: Optional[torch.Tensor] = None):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`, *optional*):
            Deprecated and unused.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    query_len, key_len = q.shape[-2], k.shape[-2]

    if block_end_index is None:
        q_embed = (q * cos[:, :, key_len - query_len : key_len, :]) + (rotate_half(q) * sin[:, :, key_len - query_len : key_len, :])
    else:
        q_embed = (q * cos[:, :, block_end_index.item() - query_len : block_end_index.item(), :]) + (rotate_half(q) * sin[:, :, block_end_index.item() - query_len : block_end_index.item(), :])
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

# Copied from transformers.models.mistral.modeling_mistral.MistralMLP with Mistral->Dream
class DreamMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, hidden_state):
        return self.down_proj(self.act_fn(self.gate_proj(hidden_state)) * self.up_proj(hidden_state))

# Copied from transformers.models.llama.modeling_llama.repeat_kv
def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)

class DreamAttention(nn.Module):
    """
    Multi-headed attention from 'Attention Is All You Need' paper. Modified to use sliding window attention: Longformer
    and "Generating Long Sequences with Sparse Transformers".
    """

    def __init__(self, config: DreamConfig, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        if layer_idx is None:
            logger.warning_once(
                f"Instantiating {self.__class__.__name__} without passing `layer_idx` is not recommended and will "
                "to errors during the forward call, if caching is used. Please make sure to provide a `layer_idx` "
                "when creating this class."
            )

        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.is_causal = False
        self.attention_dropout = config.attention_dropout

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=True)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=True)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=True)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

        self.rotary_emb = DreamRotaryEmbedding(config=self.config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        if position_embeddings is None:
            logger.warning_once(
                "The attention layers in this model are transitioning from computing the RoPE embeddings internally "
                "through `position_ids` (2D tensor with the indexes of the tokens), to using externally computed "
                "`position_embeddings` (Tuple of tensors, containing cos and sin). In v4.46 `position_ids` will be "
                "removed and `position_embeddings` will be mandatory."
            )
            cos, sin = self.rotary_emb(value_states, position_ids)
        else:
            cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)
        if attention_mask is not None:
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
            attn_weights = attn_weights + causal_mask

        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value

class DreamSdpaAttention(DreamAttention):
    """
    Dream attention module using torch.nn.functional.scaled_dot_product_attention. This module inherits from
    `DreamAttention` as the weights of the module stays untouched. The only changes are on the forward pass to adapt to
    SDPA API.
    """

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        replace_position: Optional[torch.Tensor] = None,
        dual_cache: Optional[bool] = False,
        layer_id: Optional[int] = None,
        hidden_state_past: Optional[torch.Tensor] = None,
        output_attn_weights: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        if output_attentions:

            logger.warning_once(
                "DreamModel is using DreamSdpaAttention, but `torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to the manual attention implementation, "
                'but specifying the manual implementation will be required from Transformers version v5.0.0 onwards. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
            )
            return super().forward(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
            )

        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)
        replace_indices = None
        had_past_key_value = past_key_value is not None

        if dual_cache and replace_position is not None:
            replace_indices = replace_position.nonzero(as_tuple=True)[1]

        if past_key_value is not None:
            if dual_cache:
                past_key, past_value = past_key_value
                past_key[:, replace_indices] = key_states
                key_states = past_key
                past_value[:, replace_indices] = value_states
                value_states = past_value
            else:
                past_key, past_value = past_key_value
                key_states = torch.cat([past_key, key_states], dim=-2)
                value_states = torch.cat([past_value, value_states], dim=-2)

        past_key_value = (key_states, value_states) if use_cache else None

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, -1, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, -1, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        if position_embeddings is None:
            logger.warning_once(
                "The attention layers in this model are transitioning from computing the RoPE embeddings internally "
                "through `position_ids` (2D tensor with the indexes of the tokens), to using externally computed "
                "`position_embeddings` (Tuple of tensors, containing cos and sin). In v4.46 `position_ids` will be "
                "removed and `position_embeddings` will be mandatory."
            )
            cos, sin = self.rotary_emb(value_states, position_ids)
        else:
            cos, sin = position_embeddings

        if dual_cache and had_past_key_value and replace_indices is not None and replace_indices.numel() > 0:
            query_states, key_states = apply_rotary_pos_emb(
                query_states,
                key_states,
                cos,
                sin,
                block_end_index=replace_indices.max() + 1,
            )
        else:
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        rope_k = key_states
        rope_v = value_states

        if query_states.device.type == "cuda" and attention_mask is not None:
            query_states = query_states.contiguous()
            key_states = key_states.contiguous()
            value_states = value_states.contiguous()

        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=attention_mask if isinstance(attention_mask, torch.Tensor) else None,
            dropout_p=self.attention_dropout if self.training else 0.0,
            is_causal=False,
        )
        attn_weights = None
        if output_attn_weights:
            scale = 1.0 / math.sqrt(self.head_dim)
            attn_weights = torch.matmul(query_states, key_states.transpose(-2, -1)) * scale
            attn_weights = torch.softmax(attn_weights, dim=-1)

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(bsz, q_len, self.hidden_size)

        attn_output = self.o_proj(attn_output)

        return attn_output, attn_weights, past_key_value, rope_k, rope_v

class DreamDecoderLayer(nn.Module):
    def __init__(self, config: DreamConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size

        if config.sliding_window and config._attn_implementation != "flash_attention_2":
            logger.warning_once(
                f"Sliding Window Attention is enabled but not implemented for `{config._attn_implementation}`; "
                "unexpected results may be encountered."
            )

        self.self_attn = DreamSdpaAttention(config, layer_idx)

        self.mlp = DreamMLP(config)
        self.input_layernorm = DreamRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = DreamRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        position_embeddings_for_cluster: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        dual_cache: Optional[bool] = False,
        replace_position: Optional[torch.Tensor] = None,
        layer_id: Optional[int] = None,
        hidden_state_past: Optional[torch.Tensor] = None,
        clustered_hidden_state: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None,
        update_packet_prev: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        cluster: bool = False,
        topk: Optional[int] = None,
        efficient: bool = False,
        efficient_window_position: Optional[torch.Tensor] = None,
        get_token_drift_score: bool = False,
        use_cluster: bool = True,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*): attention mask of size
                `(batch, sequence_length)` where padding elements are indicated by 0.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
            cache_position (`torch.LongTensor` of shape `(sequence_length)`, *optional*):
                Indices depicting the position of the input sequence tokens in the sequence.
            position_embeddings (`Tuple[torch.FloatTensor, torch.FloatTensor]`, *optional*):
                Tuple containing the cosine and sine positional embeddings of shape `(batch_size, seq_len, head_dim)`,
                with `head_dim` being the embedding dimension of each attention head.
            kwargs (`dict`, *optional*):
                Arbitrary kwargs to be ignored, used for FSDP and other methods that injects code
                into the model
        """

        residual = hidden_states
        cluster = cluster or (efficient and (layer_id > 28 or layer_id == 0))
        current_hidden_state = None
        window_replace_position = efficient_window_position if efficient and efficient_window_position is not None else replace_position
        local_q_start = None
        local_q_end = None
        if layer_id == self.self_attn.layer_idx == self.self_attn.config.num_hidden_layers - 1:
            topk = 1

        idx = None
        if update_packet_prev is not None and hidden_state_past is not None and cluster and use_cluster:
            idx_raw, h_new_raw = update_packet_prev
            if clustered_hidden_state is not None:
                _, current_labels, _ = clustered_hidden_state
                active_mask = (current_labels[0, idx_raw] != -1)
                idx = idx_raw[active_mask]
                h_new = h_new_raw[:, active_mask, :]
            else:
                idx = idx_raw
                h_new = h_new_raw

            idx = idx.to(hidden_state_past.device)
            if idx.numel() == 0:
                h_new = None

            if h_new is not None:
                hidden_state_past = hidden_state_past.clone()
                h_old = hidden_state_past[:, idx, :].clone()
                hidden_state_past[:, idx, :] = h_new

                if past_key_value is not None:
                    pk, pv = past_key_value

                    h_new_norm = self.input_layernorm(h_new)
                    k_new_raw = self.self_attn.k_proj(h_new_norm)
                    v_new_raw = self.self_attn.v_proj(h_new_norm)

                    Lpk = pk.shape[1]
                    assert idx.max().item() < Lpk, f"idx max {idx.max().item()} >= cache len {Lpk}"

                    pk = pk.clone()
                    pv = pv.clone()
                    pk[:, idx, :] = k_new_raw
                    pv[:, idx, :] = v_new_raw
                    past_key_value = (pk, pv)

                if clustered_hidden_state is not None:
                    centroids, labels, counts = clustered_hidden_state
                    B_c, K, _ = centroids.shape

                    cluster_total_counts = counts
                    labels_sel = labels[:, idx]
                    valid_sel_mask = (labels_sel >= 0) & (labels_sel < K)
                    safe_labels_sel = labels_sel.clamp(min=0)

                    labels_sel_onehot = torch.nn.functional.one_hot(
                        safe_labels_sel, num_classes=K
                    ).to(dtype=centroids.dtype)
                    labels_sel_onehot = labels_sel_onehot * valid_sel_mask.unsqueeze(-1)

                    sum_old = torch.einsum("bsk,bsd->bkd", labels_sel_onehot, h_old.to(centroids.dtype))
                    sum_new = torch.einsum("bsk,bsd->bkd", labels_sel_onehot, h_new.to(centroids.dtype))

                    delta = sum_new - sum_old
                    eps = 1e-6
                    denom = cluster_total_counts + eps
                    valid_mask_c = cluster_total_counts > 0
                    centroids_update = centroids + delta / denom
                    centroids = torch.where(valid_mask_c, centroids_update, centroids)

                    clustered_hidden_state = (centroids, labels, counts)

        if (
            use_cache
            and use_cluster
            and hidden_state_past is None
            and clustered_hidden_state is None
            and replace_position is not None
        ):
            current_hidden_state = hidden_states if use_cache else None
            block_mask = replace_position[0]
            L_total = block_mask.size(0)
            block_pos = block_mask.nonzero(as_tuple=True)[0]
            curr_start = block_pos[0].item()
            curr_end = block_pos[-1].item() + 1

            window_size = 32
            if efficient and efficient_window_position is not None:
                window_pos = efficient_window_position[0].nonzero(as_tuple=True)[0]
                w_start = window_pos[0].item()
                w_end = window_pos[-1].item() + 1
                local_q_start = curr_start - w_start
                local_q_end = curr_end - w_start
                window_replace_position = efficient_window_position
            else:
                w_start = max(0, curr_start - 2 * window_size)
                w_end = min(L_total, curr_end + window_size)
                local_q_start = curr_start - w_start
                local_q_end = curr_end - w_start
                window_replace_position = torch.zeros_like(replace_position)
                window_replace_position[:, w_start:w_end] = 1

            if current_hidden_state is not None and current_hidden_state.size(1) < L_total and efficient:
                pre_window = current_hidden_state[:, : curr_start - w_start, :]
                post_window = current_hidden_state[:, curr_end - w_start : w_end - w_start, :]
            else:
                pre_window = current_hidden_state[:, w_start:curr_start, :]
                post_window = current_hidden_state[:, curr_end:w_end, :]
            total_window_len = (curr_start - w_start) + (w_end - curr_end)

            target_total = 8
            num_pre_clusters = max(1, target_total * (curr_start - w_start) // max(total_window_len, 1))
            num_pre_clusters = min(target_total - 1, num_pre_clusters)
            if pre_window.size(1) == 0:
                num_pre_clusters = 0
            num_post_clusters = target_total - num_pre_clusters

            all_centroids = []
            full_labels = torch.full((hidden_states.size(0), L_total), -1, dtype=torch.long, device=hidden_states.device)
            if pre_window.size(1) > 0:
                pre_centroids, pre_labels = cluster_past_hidden_states([pre_window.detach()], num_clusters=num_pre_clusters)[0]
                all_centroids.append(pre_centroids)
                full_labels[:, w_start:curr_start] = pre_labels
            else:
                all_centroids.append(torch.empty((hidden_states.size(0), 0, hidden_states.size(-1)), device=hidden_states.device))

            if post_window.size(1) > 0:
                post_centroids, post_labels = cluster_past_hidden_states([post_window.detach()], num_clusters=num_post_clusters)[0]
                all_centroids.append(post_centroids)
                full_labels[:, curr_end:w_end] = post_labels + num_pre_clusters
            else:
                all_centroids.append(torch.empty((hidden_states.size(0), 0, hidden_states.size(-1)), device=hidden_states.device))

            centroids = torch.cat(all_centroids, dim=1).to(hidden_states.dtype)
            B_c, K, _ = centroids.shape

            valid = (full_labels >= 0) & (full_labels < K)
            labels_safe = full_labels.clamp(min=0)
            counts = torch.zeros((B_c, K), device=hidden_states.device, dtype=torch.float32)
            counts.scatter_add_(1, labels_safe, valid.to(counts.dtype))
            counts = counts.unsqueeze(-1).to(hidden_states.dtype)

            clustered_hidden_state = (centroids, full_labels, counts)

        hidden_states = self.input_layernorm(hidden_states)

        hidden_states, attn_weights, present_key_value, rope_k, rope_v = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            dual_cache=dual_cache,
            replace_position=replace_position if not efficient else window_replace_position,
            layer_id=layer_id,
            hidden_state_past=hidden_state_past,
            output_attn_weights=get_token_drift_score and layer_id == self.self_attn.config.num_hidden_layers - 1,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        drift_score = None
        if attn_weights is not None and get_token_drift_score and layer_id == self.self_attn.config.num_hidden_layers - 1:
            block_mask = replace_position[0] if replace_position is not None else None
            if block_mask is not None:
                L_total = block_mask.size(0)
                block_pos = block_mask.nonzero(as_tuple=True)[0]
                curr_start = block_pos[0].item()
                curr_end = block_pos[-1].item() + 1
                if attn_weights.size(-2) == L_total:
                    attn_weights = attn_weights[:, :, curr_start:curr_end, :]
                elif efficient and local_q_start is not None and local_q_end is not None:
                    attn_weights = attn_weights[:, :, local_q_start:local_q_end, :]

            def token_drift_kl(curr_attn, prev_attn, eps=1e-12):
                curr = curr_attn.detach().to(torch.float32).clamp_min(eps)
                prev = prev_attn.detach().to(torch.float32).clamp_min(eps)
                kl_bhq = (curr * (torch.log(curr) - torch.log(prev))).sum(dim=-1)
                kl_bhq = kl_bhq.clamp_min(0.0)
                return kl_bhq.mean(dim=1)

            if getattr(self, "attn_weights", None) is not None:
                if self.attn_weights.shape == attn_weights.shape:
                    drift_bq = token_drift_kl(attn_weights, self.attn_weights)
                    drift_score = drift_bq[0]
            self.attn_weights = attn_weights.detach()
        elif not get_token_drift_score:
            self.attn_weights = None

        update_packet_L = None
        if (
            use_cache
            and use_cluster
            and clustered_hidden_state is not None
            and present_key_value is not None
            and (cluster or hidden_state_past is None)
        ):
            key_present, value_present = present_key_value
            centroids, labels, counts = clustered_hidden_state

            B, K, D = centroids.shape
            L_total = labels.shape[1]
            hd = self.self_attn.head_dim
            n_h = self.self_attn.num_heads
            n_hkv = self.self_attn.num_key_value_heads

            centers_norm = self.input_layernorm(centroids)
            cluster_q = self.self_attn.q_proj(centers_norm)
            cluster_q = cluster_q.view(B, K, n_h, hd).transpose(1, 2)

            key_present = key_present.view(B, -1, n_hkv, hd).transpose(1, 2)
            value_present = value_present.view(B, -1, n_hkv, hd).transpose(1, 2)
            key_mean = key_present.mean(dim=1)
            cluster_q_flat = cluster_q.mean(dim=1)

            attn_scores = torch.matmul(cluster_q_flat, key_mean.transpose(-2, -1)) / math.sqrt(hd)
            attn_scores = torch.softmax(attn_scores, dim=-1)

            use_prev = (
                hasattr(self, "_prev_attn_scores")
                and (self._prev_attn_scores is not None)
                and (hidden_state_past is not None)
                and (self._prev_attn_scores.shape == attn_scores.shape)
            )
            if use_prev:
                prev_dist = self._prev_attn_scores.detach().to(torch.float32)
                curr_dist = attn_scores.detach().to(torch.float32)

                prev_dist_32 = prev_dist.to(torch.float32)
                curr_dist_32 = curr_dist.to(torch.float32)

                eps = 1e-12

                log_curr = torch.log(curr_dist_32 + eps)
                log_prev = torch.log(prev_dist_32 + eps)

                log_delta = log_curr - log_prev
                kl_drift = (curr_dist_32 * log_delta).sum(dim=-1)
                kl_drift = torch.clamp(kl_drift, min=0.0)
                scores0 = kl_drift[0]
            else:
                scores0 = None
            self._prev_attn_scores = attn_scores.detach()
            if (replace_position is not None) and (hidden_state_past is not None) and (scores0 is not None):
                block_mask = replace_position[0]
                non_block_mask = ~block_mask

                K = centroids.shape[1]
                M_req = topk if topk is not None else 0
                M = min(M_req, K)
                if M > 0:
                    top_clusters = torch.topk(scores0, k=M, dim=-1).indices

                    labels_b0 = labels[0]
                    valid = labels_b0 >= 0
                    labels_safe = labels_b0.clamp(min=0)

                    top_mask = torch.zeros(K, device=labels_b0.device, dtype=torch.bool)
                    top_mask[top_clusters] = True

                    sel_mask = valid & top_mask[labels_safe] & non_block_mask
                    idx_global = sel_mask.nonzero(as_tuple=True)[0]

                    if idx_global.numel() > 0:
                        cos, sin = position_embeddings_for_cluster

                        res0 = hidden_state_past[:, idx_global, :]
                        hs_full = self.input_layernorm(hidden_state_past)
                        q_full = self.self_attn.q_proj(hs_full)
                        q_full = q_full.view(B, -1, n_h, hd).transpose(1, 2)

                        q_rot, _ = apply_rotary_pos_emb(q_full, key_present, cos, sin)
                        q_rot = q_rot[:, :, idx_global, :]
                        if q_rot.device.type == "cuda" and attention_mask is not None:
                            q_rot = q_rot.contiguous()
                            rope_k = rope_k.contiguous()
                            rope_v = rope_v.contiguous()

                        attn_out = torch.nn.functional.scaled_dot_product_attention(
                            q_rot,
                            rope_k,
                            rope_v,
                            attn_mask=None,
                            dropout_p=0.0,
                            is_causal=False,
                        )

                        attn_out = attn_out.transpose(1, 2).contiguous().view(B, -1, D)

                        attn_out = self.self_attn.o_proj(attn_out)
                        h_sel = res0 + attn_out

                        res1 = h_sel
                        h_sel = self.post_attention_layernorm(h_sel)
                        h_sel = self.mlp(h_sel)
                        h_sel = res1 + h_sel

                        update_packet_L = (idx_global, h_sel)

        if use_cache:
            if not use_cluster:
                hidden_state_past_new = None
            else:
                hidden_state_past_new = hidden_state_past if hidden_state_past is not None else current_hidden_state
        else:
            hidden_state_past_new = None
        clustered_hidden_state_new = clustered_hidden_state if use_cluster else None

        outputs = (hidden_states,)
        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        outputs += (hidden_state_past_new, clustered_hidden_state_new, update_packet_L, drift_score)

        return outputs

class DreamPreTrainedModel(PreTrainedModel):
    config_class = DreamConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["DreamDecoderLayer"]
    _skip_keys_device_placement = "past_key_values"
    _supports_flash_attn_2 = True
    _supports_sdpa = True
    _supports_cache_class = True
    _supports_quantized_cache = True
    _supports_static_cache = True

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: Optional[Union[str, os.PathLike]],
        *model_args,
        config: Optional[Union[PretrainedConfig, str, os.PathLike]] = None,
        cache_dir: Optional[Union[str, os.PathLike]] = None,
        ignore_mismatched_sizes: bool = False,
        force_download: bool = False,
        local_files_only: bool = False,
        token: Optional[Union[str, bool]] = None,
        revision: str = "main",
        use_safetensors: Optional[bool] = None,
        weights_only: bool = True,
        **kwargs,
    ):
        _model = super().from_pretrained(
            pretrained_model_name_or_path,
            *model_args,
            config=config,
            cache_dir=cache_dir,
            ignore_mismatched_sizes=ignore_mismatched_sizes,
            force_download=force_download,
            local_files_only=local_files_only,
            token=token,
            revision=revision,
            use_safetensors=use_safetensors,
            weights_only=weights_only,
            **kwargs,
        )

        resume_download = kwargs.get("resume_download", None)
        proxies = kwargs.get("proxies", None)
        subfolder = kwargs.get("subfolder", "")
        from_auto_class = kwargs.get("_from_auto", False)
        from_pipeline = kwargs.get("_from_pipeline", None)
        _model.generation_config = DreamGenerationConfig.from_pretrained(
            pretrained_model_name_or_path,
            cache_dir=cache_dir,
            force_download=force_download,
            resume_download=resume_download,
            proxies=proxies,
            local_files_only=local_files_only,
            token=token,
            revision=revision,
            subfolder=subfolder,
            _from_auto=from_auto_class,
            _from_pipeline=from_pipeline,
        )
        return _model

class DreamBaseModel(DreamPreTrainedModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`DreamDecoderLayer`]

    Args:
        config: DreamConfig
    """

    def __init__(self, config: DreamConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [DreamDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self._attn_implementation = config._attn_implementation
        self.norm = DreamRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = DreamRotaryEmbedding(config=config)

        self.gradient_checkpointing = False

        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        past_hidden_states: Optional[List[torch.Tensor]] = None,
        clustered_hidden_states: Optional[List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]] = None,
        cluster: bool = False,
        topk: Optional[int] = None,
        efficient: bool = False,
        efficient_window_position: Optional[torch.Tensor] = None,
        get_token_drift_score: bool = False,
        use_cluster: bool = True,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        dual_cache: Optional[bool] = False,
        replace_position: Optional[torch.Tensor] = None,
    ) -> Union[Tuple, BaseModelOutput]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else False

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        past_seen_tokens = past_key_values[0][0].shape[1] if past_key_values is not None else 0
        if not dual_cache:
            position_ids = torch.arange(past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device).unsqueeze(0)
        else:
            if past_key_values is not None:
                position_ids = torch.arange(past_seen_tokens, device=inputs_embeds.device).unsqueeze(0)
            else:
                position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device).unsqueeze(0)

        hidden_states = inputs_embeds
        attn_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = [] if use_cache else None
        updated_past_hidden_states = [] if use_cache else None
        updated_clustered_hidden_states = [] if use_cache else None
        updated_drift_scores = [] if use_cache else None
        update_packet = None

        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        position_embeddings_for_cluster = None
        if dual_cache and cluster:
            L = past_key_values[0][0].shape[1]
            position_ids_full = torch.arange(L, device=inputs_embeds.device).unsqueeze(0)
            dummy = inputs_embeds.new_empty((inputs_embeds.size(0), L, inputs_embeds.size(-1)))
            position_embeddings_for_cluster = self.rotary_emb(dummy, position_ids_full)

        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None

        for layer_idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            layer_past_key_value = past_key_values[layer_idx] if past_key_values is not None else None
            layer_hidden_state_past = past_hidden_states[layer_idx] if past_hidden_states is not None else None
            layer_clustered_state = clustered_hidden_states[layer_idx] if clustered_hidden_states is not None else None

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    attention_mask,
                    position_ids,
                    layer_past_key_value,
                    output_attentions,
                    use_cache,
                    cache_position,
                    position_embeddings,
                    dual_cache,
                    replace_position,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=layer_past_key_value,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                    dual_cache=dual_cache,
                    replace_position=replace_position,

                    layer_id=layer_idx,
                    hidden_state_past=layer_hidden_state_past,
                    clustered_hidden_state=layer_clustered_state,
                    update_packet_prev=update_packet,
                    cluster=cluster,
                    topk=topk,
                    efficient=efficient,
                    efficient_window_position=efficient_window_position,
                    position_embeddings_for_cluster=position_embeddings_for_cluster,
                    get_token_drift_score=get_token_drift_score,
                    use_cluster=use_cluster,
                )

            hidden_states = layer_outputs[0]
            cursor = 1
            if output_attentions:
                self_attn = layer_outputs[cursor]
                cursor += 1

            present_kv = None
            if use_cache:
                present_kv = layer_outputs[cursor]
                cursor += 1

            hidden_state_past_new = layer_outputs[cursor]; cursor += 1
            clustered_state_new = layer_outputs[cursor]; cursor += 1
            update_packet = layer_outputs[cursor]; cursor += 1
            drift_score = layer_outputs[cursor]; cursor += 1

            if use_cache:
                attn_key_values.append(present_kv)
                updated_past_hidden_states.append(hidden_state_past_new)
                updated_clustered_hidden_states.append(
                    clustered_state_new if clustered_state_new is not None else layer_clustered_state
                )
                updated_drift_scores.append(drift_score)
            if output_attentions:
                all_self_attns += (self_attn,)
        logits_sel = None
        sel_indices = None
        if update_packet is not None and cluster:
            sel_indices, h_sel = update_packet
        hidden_states = self.norm(hidden_states)

        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states, all_self_attns] if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
            past_key_values=attn_key_values,

            past_hidden_states=updated_past_hidden_states,
            clustered_hidden_states=updated_clustered_hidden_states,
            update_packet=update_packet,
            drift_scores=updated_drift_scores,
            logits_sel=logits_sel,
            sel_indices=sel_indices,
        )

class DreamModel(DreamGenerationMixin, DreamPreTrainedModel):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.model = DreamBaseModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.post_init()

    def reset_rope_parameters(self):
        self.model.rotary_emb.reset_parameters()
        for layer in self.model.layers:
            layer.self_attn.rotary_emb.reset_parameters()

    def reset_dual_cache_debug_state(self):
        for layer in self.model.layers:
            layer.attn_weights = None
            layer._prev_attn_scores = None

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        num_logits_to_keep: int = 0,
        dual_cache: Optional[bool] = False,
        replace_position: Optional[torch.Tensor] = None,
        **loss_kwargs,
    ) -> Union[Tuple, MaskedLMOutput]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            past_hidden_states=loss_kwargs.get("past_hidden_states", None),
            clustered_hidden_states=loss_kwargs.get("clustered_hidden_states", None),
            cluster=loss_kwargs.get("cluster", False),
            topk=loss_kwargs.get("topk", None),
            efficient=loss_kwargs.get("efficient", False),
            efficient_window_position=loss_kwargs.get("efficient_window_position", None),
            get_token_drift_score=loss_kwargs.get("get_token_drift_score", False),
            use_cluster=loss_kwargs.get("use_cluster", True),
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
            dual_cache=dual_cache,
            replace_position=replace_position,
        )
        hidden_states = outputs[0]

        logits = self.lm_head(hidden_states[:, -num_logits_to_keep:, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(logits, labels, self.vocab_size, **loss_kwargs)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        logits_sel = None
        sel_indices = getattr(outputs, "sel_indices", None)
        if sel_indices is None:
            up = getattr(outputs, "update_packet", None)
            if up is not None:
                sel_indices, h_sel = up
        else:
            up = getattr(outputs, "update_packet", None)
            if up is not None:
                _, h_sel = up

        if sel_indices is not None and up is not None:
            h_sel_norm = self.model.norm(h_sel)
            logits_sel = self.lm_head(h_sel_norm)
        return MaskedLMOutputWithPastKeyValues(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            past_key_values=outputs.past_key_values,

            past_hidden_states=getattr(outputs, "past_hidden_states", None),
            clustered_hidden_states=getattr(outputs, "clustered_hidden_states", None),
            update_packet=getattr(outputs, "update_packet", None),
            drift_scores=getattr(outputs, "drift_scores", None),
            logits_sel=logits_sel,
            sel_indices=sel_indices,
        )
