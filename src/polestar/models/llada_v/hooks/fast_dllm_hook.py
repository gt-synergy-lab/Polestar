"""Polestar generation and attention hooks for LLaDA-V."""

import math
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from .clustering_utils import k_means


class PolestarGenerationHook:
    def __init__(self, model):
        self.model = model
        self.original_methods = {}
        self.is_registered = False

    def register_hooks(self):
        if self.is_registered:
            return

        self.original_methods["generate"] = self.model.generate

        for layer_idx, layer in enumerate(self.model.model.layers):
            self.original_methods[f"attention_{layer_idx}"] = layer.self_attn.forward
            layer.self_attn.forward = self._create_fast_attention_forward(layer.self_attn, layer_idx)

        self.model.generate = self._fast_generate
        self.is_registered = True

    def unregister_hooks(self):
        if not self.is_registered:
            return

        self.model.generate = self.original_methods["generate"]
        for layer_idx, layer in enumerate(self.model.model.layers):
            layer.self_attn.forward = self.original_methods[f"attention_{layer_idx}"]
        self.is_registered = False

    def _create_fast_attention_forward(self, attention_layer, layer_idx):
        def fast_attention_forward(
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_value: Optional[Tuple[torch.Tensor]] = None,
            output_attentions: bool = False,
            use_cache: bool = False,
            cache_position: Optional[torch.LongTensor] = None,
            fast_dllm_cache: Optional[Sequence[Tuple[torch.Tensor, torch.Tensor]]] = None,
            **kwargs,
        ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
            if output_attentions:
                return self.original_methods[f"attention_{layer_idx}"](
                    hidden_states=hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    **kwargs,
                )

            bsz, q_len, _ = hidden_states.size()

            query_states = attention_layer.q_proj(hidden_states)
            key_states = attention_layer.k_proj(hidden_states)
            value_states = attention_layer.v_proj(hidden_states)

            query_states = query_states.view(
                bsz, q_len, attention_layer.num_heads, attention_layer.head_dim
            ).transpose(1, 2)
            key_states = key_states.view(
                bsz, q_len, attention_layer.num_key_value_heads, attention_layer.head_dim
            ).transpose(1, 2)
            value_states = value_states.view(
                bsz, q_len, attention_layer.num_key_value_heads, attention_layer.head_dim
            ).transpose(1, 2)

            if position_ids is None:
                cache_offset = 0
                if fast_dllm_cache and len(fast_dllm_cache) > layer_idx:
                    cache_offset = fast_dllm_cache[layer_idx][0].shape[-2]
                position_ids = torch.arange(
                    cache_offset, cache_offset + q_len, device=hidden_states.device
                ).unsqueeze(0).expand(bsz, -1)

            cos, sin = attention_layer.rotary_emb(value_states, position_ids)
            query_states, key_states = self._apply_rotary_pos_emb(query_states, key_states, cos, sin)

            past_key_value = getattr(attention_layer, "past_key_value", past_key_value)
            if past_key_value is not None:
                cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
                key_states, value_states = past_key_value.update(
                    key_states, value_states, layer_idx, cache_kwargs
                )

            if fast_dllm_cache is not None:
                if len(fast_dllm_cache) <= layer_idx:
                    fast_dllm_cache.append((key_states, value_states))
                else:
                    past_key, past_value = fast_dllm_cache[layer_idx]
                    key_states = torch.cat([past_key, key_states], dim=-2)
                    value_states = torch.cat([past_value, value_states], dim=-2)

            key_states = self._repeat_kv(key_states, attention_layer.num_key_value_groups)
            value_states = self._repeat_kv(value_states, attention_layer.num_key_value_groups)

            if attention_mask is not None:
                target_k_len = key_states.shape[-2]
                current_k_len = attention_mask.shape[-1]
                if current_k_len < target_k_len:
                    pad_len = target_k_len - current_k_len
                    if attention_mask.dtype == torch.bool:
                        # Bool SDPA masks use True for allowed positions.
                        prefix_pad = torch.ones(
                            attention_mask.shape[0],
                            attention_mask.shape[1],
                            attention_mask.shape[2],
                            pad_len,
                            device=attention_mask.device,
                            dtype=attention_mask.dtype,
                        )
                    else:
                        # Float SDPA masks are additive; zeros mean "no extra masking".
                        prefix_pad = torch.zeros(
                            attention_mask.shape[0],
                            attention_mask.shape[1],
                            attention_mask.shape[2],
                            pad_len,
                            device=attention_mask.device,
                            dtype=attention_mask.dtype,
                        )
                    attention_mask = torch.cat([prefix_pad, attention_mask], dim=-1)
                elif current_k_len > target_k_len:
                    attention_mask = attention_mask[:, :, :, :target_k_len]

            if query_states.device.type == "cuda" and attention_mask is not None:
                query_states = query_states.contiguous()
                key_states = key_states.contiguous()
                value_states = value_states.contiguous()

            attn_output = torch.nn.functional.scaled_dot_product_attention(
                query_states,
                key_states,
                value_states,
                attn_mask=attention_mask,
                is_causal=False,
                dropout_p=attention_layer.attention_dropout if attention_layer.training else 0.0,
            )

            attn_output = attn_output.transpose(1, 2).contiguous()
            attn_output = attn_output.view(bsz, q_len, attention_layer.hidden_size)
            attn_output = attention_layer.o_proj(attn_output)

            return attn_output, None, past_key_value

        return fast_attention_forward

    @torch.no_grad()
    def _fast_generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        image_sizes: Optional[torch.Tensor] = None,
        modalities: Optional[List[str]] = ["image"],
        **kwargs,
    ):
        modalities = kwargs.pop("modalities", None) if "modalities" in kwargs and modalities is None else modalities
        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)

        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported")

        if images is not None:
            (
                inputs,
                position_ids,
                attention_mask,
                _,
                inputs_embeds,
                _,
            ) = self.model.prepare_inputs_labels_for_multimodal(
                inputs,
                position_ids,
                attention_mask,
                None,
                None,
                images,
                modalities,
                image_sizes=image_sizes,
            )
        else:
            inputs_embeds = self.model.get_model().embed_tokens(inputs)

        return self._fast_generate_with_embeds(inputs_embeds=inputs_embeds, **kwargs)

    @torch.no_grad()
    def _fast_generate_with_embeds(
        self,
        inputs_embeds,
        steps=128,
        gen_length=128,
        block_length=128,
        temperature=0.0,
        cfg_scale=0.0,
        remasking="low_confidence",
        mask_id=126336,
        tokenizer=None,
        stopping_criteria=None,
        generation_suffix=None,
        threshold=None,
        early_stop=False,
        prefix_refresh_interval=32,
        threshold_early=None,
        use_cluster=True,
        cluster_num_centroids=8,
        cluster_topk=4,
        efficient_window_stride=0,
        use_suffix_preunmask=False,
        **kwargs,
    ):
        with torch.cuda.amp.autocast(enabled=True):
            nfe = 0
            block_stats = []
            prompt_len = inputs_embeds.shape[1]
            suffix_embeds = None
            suffix_token_ids = None
            suffix_len = 0
            if generation_suffix is not None and tokenizer is not None and len(generation_suffix) > 0:
                suffix_token_ids = tokenizer.encode(generation_suffix, add_special_tokens=False)
                suffix_token_ids = torch.tensor(
                    suffix_token_ids, dtype=torch.long, device=inputs_embeds.device
                ).unsqueeze(0)
                suffix_embeds = self.model.model.embed_tokens(suffix_token_ids)
                suffix_len = suffix_embeds.shape[1]

            total_length = prompt_len + gen_length + suffix_len
            sequence_end = prompt_len + gen_length
            masked_embed = self.model.model.embed_tokens(
                torch.tensor([mask_id], device=inputs_embeds.device)
            )
            x_embeds = masked_embed.repeat(1, total_length, 1).to(inputs_embeds.device)
            x_embeds[:, :prompt_len] = inputs_embeds.clone()
            if suffix_embeds is not None:
                x_embeds[:, -suffix_len:] = suffix_embeds

            x = torch.full((1, total_length), mask_id, dtype=torch.long, device=inputs_embeds.device)
            if suffix_token_ids is not None:
                x[:, -suffix_len:] = suffix_token_ids

            prompt_index = torch.zeros((1, total_length), dtype=torch.bool, device=inputs_embeds.device)
            prompt_index[:, :prompt_len] = 1

            assert gen_length % block_length == 0
            num_blocks = gen_length // block_length
            assert steps % num_blocks == 0
            steps = steps // num_blocks

            stop_position = sequence_end
            found_stop_seq = False
            stop_tokens = []
            if early_stop and stopping_criteria is not None:
                assert tokenizer is not None, "tokenizer is required when stopping_criteria is not None"
                for stop_str in stopping_criteria:
                    stop_tokens.append(tokenizer.encode(stop_str, add_special_tokens=False))

            cached_final_hidden = None
            fast_dllm_cache = []

            for num_block in range(num_blocks):
                block_nfe = 0
                block_start = prompt_len + num_block * block_length
                block_end = block_start + block_length
                drift_window = []
                dec_tok_number_until_update = 0
                cluster_state = None

                if early_stop and found_stop_seq and stop_position <= block_start:
                    break

                block_mask_index = self._get_masked_indices_from_embeds(
                    x_embeds[:, block_start:block_end], masked_embed
                )
                num_transfer_tokens = self._get_num_transfer_tokens(block_mask_index, steps)

                efficient = False
                if efficient:
                    refresh_start = max(0, block_start - 2 * block_length)
                    refresh_end = min(total_length, block_end + block_length)
                    cache_prefix_len = block_start - refresh_start
                else:
                    refresh_start = 0
                    refresh_end = total_length
                    cache_prefix_len = block_start

                prev_cache = fast_dllm_cache
                logits_full, hidden_full, fast_dllm_cache = self._run_model_pass(
                    x_embeds,
                    prompt_index,
                    masked_embed,
                    refresh_start,
                    refresh_end,
                    cfg_scale=cfg_scale,
                    output_hidden_states=use_cluster,
                    fresh_cache=True,
                    slice_prefix_len=cache_prefix_len,
                )
                nfe += 1
                block_nfe += 1
                if refresh_start > 0 and prev_cache:
                    fast_dllm_cache = self._merge_prefix_caches(
                        prev_cache,
                        fast_dllm_cache,
                        refresh_start,
                    )
                self._mask_forbidden_tokens(
                    logits_full,
                    self._get_masked_indices_from_embeds(x_embeds[:, refresh_start:refresh_end], masked_embed),
                )

                if use_cluster:
                    cached_final_hidden = self._update_hidden_cache(
                        cached_final_hidden, hidden_full, refresh_start, refresh_end, total_length
                    )
                    cluster_state = self._build_cluster_state(
                        cached_final_hidden,
                        prompt_len,
                        block_start,
                        block_end,
                        sequence_end,
                        block_length,
                        cluster_num_centroids,
                    )

                local_block_start = block_start - refresh_start
                local_block_end = block_end - refresh_start
                block_logits = logits_full[:, local_block_start:local_block_end, :]
                x0_block, transfer_index, _ = self._get_transfer_index(
                    block_logits,
                    temperature,
                    remasking,
                    self._get_masked_indices_from_embeds(x_embeds[:, block_start:block_end], masked_embed),
                    x[:, block_start:block_end],
                    num_transfer_tokens[:, 0] if threshold is None else None,
                    threshold,
                )
                x_block_embeds = self.model.model.embed_tokens(x0_block)
                x_embeds[:, block_start:block_end][transfer_index] = x_block_embeds[transfer_index]
                x[:, block_start:block_end][transfer_index] = x0_block[transfer_index]
                if early_stop:
                    found_stop_seq, stop_position = self._update_stop_position(
                        x,
                        prompt_len,
                        gen_length,
                        stop_tokens,
                        found_stop_seq,
                        stop_position,
                    )

                i = 1
                while True:
                    mask_index = self._get_masked_indices_from_embeds(x_embeds, masked_embed)
                    if early_stop and found_stop_seq:
                        pre_stop_masks = mask_index[0, prompt_len:stop_position]
                        if not pre_stop_masks.any():
                            break

                    if not mask_index[0, block_start:block_end].any():
                        break
                    if threshold is None and i >= steps:
                        break

                    prev_block_hidden = None
                    if use_cluster and cached_final_hidden is not None:
                        prev_block_hidden = cached_final_hidden[:, block_start:block_end, :].clone()

                    do_cluster_refresh = False
                    refresh_start = block_start
                    refresh_end = total_length
                    selected_positions = None
                    if use_cluster and cluster_state is not None:
                        cluster_trigger = transfer_index.sum().item() > 1 or dec_tok_number_until_update > 2
                        if cluster_trigger:
                            selected_positions = self._select_refresh_positions(
                                cluster_state,
                                prev_block_hidden,
                                topk=cluster_topk,
                            )
                            if selected_positions is not None and selected_positions.numel() > 0:
                                refresh_start = min(block_start, int(selected_positions.min().item()))
                                do_cluster_refresh = True
                                dec_tok_number_until_update = 0

                    if i % prefix_refresh_interval == 0 and not do_cluster_refresh:
                        logits_step, hidden_step, fast_dllm_cache = self._run_model_pass(
                            x_embeds,
                            prompt_index,
                            masked_embed,
                            0,
                            total_length,
                            cfg_scale=cfg_scale,
                            output_hidden_states=use_cluster,
                            fresh_cache=True,
                            slice_prefix_len=block_start,
                        )
                        nfe += 1
                        block_nfe += 1
                        segment_start = 0
                    elif do_cluster_refresh:
                        prev_cache = fast_dllm_cache
                        logits_step, hidden_step, fast_dllm_cache = self._run_model_pass(
                            x_embeds,
                            prompt_index,
                            masked_embed,
                            refresh_start,
                            total_length,
                            cfg_scale=cfg_scale,
                            output_hidden_states=use_cluster,
                            fresh_cache=True,
                            slice_prefix_len=block_start - refresh_start,
                        )
                        nfe += 1
                        block_nfe += 1
                        if refresh_start > 0 and prev_cache:
                            fast_dllm_cache = self._merge_prefix_caches(
                                prev_cache,
                                fast_dllm_cache,
                                refresh_start,
                            )
                        segment_start = refresh_start
                    else:
                        logits_step, hidden_step, _ = self._run_model_pass(
                            x_embeds,
                            prompt_index,
                            masked_embed,
                            block_start,
                            total_length,
                            cfg_scale=cfg_scale,
                            output_hidden_states=use_cluster,
                            fresh_cache=False,
                            existing_cache=fast_dllm_cache,
                        )
                        nfe += 1
                        block_nfe += 1
                        segment_start = block_start

                    step_mask_index = self._get_masked_indices_from_embeds(
                        x_embeds[:, segment_start:total_length], masked_embed
                    )
                    self._mask_forbidden_tokens(logits_step, step_mask_index)

                    if use_cluster and hidden_step is not None:
                        cached_final_hidden = self._update_hidden_cache(
                            cached_final_hidden, hidden_step, segment_start, total_length, total_length
                        )
                        current_block_hidden = hidden_step[
                            :, block_start - segment_start : block_end - segment_start, :
                        ]
                        d_block = self._compute_block_drift(current_block_hidden, prev_block_hidden)
                        if d_block is not None:
                            if len(drift_window) > 0:
                                prev_stack = torch.stack(drift_window, dim=0)
                                delta = d_block - prev_stack.mean(dim=0)
                            else:
                                delta = d_block
                            drift_window.append(d_block.detach())
                            if len(drift_window) > 5:
                                drift_window.pop(0)
                        else:
                            delta = None

                        cluster_state = self._build_cluster_state(
                            cached_final_hidden,
                            prompt_len,
                            block_start,
                            block_end,
                            sequence_end,
                            block_length,
                            cluster_num_centroids,
                        )
                    else:
                        delta = None

                    if use_cluster and use_suffix_preunmask and selected_positions is not None and selected_positions.numel() > 0:
                        self._maybe_preunmask_suffix(
                            x=x,
                            x_embeds=x_embeds,
                            logits=logits_step,
                            segment_start=segment_start,
                            selected_positions=selected_positions,
                            current_block_end=block_end,
                            sequence_end=sequence_end,
                            masked_embed=masked_embed,
                            temperature=temperature,
                            remasking=remasking,
                            threshold=threshold,
                        )

                    block_logits = logits_step[
                        :, block_start - segment_start : block_end - segment_start, :
                    ]
                    block_mask_index = self._get_masked_indices_from_embeds(
                        x_embeds[:, block_start:block_end], masked_embed
                    )
                    x0, transfer_index, conf = self._get_transfer_index(
                        block_logits,
                        temperature,
                        remasking,
                        block_mask_index,
                        x[:, block_start:block_end],
                        num_transfer_tokens[:, i] if threshold is None else None,
                        threshold,
                    )

                    if threshold_early is not None and threshold is not None and delta is not None:
                        conf_block = conf
                        diff = torch.clamp(0.9 - conf_block, min=0.0)
                        dynamic_gate = 10 * (diff**2)
                        early_conf_mask = (conf_block > threshold_early) & (conf_block < threshold)
                        drift_mask = (delta.unsqueeze(0) >= dynamic_gate) & (delta.unsqueeze(0) < 2.0)
                        transfer_index = transfer_index | (early_conf_mask & drift_mask & block_mask_index)

                    x0_embeds = self.model.model.embed_tokens(x0)
                    x_embeds[:, block_start:block_end][transfer_index] = x0_embeds[transfer_index]
                    x[:, block_start:block_end][transfer_index] = x0[transfer_index]
                    if early_stop:
                        found_stop_seq, stop_position = self._update_stop_position(
                            x,
                            prompt_len,
                            gen_length,
                            stop_tokens,
                            found_stop_seq,
                            stop_position,
                        )
                    dec_tok_number_until_update += transfer_index.sum().item()
                    i += 1

                block_stats.append(
                    {
                        "block_id": int(num_block),
                        "block_start": int(block_start),
                        "block_end": int(block_end),
                        "block_nfe": int(block_nfe),
                    }
                )

            self.model._fast_dllm_last_stats = {
                "nfe": int(nfe),
                "block_stats": block_stats,
            }

            if early_stop and found_stop_seq:
                if suffix_len > 0:
                    return torch.cat(
                        [x[:, prompt_len:stop_position], x[:, -suffix_len:]], dim=1
                    )
                return x[:, prompt_len:stop_position]

            if suffix_len > 0:
                return torch.cat(
                    [
                        x[:, prompt_len:prompt_len + gen_length],
                        x[:, -suffix_len:],
                    ],
                    dim=1,
                )
            return x[:, prompt_len:prompt_len + gen_length]

    def _run_model_pass(
        self,
        x_embeds,
        prompt_index,
        masked_embed,
        start_idx,
        end_idx,
        cfg_scale=0.0,
        output_hidden_states=False,
        fresh_cache=False,
        slice_prefix_len=None,
        existing_cache=None,
    ):
        slice_embeds = x_embeds[:, start_idx:end_idx]
        position_ids = torch.arange(start_idx, end_idx, device=x_embeds.device).unsqueeze(0)

        cache_ref = [] if fresh_cache else existing_cache

        if cfg_scale > 0.0:
            un_embeds = x_embeds.clone()
            un_mask = prompt_index.unsqueeze(-1).expand_as(x_embeds)
            un_embeds[un_mask] = masked_embed.repeat(x_embeds.shape[0], x_embeds.shape[1], 1)[un_mask]
            combined_embeds = torch.cat([slice_embeds, un_embeds[:, start_idx:end_idx]], dim=0)
            outputs = self.model.model(
                inputs_embeds=combined_embeds,
                position_ids=position_ids.repeat(combined_embeds.shape[0], 1),
                use_cache=False,
                output_hidden_states=output_hidden_states,
            )
            logits = self.model.lm_head(outputs[0]).float()
            logits, un_logits = torch.chunk(logits, 2, dim=0)
            logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
            hidden = outputs.hidden_states[-1][:1].detach() if output_hidden_states else None
            return logits, hidden, cache_ref

        outputs = self.model.model(
            inputs_embeds=slice_embeds,
            position_ids=position_ids,
            use_cache=False,
            output_hidden_states=output_hidden_states,
            fast_dllm_cache=cache_ref,
        )
        logits = self.model.lm_head(outputs[0]).float()
        hidden = outputs.hidden_states[-1].detach() if output_hidden_states else None

        if fresh_cache and slice_prefix_len is not None:
            cache_ref = self._create_cache_slice(cache_ref, slice_prefix_len)

        return logits, hidden, cache_ref

    def _mask_forbidden_tokens(self, logits, mask_index):
        forbidden_tokens = [126081, 126080, 126346, 126347]
        for token_id in forbidden_tokens:
            logits[:, :, token_id] = torch.where(mask_index, -float("inf"), logits[:, :, token_id])

    def _update_hidden_cache(self, cached_hidden, hidden_slice, start_idx, end_idx, total_length):
        if hidden_slice is None:
            return cached_hidden
        if cached_hidden is None:
            cached_hidden = torch.zeros(
                hidden_slice.shape[0],
                total_length,
                hidden_slice.shape[-1],
                device=hidden_slice.device,
                dtype=hidden_slice.dtype,
            )
        cached_hidden[:, start_idx:end_idx, :] = hidden_slice
        return cached_hidden

    def _build_cluster_state(
        self,
        cached_final_hidden,
        prompt_len,
        block_start,
        block_end,
        sequence_end,
        block_length,
        num_clusters,
    ):
        if cached_final_hidden is None:
            return None

        prefix_end = prompt_len
        local_start = max(prefix_end, block_start - 2 * block_length)
        w_end = min(sequence_end, block_end + block_length)

        chunks = []
        positions = []

        # Always include the original multimodal prefix as a global anchor.
        if prefix_end > 0:
            chunks.append(cached_final_hidden[:, :prefix_end, :])
            positions.append(torch.arange(0, prefix_end, device=cached_final_hidden.device))

        # Only include recent generated history before the current block.
        if local_start < block_start:
            chunks.append(cached_final_hidden[:, local_start:block_start, :])
            positions.append(torch.arange(local_start, block_start, device=cached_final_hidden.device))

        # Keep a small future/post window for nearby context after the current block.
        if block_end < w_end:
            chunks.append(cached_final_hidden[:, block_end:w_end, :])
            positions.append(torch.arange(block_end, w_end, device=cached_final_hidden.device))

        if not chunks:
            return None

        hidden_window = torch.cat(chunks, dim=1)
        position_window = torch.cat(positions, dim=0)
        if hidden_window.shape[1] == 0:
            return None

        k = min(num_clusters, hidden_window.shape[1])
        centroids, labels = k_means(hidden_window.unsqueeze(1), k, distance="cosine")
        counts = F.one_hot(labels, k).sum(dim=2)
        return {
            "centroids": centroids.squeeze(1),
            "labels": labels.squeeze(1),
            "counts": counts.squeeze(1),
            "positions": position_window,
            "prev_sim": None,
        }

    def _select_refresh_positions(self, cluster_state, block_hidden, topk=2):
        if cluster_state is None or block_hidden is None:
            return None

        q = F.normalize(block_hidden.mean(dim=1), p=2, dim=-1)
        centroids = F.normalize(cluster_state["centroids"], p=2, dim=-1)
        sim = torch.matmul(q.unsqueeze(1), centroids.transpose(-2, -1)).squeeze(1)

        prev_sim = cluster_state.get("prev_sim")
        if prev_sim is None:
            drift = 1.0 - sim
        else:
            drift = (sim - prev_sim).abs()
        cluster_state["prev_sim"] = sim.detach()

        topk = min(topk, drift.shape[-1])
        if topk == 0:
            return None

        top_clusters = torch.topk(drift[0], k=topk).indices
        selected_mask = torch.zeros_like(cluster_state["labels"][0], dtype=torch.bool)
        for cluster_idx in top_clusters:
            selected_mask |= cluster_state["labels"][0] == cluster_idx
        if not selected_mask.any():
            return None
        return cluster_state["positions"][selected_mask]

    def _compute_block_drift(self, current_hidden, previous_hidden):
        if current_hidden is None or previous_hidden is None or current_hidden.shape != previous_hidden.shape:
            return None
        cur = F.normalize(current_hidden.float(), p=2, dim=-1)
        prev = F.normalize(previous_hidden.float(), p=2, dim=-1)
        return 1.0 - (cur * prev).sum(dim=-1).squeeze(0)

    def _maybe_preunmask_suffix(
        self,
        x,
        x_embeds,
        logits,
        segment_start,
        selected_positions,
        current_block_end,
        sequence_end,
        masked_embed,
        temperature,
        remasking,
        threshold,
    ):
        future_positions = selected_positions[
            (selected_positions >= current_block_end) & (selected_positions < sequence_end)
        ]
        if future_positions.numel() == 0:
            return

        local_positions = future_positions - segment_start
        valid = (local_positions >= 0) & (local_positions < logits.shape[1])
        if not valid.any():
            return

        future_positions = future_positions[valid]
        local_positions = local_positions[valid]
        suffix_logits = logits[:, local_positions, :]
        suffix_mask = self._get_masked_indices_from_embeds(x_embeds[:, future_positions], masked_embed)
        if not suffix_mask.any():
            return

        num_transfer = torch.ones((1, 1), dtype=torch.int64, device=x.device)
        x_new, transfer_index = self._get_pre_transfer_index(
            suffix_logits,
            temperature,
            remasking,
            suffix_mask,
            x[:, future_positions],
            num_transfer,
            threshold,
        )
        chosen = transfer_index[0].nonzero(as_tuple=True)[0]
        if chosen.numel() == 0:
            return

        target_positions = future_positions[chosen]
        x[0, target_positions] = x_new[0, chosen]
        x_embeds[0, target_positions] = self.model.model.embed_tokens(x_new[0, chosen])

    @staticmethod
    def _update_stop_position(x, prompt_len, gen_length, stop_tokens, found_stop_seq, stop_position):
        if not stop_tokens:
            return found_stop_seq, stop_position

        generated_part = x[0, prompt_len : prompt_len + gen_length]
        for stop_seq in stop_tokens:
            if not isinstance(stop_seq, list):
                stop_seq = [stop_seq]
            stop_seq_tensor = torch.tensor(stop_seq, device=x.device)
            max_start = generated_part.size(0) - len(stop_seq) + 1
            for start_idx in range(max(0, max_start)):
                if torch.all(generated_part[start_idx : start_idx + len(stop_seq)] == stop_seq_tensor):
                    current_position = prompt_len + start_idx
                    if (not found_stop_seq) or current_position < stop_position:
                        stop_position = current_position
                        found_stop_seq = True
                    break
            if found_stop_seq:
                break
        return found_stop_seq, stop_position

    @staticmethod
    def _get_masked_indices_from_embeds(noisy_embeds, masked_embed):
        b, l, d = noisy_embeds.shape
        masked_embed_expanded = masked_embed.expand(b, l, d)
        abs_diff = torch.abs(noisy_embeds - masked_embed_expanded)
        tolerance = 1e-5 + 1e-5 * torch.abs(masked_embed_expanded)
        return (abs_diff <= tolerance).all(dim=-1)

    @staticmethod
    def _apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
        cos = cos.unsqueeze(unsqueeze_dim)
        sin = sin.unsqueeze(unsqueeze_dim)
        q_embed = (q * cos) + (PolestarGenerationHook._rotate_half(q) * sin)
        k_embed = (k * cos) + (PolestarGenerationHook._rotate_half(k) * sin)
        return q_embed, k_embed

    @staticmethod
    def _rotate_half(x):
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    @staticmethod
    def _repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
        batch, num_key_value_heads, slen, head_dim = hidden_states.shape
        if n_rep == 1:
            return hidden_states
        hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
        return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)

    @staticmethod
    def _add_gumbel_noise(logits, temperature):
        if temperature == 0:
            return logits
        logits = logits.to(torch.float64)
        noise = torch.rand_like(logits, dtype=torch.float64)
        gumbel_noise = (-torch.log(noise)) ** temperature
        return logits.exp() / gumbel_noise

    @staticmethod
    def _get_num_transfer_tokens(mask_index, steps):
        mask_num = mask_index.sum(dim=1, keepdim=True)
        base = mask_num // steps
        remainder = mask_num % steps
        num_transfer_tokens = base.expand(-1, steps).clone()
        if remainder.sum() > 0:
            indices = torch.arange(steps, device=mask_index.device)
            remainder_mask = indices.unsqueeze(0) < remainder
            num_transfer_tokens[remainder_mask] += 1
        return num_transfer_tokens.to(torch.int64)

    def _get_transfer_index(
        self, logits, temperature, remasking, mask_index, x, num_transfer_tokens, threshold=None
    ):
        logits_with_noise = self._add_gumbel_noise(logits, temperature=temperature)
        x0 = torch.argmax(logits_with_noise, dim=-1)

        if remasking == "low_confidence":
            p = F.softmax(logits.to(torch.float32), dim=-1)
            x0_p = torch.squeeze(torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1)
        elif remasking == "random":
            x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
        else:
            raise NotImplementedError(remasking)

        x0 = torch.where(mask_index, x0, x)
        confidence = torch.where(mask_index, x0_p, -np.inf)

        transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
        if threshold is not None:
            num_transfer_tokens = mask_index.sum(dim=1, keepdim=True)

        for j in range(confidence.shape[0]):
            _, select_index = torch.topk(confidence[j], k=num_transfer_tokens[j])
            transfer_index[j, select_index] = True
            if threshold is not None:
                for k in range(1, num_transfer_tokens[j]):
                    if confidence[j, select_index[k]] < threshold:
                        transfer_index[j, select_index[k]] = False

        return x0, transfer_index, confidence

    def _get_pre_transfer_index(
        self, logits, temperature, remasking, mask_index, x, num_transfer_tokens, threshold=None
    ):
        logits_with_noise = self._add_gumbel_noise(logits, temperature=temperature)
        x0 = torch.argmax(logits_with_noise, dim=-1)

        if remasking == "low_confidence":
            p = F.softmax(logits.to(torch.float32), dim=-1)
            x0_p = torch.squeeze(torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1)
        elif remasking == "random":
            x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
        else:
            raise NotImplementedError(remasking)

        x0 = torch.where(mask_index, x0, x)
        confidence = torch.where(mask_index, x0_p, -np.inf)

        transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
        if threshold is not None:
            num_transfer_tokens = mask_index.sum(dim=1, keepdim=True)

        for j in range(confidence.shape[0]):
            _, select_index = torch.topk(confidence[j], k=num_transfer_tokens[j])
            transfer_index[j, select_index] = True
            if threshold is not None:
                for k in range(0, num_transfer_tokens[j]):
                    if confidence[j, select_index[k]] < threshold:
                        transfer_index[j, select_index[k]] = False

        return x0, transfer_index

    @staticmethod
    def _create_cache_slice(fast_dllm_cache, prefix_len):
        new_past_key_values = []
        for layer_cache in fast_dllm_cache:
            new_past_key_values.append([])
            for kv in layer_cache:
                new_past_key_values[-1].append(kv[:, :, :prefix_len])
        return new_past_key_values

    @staticmethod
    def _merge_prefix_caches(existing_cache, new_cache, existing_prefix_len):
        if not existing_cache:
            return new_cache

        merged_cache = []
        for old_layer, new_layer in zip(existing_cache, new_cache):
            merged_layer = []
            for old_kv, new_kv in zip(old_layer, new_layer):
                merged_layer.append(
                    torch.cat([old_kv[:, :, :existing_prefix_len], new_kv], dim=-2)
                )
            merged_cache.append(merged_layer)
        return merged_cache


def register_polestar_hook(model):
    hook = PolestarGenerationHook(model)
    hook.register_hooks()
    return hook


def unregister_polestar_hook(hook):
    hook.unregister_hooks()
