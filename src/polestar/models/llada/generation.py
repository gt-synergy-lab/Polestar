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
# Modified from LLaDA repos: https://github.com/ML-GSAI/LLaDA

import torch
import numpy as np
import torch.nn.functional as F

def add_gumbel_noise(logits, temperature):
    '''
    The Gumbel max is a method for sampling categorical distributions.
    According to arXiv:2409.02908, for MDM, low-precision Gumbel Max improves perplexity score but reduces generation quality.
    Thus, we use float64.
    '''
    if temperature == 0:
        return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (- torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise

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

@ torch.no_grad()
def generate(model, prompt, steps=128, gen_length=128, block_length=128, temperature=0.,
             remasking='low_confidence', mask_id=126336, threshold=None, factor=None):
    '''
    Args:
        model: Mask predictor.
        prompt: A tensor of shape (1, L).
        steps: Sampling steps, less than or equal to gen_length.
        gen_length: Generated answer length.
        block_length: Block length, less than or equal to gen_length. If less than gen_length, it means using semi_autoregressive remasking.
        temperature: Categorical distribution sampling temperature.
        remasking: Remasking strategy. 'low_confidence' or 'random'.
        mask_id: The token id of [MASK] is 126336.
    '''
    x = torch.full((prompt.shape[0], prompt.shape[1] + gen_length), mask_id, dtype=torch.long).to(model.device)
    x[:, :prompt.shape[1]] = prompt.clone()

    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length

    assert steps % num_blocks == 0
    steps = steps // num_blocks

    nfe = 0
    block_stats = []
    seq_len = prompt.shape[1]
    for num_block in range(num_blocks):
        block_nfe = 0
        block_cluster_true = 0
        block_efficient_true = 0
        block_block0_true = 0
        current_block_start = prompt.shape[1] + num_block * block_length
        current_block_end = current_block_start + block_length
        block_mask_index = (x[:, prompt.shape[1] + num_block * block_length: prompt.shape[1] + (num_block + 1) * block_length] == mask_id)
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps)
        i = 0
        while True:
            block_block0_true += 1
            nfe += 1
            block_nfe += 1
            mask_index = (x == mask_id)
            logits = model(x).logits
            mask_index[:, prompt.shape[1] + (num_block + 1) * block_length:] = 0
            if factor is None:
                x0, transfer_index, _ = get_transfer_index(logits, temperature, remasking, mask_index, x, num_transfer_tokens[:, i] if threshold is None else None, threshold)
            else:
                x0, transfer_index = get_transfer_index_dynamic(logits, temperature, remasking, mask_index, x, None, factor)
            x[transfer_index] = x0[transfer_index]
            i += 1
            if (x[:, prompt.shape[1] + num_block * block_length: prompt.shape[1] + (num_block + 1) * block_length] == mask_id).sum() == 0:
                break
        block_stats.append({
                "block_id": num_block,
                "block_start": int(current_block_start),
                "block_end": int(current_block_end),
                "block_nfe": int(block_nfe),
                "block0_true": int(block_block0_true),
                "cluster_true": int(block_cluster_true),
                "efficient_true": int(block_efficient_true)
            })
    return x, nfe, block_stats, seq_len

@ torch.no_grad()
def generate_with_prefix_cache(model, prompt, steps=128, gen_length=128, block_length=128, temperature=0.,
             remasking='low_confidence', mask_id=126336, threshold=None, factor=None):
    '''
    Args:
        model: Mask predictor.
        prompt: A tensor of shape (1, L).
        steps: Sampling steps, less than or equal to gen_length.
        gen_length: Generated answer length.
        block_length: Block length, less than or equal to gen_length. If less than gen_length, it means using semi_autoregressive remasking.
        temperature: Categorical distribution sampling temperature.
        remasking: Remasking strategy. 'low_confidence' or 'random'.
        mask_id: The token id of [MASK] is 126336.
    '''
    x = torch.full((prompt.shape[0], prompt.shape[1] + gen_length), mask_id, dtype=torch.long).to(model.device)
    x[:, :prompt.shape[1]] = prompt.clone()

    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length

    assert steps % num_blocks == 0
    steps = steps // num_blocks

    nfe = 0

    for num_block in range(num_blocks):
        current_block_start = prompt.shape[1] + num_block * block_length
        current_block_end = current_block_start + block_length

        block_mask_index = (x[:, current_block_start:current_block_end] == mask_id)
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps)

        output = model(x, use_cache=True)
        past_key_values = output.past_key_values

        mask_index = (x == mask_id)
        mask_index[:, current_block_end:] = 0
        if factor is None:
            x0, transfer_index, _ = get_transfer_index(output.logits, temperature, remasking, mask_index, x, num_transfer_tokens[:, 0] if threshold is None else None, threshold)
        else:
            x0, transfer_index = get_transfer_index_dynamic(output.logits, temperature, remasking, mask_index, x, None, factor)
        x[transfer_index] = x0[transfer_index]

        new_past_key_values = []
        for i in range(len(past_key_values)):
            new_past_key_values.append(())
            for j in range(len(past_key_values[i])):
                new_past_key_values[i] += (past_key_values[i][j][:, :, :current_block_start],)

        past_key_values = new_past_key_values
        nfe += 1

        i = 1
        while True:
            if (x[:, current_block_start:current_block_end] == mask_id).sum() == 0:
                break
            nfe += 1
            mask_index = (x[:, current_block_start:] == mask_id)
            mask_index[:, block_length:] = 0

            logits = model(x[:, current_block_start:], past_key_values=past_key_values, use_cache=True).logits

            logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
            x0 = torch.argmax(logits_with_noise, dim=-1)

            if factor is None:
                x0, transfer_index, _ = get_transfer_index(logits, temperature, remasking, mask_index,
                                                x[:, current_block_start:], num_transfer_tokens[:, i] if threshold is None else None, threshold)
            else:
                x0, transfer_index = get_transfer_index_dynamic(logits, temperature, remasking, mask_index,
                                                x[:, current_block_start:], None, factor)
            x[:, current_block_start:][transfer_index] = x0[transfer_index]

            i += 1

    return x, nfe

@ torch.no_grad()
def generate_with_dual_cache(model, prompt, steps=128, gen_length=128, block_length=128, temperature=0.,
            remasking='low_confidence', mask_id=126336, threshold=None, factor=None, threshold_early=None, use_cluster=True):
    '''
    Args:
        model: Mask predictor.
        prompt: A tensor of shape (1, L).
        steps: Sampling steps, less than or equal to gen_length.
        gen_length: Generated answer length.
        block_length: Block length, less than or equal to gen_length. If less than gen_length, it means using semi_autoregressive remasking.
        temperature: Categorical distribution sampling temperature.
        remasking: Remasking strategy. 'low_confidence' or 'random'.
        mask_id: The token id of [MASK] is 126336.
    '''
    x = torch.full((prompt.shape[0], prompt.shape[1] + gen_length), mask_id, dtype=torch.long).to(model.device)
    x[:, :prompt.shape[1]] = prompt.clone()

    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length

    assert steps % num_blocks == 0
    steps = steps // num_blocks
    eps = 1e-8
    peak_min = 0.1
    nfe = 0
    efficient = False
    block_stats = []
    seq_len = prompt.shape[1]

    if not use_cluster:
        for num_block in range(num_blocks):
            block_nfe = 0
            current_block_start = prompt.shape[1] + num_block * block_length
            current_block_end = current_block_start + block_length

            block_mask_index = (x[:, current_block_start:current_block_end] == mask_id)
            num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps)

            output = model(x, use_cache=True, use_cluster=False)
            past_key_values = output.past_key_values
            nfe += 1
            block_nfe += 1

            replace_position = torch.zeros_like(x, dtype=torch.bool)
            replace_position[:, current_block_start:current_block_end] = 1

            mask_index = (x == mask_id)
            mask_index[:, current_block_end:] = 0
            if factor is None:
                x0, transfer_index, _ = get_transfer_index(
                    output.logits,
                    temperature,
                    remasking,
                    mask_index,
                    x,
                    num_transfer_tokens[:, 0] if threshold is None else None,
                    threshold,
                )
            else:
                x0, transfer_index = get_transfer_index_dynamic(
                    output.logits,
                    temperature,
                    remasking,
                    mask_index,
                    x,
                    None,
                    factor,
                )
            x[transfer_index] = x0[transfer_index]

            i = 1
            while True:
                if (x[:, current_block_start:current_block_end] == mask_id).sum() == 0:
                    break

                nfe += 1
                block_nfe += 1
                mask_index = (x[:, current_block_start:current_block_end] == mask_id)

                logits = model(
                    x[:, current_block_start:current_block_end],
                    past_key_values=past_key_values,
                    use_cache=True,
                    replace_position=replace_position,
                    use_cluster=False,
                ).logits

                if factor is None:
                    x0, transfer_index, _ = get_transfer_index(
                        logits,
                        temperature,
                        remasking,
                        mask_index,
                        x[:, current_block_start:current_block_end],
                        num_transfer_tokens[:, i] if threshold is None else None,
                        threshold,
                    )
                else:
                    x0, transfer_index = get_transfer_index_dynamic(
                        logits,
                        temperature,
                        remasking,
                        mask_index,
                        x[:, current_block_start:current_block_end],
                        None,
                        factor,
                    )
                blk_old = x[:, current_block_start:current_block_end]
                blk_new = torch.where(transfer_index, x0, blk_old)
                x = torch.cat([x[:, :current_block_start], blk_new, x[:, current_block_end:]], dim=1)
                i += 1

            block_stats.append({
                "block_id": num_block,
                "block_start": int(current_block_start),
                "block_end": int(current_block_end),
                "block_nfe": int(block_nfe),
                "block0_true": int(block_nfe),
                "cluster_true": 0,
                "efficient_true": 0,
            })
        return x, nfe, block_stats, seq_len

    for num_block in range(num_blocks):
        block_nfe = 0
        block_cluster_true = 0
        block_efficient_true = 0
        block_block0_true = 0
        dec_tok_number_until_update = 0
        current_block_start = prompt.shape[1] + num_block * block_length
        current_block_end = current_block_start + block_length
        replace_position = torch.zeros_like(x, dtype=torch.bool)
        replace_position[:, current_block_start:current_block_end] = 1

        block_mask_index = (x[:, current_block_start:current_block_end] == mask_id)
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps)

        num_of_mask = (x[:, current_block_start:current_block_end] == mask_id).sum().item()
        if num_of_mask == 0:
            continue
        if num_block % 4 == 3:

            block_prefix_window = max(0, current_block_start - 2*block_length)
            block_suffix_window = min(prompt.shape[1] + gen_length, current_block_end + block_length)

            x_window = x[:, block_prefix_window:block_suffix_window]
            output = model(
                x_window,
                use_cache=True,
                replace_position=replace_position,
                past_key_values=past_key_values,
                cluster=False,
                get_token_drift_score=True,
                efficient=True,
                use_cluster=True,
            )

            past_key_values = output.past_key_values
            drift_scores = output.drift_scores

            local_block_start = current_block_start - block_prefix_window
            local_block_end = current_block_end - block_prefix_window
            block_logits = output.logits[:, local_block_start:local_block_end, :]
            if output.past_hidden_states is not None:
                if past_hidden_states is None:
                    past_hidden_states = output.past_hidden_states
                else:
                    for layer_idx, layer_hidden in enumerate(output.past_hidden_states):
                        past_hidden_states[layer_idx][:, block_prefix_window:block_suffix_window, :] = layer_hidden

            clustered_hidden_states = output.clustered_hidden_states
        else:
            output = model(
                x,
                use_cache=True,
                replace_position=replace_position,
                cluster=False,
                get_token_drift_score=True,
                use_cluster=True,
            )
            past_key_values = output.past_key_values
            past_hidden_states = output.past_hidden_states
            clustered_hidden_states = output.clustered_hidden_states
            drift_scores = output.drift_scores

            block_logits = output.logits[:, current_block_start:current_block_end, :]

        mask_index_block = (x[:, current_block_start:current_block_end] == mask_id)
        x_block = x[:, current_block_start:current_block_end]
        nfe += 1
        block_nfe += 1
        block_block0_true += int(num_block % 4 != 3)
        block_efficient_true += int(num_block % 4 == 3)

        mask_index = (x == mask_id)
        mask_index[:, current_block_end:] = 0
        if factor is None:
            x0_block, transfer_index, _ = get_transfer_index(
                block_logits,
                temperature,
                remasking,
                mask_index_block,
                x_block,
                num_transfer_tokens[:, 0] if threshold is None else None,
                threshold,
            )
        else:
            x0_block, transfer_index = get_transfer_index_dynamic(
                block_logits,
                temperature,
                remasking,
                mask_index_block,
                x_block,
                None,
                factor,
            )

        x[:, current_block_start:current_block_end][transfer_index] = x0_block[transfer_index]

        i = 1
        replace_position = torch.zeros_like(x, dtype=torch.bool)
        replace_position[:, current_block_start:current_block_end] = 1

        w = 5
        drift_window = []

        while True:
            num_of_mask = (x[:, current_block_start:current_block_end] == mask_id).sum().item()
            if num_of_mask == 0:
                break

            nfe += 1
            block_nfe += 1
            dec_tok_number_until_update += transfer_index.sum().item()
            mask_index = (x[:, current_block_start:current_block_end] == mask_id)

            cluster = transfer_index.sum().item() > 1 or dec_tok_number_until_update > 2

            topk=4

            if cluster:
                block_cluster_true += 1
                dec_tok_number_until_update = 0
            if efficient:
                block_efficient_true += 1
            output_decode = model(x[:, current_block_start:current_block_end], past_key_values=past_key_values,
                                  past_hidden_states=past_hidden_states, clustered_hidden_states=clustered_hidden_states, transfer_index=None, cluster=cluster, topk=topk,efficient=efficient,
                                  use_cache=True, replace_position=replace_position, get_token_drift_score=True, use_cluster=True)
            logits = output_decode.logits
            clustered_hidden_states = output_decode.clustered_hidden_states if cluster else clustered_hidden_states
            past_hidden_states = output_decode.past_hidden_states if cluster else past_hidden_states
            past_key_values = output_decode.past_key_values
            logits_sel = output_decode.logits_sel
            sel_indices = output_decode.sel_indices
            drift_scores = output_decode.drift_scores
            if drift_scores is not None:
                if isinstance(drift_scores, (list,tuple)):
                    d = drift_scores[-1]
                else:
                    d = drift_scores

                if d.dim() == 2:
                    d = d[0]

                if d.size(0) > block_length:
                    d_block = d[current_block_start:current_block_end]
                else:
                    d_block = d

                if len(drift_window) > 0:

                    prev_stack = torch.stack(drift_window, dim=0)
                    prev_base = prev_stack.mean(dim=0)
                    delta = (d_block - prev_base)

                else:
                    delta = d_block
                drift_window.append(d_block.detach())
                if len(drift_window) > w:
                    drift_window.pop(0)

            if factor is None:
                x0, transfer_index, conf = get_transfer_index(logits, temperature, remasking, mask_index,
                                                x[:, current_block_start:current_block_end], num_transfer_tokens[:, i] if threshold is None else None, threshold)
            else:
                x0, transfer_index = get_transfer_index_dynamic(logits, temperature, remasking, mask_index,
                                                x[:, current_block_start:current_block_end], None, factor)

            if threshold_early is not None:
                mask_index_step = mask_index.clone()

                conf_block = conf

                alpha = 1.0

                diff = torch.clamp(0.9 - conf_block, min=0.0)
                diff = 10 * (diff ** 2)
                dynamic_gate = alpha * diff
                early_conf_mask = (conf_block > threshold_early) & (conf_block < threshold)
                block_mask = mask_index_step
                drift_mask = (delta.unsqueeze(0) >= dynamic_gate) & (delta.unsqueeze(0) < 2.0)
                early_commit_block = early_conf_mask & drift_mask & block_mask
                transfer_index |= early_commit_block

            x[:, current_block_start:current_block_end][transfer_index] = x0[transfer_index]
            num_of_mask = (x[:, current_block_start:current_block_end] == mask_id).sum().item()
            if logits_sel is not None and sel_indices is not None and cluster:
                suffix_mask = (sel_indices >= current_block_end)

                if suffix_mask.any():
                    suffix_sel_indices = sel_indices[suffix_mask]
                    suffix_logits = logits_sel[:, suffix_mask, :]
                    suffix_actual_mask = (x[:, suffix_sel_indices] == mask_id)

                    if suffix_actual_mask.any():
                        x_suffix_new, transfer_idx_local = get_pre_transfer_index(
                            suffix_logits,
                            temperature,
                            remasking,
                            suffix_actual_mask,
                            x[:, suffix_sel_indices],
                            num_transfer_tokens=None,
                            threshold=threshold
                        )
                        selected_local_pos = transfer_idx_local[0].nonzero(as_tuple=True)[0]
                        if selected_local_pos.numel() > 0:
                            target_global_indices = suffix_sel_indices[selected_local_pos]
                            x[0, target_global_indices] = x_suffix_new[0, selected_local_pos]
            i += 1
        block_stats.append({
            "block_id": num_block,
            "block_start": int(current_block_start),
            "block_end": int(current_block_end),
            "block_nfe": int(block_nfe),
            "block0_true": int(block_block0_true),
            "cluster_true": int(block_cluster_true),
            "efficient_true": int(block_efficient_true),
        })
    return x, nfe, block_stats, seq_len


def get_pre_transfer_index(logits, temperature, remasking, mask_index, x, num_transfer_tokens, threshold=None):
    logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
    x0 = torch.argmax(logits_with_noise, dim=-1)

    if remasking == 'low_confidence':
        p = F.softmax(logits.to(torch.float32), dim=-1)
        x0_p = torch.squeeze(
            torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1)
    elif remasking == 'random':
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

def get_transfer_index(logits, temperature, remasking, mask_index, x, num_transfer_tokens, threshold=None):
    logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
    x0 = torch.argmax(logits_with_noise, dim=-1)

    if remasking == 'low_confidence':
        p = F.softmax(logits.to(torch.float32), dim=-1)
        x0_p = torch.squeeze(
            torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1)
    elif remasking == 'random':
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

def get_transfer_index_dynamic(logits, temperature, remasking, mask_index, x, num_transfer_tokens, factor=1):
    logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
    x0 = torch.argmax(logits_with_noise, dim=-1)
    if remasking == 'low_confidence':
        p = F.softmax(logits.to(torch.float64), dim=-1)
        x0_p = torch.squeeze(
            torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1)
    elif remasking == 'random':
        x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
    else:
        raise NotImplementedError(remasking)

    x0 = torch.where(mask_index, x0, x)
    confidence = torch.where(mask_index, x0_p, -np.inf)

    transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
    num_transfer_tokens = mask_index.sum(dim=1, keepdim=True)

    for j in range(confidence.shape[0]):
        ns=list(range(1,num_transfer_tokens[j]+1))
        es=[factor/(n+1) for n in ns]
        threshs=[1-e for e in es]

        threshs[0]=-1
        sorted_confidence=torch.sort(confidence[j][mask_index[j]],dim=-1,descending=True)[0]
        assert len(sorted_confidence)==len(threshs)
        for top_i in range(len(threshs)):
            if sorted_confidence[top_i]<threshs[top_i]:
                break

        if top_i == 0 or top_i == len(threshs)-1:
            top_i+=1

        _, select_index = torch.topk(confidence[j], k=top_i)
        transfer_index[j, select_index] = True

    return x0, transfer_index
