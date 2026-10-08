# Polestar implementation

Polestar uses representation drift to connect cache calibration and token commitment. Read [Algorithm 1 in the paper](https://arxiv.org/pdf/2607.14107) for the complete method.

| Operation | LLaDA implementation | Dream implementation |
|---|---|---|
| Block generation and commitment | `models/llada/generation.py` | `models/dream/generation.py` |
| Sparse layer updates and cache patching | `models/llada/modeling_llada.py` | `models/dream/modeling_dream.py` |
| Spherical clustering | `models/llada/clustering.py` | `models/dream/clustering.py` |

Paths above are relative to `src/polestar/`.

## Cache calibration

The current block is recomputed during denoising. Cached hidden states in the prefix/suffix window are summarized by cluster centroids. Proxy attention measures KL drift and ranks clusters for sparse update. Selected hidden states are recomputed, and their layer outputs patch the corresponding hidden states, KV entries, and centroids in the next layer.

## Token commitment

Current-block token drift is compared with its recent history. A confidence-conditioned gate uses this delta to identify contextual adaptation events relevant to commitment. Confidence-based selection and a progress fallback keep decoding moving. Selected final-layer outputs also provide candidate logits for nearby suffix positions.

## Multimodal integration

`models/llada_v/` contains image preparation, the language model and vision tower, and its generation/attention hook. `examples/image_generation.py` provides the input path; the model guide and vision evaluation preset provide the checkpoint and task paths.

## Configuration

Use the checked-in model presets for standard execution. Decoder policy and layer-specific update logic are maintained with the corresponding model implementation. When extending Polestar to a new architecture, preserve attention-head layout, token alignment, cached-state indices, and the order of update-packet propagation.
