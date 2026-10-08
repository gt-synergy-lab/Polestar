"""Identity resampler used by the LLaDA-V checkpoint."""

import torch


class IdentityMap(torch.nn.Module):
    def forward(self, x, *args, **kwargs):
        return x

    @property
    def config(self):
        return {"mm_resampler_type": None}


def build_vision_resampler(model_args, **kwargs):
    resampler_type = getattr(model_args, "mm_resampler_type", None)
    if resampler_type is not None:
        raise ValueError(f"Unsupported LLaDA-V resampler: {resampler_type}")
    return IdentityMap()
