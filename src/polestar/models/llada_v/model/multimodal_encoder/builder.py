"""Build the SigLip vision encoder used by LLaDA-V."""

from .siglip_encoder import SigLipVisionTower


def build_vision_tower(config, **kwargs):
    vision_tower = getattr(config, "mm_vision_tower", None)
    if vision_tower is None or "siglip" not in vision_tower.lower():
        raise ValueError(f"LLaDA-V requires a SigLip vision tower, received {vision_tower!r}")
    return SigLipVisionTower(vision_tower, vision_tower_cfg=config, **kwargs)
