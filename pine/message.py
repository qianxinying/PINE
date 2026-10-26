"""Original relation complex multiplication followed by optional time rotation."""

import torch


def relation_then_time_rotation(source, relation, phase=None):
    """Compute (h * r) * exp(i * phase), or just h * r when disabled."""
    if source.shape[-1] % 2:
        raise ValueError("complex message dimensions must be even")
    h_real, h_imag = source.chunk(2, dim=-1)
    r_real, r_imag = relation.chunk(2, dim=-1)
    real = h_real * r_real - h_imag * r_imag
    imag = h_real * r_imag + h_imag * r_real
    if phase is not None:
        cosine, sine = phase.cos(), phase.sin()
        real, imag = real * cosine - imag * sine, real * sine + imag * cosine
    return torch.cat([real, imag], dim=-1)
