"""Compatibility shim for GroundingDINO on modern transformers.

GroundingDINO (IDEA-Research, 2023) binds three ``transformers`` methods on
:class:`BertModel`: ``get_extended_attention_mask``, ``invert_attention_mask``
and ``get_head_mask``. They were removed in transformers 4.45+ (its encoder now
builds these masks internally via ``_create_attention_masks``), so importing or
running the model fails with ``'BertModel' object has no attribute ...``.

We can't just downgrade ``transformers``: old ``tokenizers`` (which those
versions depend on) ship no Python 3.14 wheels and tokenizers has no source
build fallback (Rust). Instead we re-attach the original implementations to the
class, matching exactly the semantics the encoder still consumes.
"""

from __future__ import annotations

_APPLIED = False


def apply() -> None:
    """Reattach the removed mask helpers to ``BertModel`` (idempotent)."""
    global _APPLIED
    if _APPLIED:
        return

    import torch
    from transformers import BertModel

    def _get_extended_attention_mask(self, attention_mask, input_shape, device=None, dtype=None):  # noqa: ARG001
        if attention_mask is None:
            return None
        if attention_mask.dim() == 3:
            extended = attention_mask[:, None, :, :]
        elif attention_mask.dim() == 2:
            extended = attention_mask[:, None, None, :]
        else:
            raise ValueError(
                f"Wrong shape for attention_mask (got {attention_mask.shape})"
            )
        dtype = dtype if dtype is not None else self.dtype
        extended = extended.to(dtype=dtype)
        if dtype == torch.float16:
            extended = (1.0 - extended) * -65500.0
        else:
            extended = (1.0 - extended) * torch.finfo(dtype).min
        return extended

    def _invert_attention_mask(self, encoder_attention_mask):
        if not isinstance(encoder_attention_mask, torch.Tensor):
            raise TypeError(
                f"Wrong type for encoder_attention_mask (got {type(encoder_attention_mask)})"
            )
        if encoder_attention_mask.dim() == 3:
            extended = encoder_attention_mask[:, None, :, :]
        elif encoder_attention_mask.dim() == 2:
            extended = encoder_attention_mask[:, None, None, :]
        else:
            raise ValueError(
                f"Wrong shape for encoder_attention_mask (got {encoder_attention_mask.shape})"
            )
        extended = extended.to(dtype=self.dtype)
        if self.dtype == torch.float16:
            extended = (1.0 - extended) * -65500.0
        else:
            extended = (1.0 - extended) * torch.finfo(self.dtype).min
        return extended

    def _get_head_mask(self, head_mask, num_hidden_layers):
        if head_mask is None:
            return [None] * num_hidden_layers
        if head_mask.dim() == 1:
            head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
            head_mask = head_mask.expand(num_hidden_layers, -1, -1, -1, -1)
        elif head_mask.dim() == 2:
            head_mask = head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)
        if head_mask.dim() != 5:
            raise ValueError(f"head_mask.dim must be 5, got {head_mask.dim()}")
        return head_mask.to(dtype=self.dtype)

    for name, fn in (
        ("get_extended_attention_mask", _get_extended_attention_mask),
        ("invert_attention_mask", _invert_attention_mask),
        ("get_head_mask", _get_head_mask),
    ):
        if not hasattr(BertModel, name):
            setattr(BertModel, name, fn)

    _APPLIED = True