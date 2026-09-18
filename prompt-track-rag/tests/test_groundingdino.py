import os

import numpy as np
import pytest


def test_sentence_caption_format():
    """The GroundingDINO caption must use the official 'class . class .'
    format; a plain comma string makes its block-attention mask builder crash
    with an empty tensor stack."""
    from zero_shot_tracking.detector import GroundingDINODetector

    split = GroundingDINODetector._split_prompt
    assert split("person, car") == ["person", "car"]
    assert split("dog") == ["dog"]

    def caption(prompt: str) -> str:
        return " . ".join(GroundingDINODetector._split_prompt(prompt)) + " ."

    assert caption("person, car") == "person . car ."
    assert caption("dog") == "dog ."
    assert caption("red square, green circle") == "red square . green circle ."


def test_gdino_compat_is_idempotent():
    """The transformers-5.x BERT mask shim applies cleanly, repeatedly."""
    from zero_shot_tracking import gdino_compat

    gdino_compat.apply()
    gdino_compat.apply()
    assert gdino_compat._APPLIED is True

    from transformers import BertModel

    for name in (
        "get_extended_attention_mask",
        "invert_attention_mask",
        "get_head_mask",
    ):
        assert hasattr(BertModel, name), f"missing reattached method {name}"


@pytest.mark.skipif(
    os.environ.get("RUN_REAL") != "1",
    reason="loads the real GroundingDINO model (~1 GB RAM, ~10 s CPU inference); set RUN_REAL=1",
)
def test_real_groundingdino_detects():
    """End-to-end real-model inference on a synthetic frame. Opt in with
    RUN_REAL=1 so the default suite stays fast and torch-free."""
    import cv2
    import torch  # noqa: F401
    from zero_shot_tracking.detector import GroundingDINODetector

    frame = np.zeros((240, 320, 3), np.uint8)
    frame[:] = (235, 240, 200)
    cv2.rectangle(frame, (10, 60), (150, 200), (0, 0, 255), -1)
    cv2.circle(frame, (250, 110), 40, (0, 180, 0), -1)

    det = GroundingDINODetector(
        config_path="checkpoints/GroundingDINO_SwinT_OGC.cfg.py",
        weights_path="checkpoints/groundingdino_swint_ogc.pth",
        device="cpu",
        box_threshold=0.3,
        text_threshold=0.25,
    )
    dets, labels, height = det.detect(frame, "red square, green circle")
    assert height == 240
    assert len(labels) == 2
    assert dets.ndim == 2 and dets.shape[1] == 6
    assert dets.dtype == np.float32
    assert len(dets) > 0, "real model found nothing on the synthetic frame"
    assert (dets[:, 4] > 0).all() and (dets[:, 4] <= 1).all()
    assert set(dets[:, 5].tolist()) <= {0.0, 1.0}