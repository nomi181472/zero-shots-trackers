"""Zero-shot open-vocabulary multi-object tracking example.

Detect arbitrary object categories from a free-text prompt with GroundingDINO,
then associate the detections across frames with a Retrieval-Augmented-
Generation (RAG) tracker — detection crops are embedded (CLIP / histogram),
the top-k similar track memories are *retrieved* from a vector store, and an
assignment head *generates* the association — no training needed.
"""

__version__ = "0.2.0"