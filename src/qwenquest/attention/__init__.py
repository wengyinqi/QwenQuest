"""Attention backends: the only thing that differs between the three modes.

=================  =======================  ===================================
``attention_mode``  SGLang backend           decode reads
=================  =======================  ===================================
``dense``           ``flashinfer``           every past token (device pool)
``quest``           ``flashinfer_quest``     Quest top-k (device pool)
``quest_hisparse``  ``flashinfer_hisparse``  Quest top-k (host pool + hot buffer)
=================  =======================  ===================================
"""

from qwenquest.attention.base import AttentionBackend
from qwenquest.attention.dense import DenseBackend
from qwenquest.attention.quest_hisparse import QuestHiSparseBackend
from qwenquest.attention.quest_only import QuestOnlyBackend

ATTENTION_MODES = ("dense", "quest", "quest_hisparse")

__all__ = [
    "ATTENTION_MODES",
    "AttentionBackend",
    "DenseBackend",
    "QuestHiSparseBackend",
    "QuestOnlyBackend",
]
