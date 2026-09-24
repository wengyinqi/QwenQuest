"""QwenQuest - a readable baseline of HiSparse's Qwen3-30B-A3B + Quest decoding.

Where to start reading (tags refer to comments in the code, explained in
docs/quest_walkthrough.md):

* :mod:`qwenquest.model.qwen3_moe` - the model; ``RadixAttention`` is the
  only seam where sparse attention plugs in ([Q1]).
* :mod:`qwenquest.quest` - ``QuestAlgorithm``: page bounds ([Q2], [Q3]),
  per-step layout ([Q4]), criticality ([Q5]), top-k layout ([Q6]).
* :mod:`qwenquest.attention` - the three decode modes; [Q7] is where the
  selected positions become the K/V rows attention reads.
* :mod:`qwenquest.hisparse` - host pool + hot buffer + LRU swap-in ([H1]-[H4]).
* :mod:`qwenquest.engine` - the serving loop that calls all of the above.
"""

from qwenquest._version import __version__
from qwenquest.config import (
    PRESETS,
    QWEN3_30B_A3B,
    QWEN3_30B_A3B_2507,
    Qwen3MoeConfig,
    SparseConfig,
    parse_hisparse_config,
    tiny_config,
)
from qwenquest.engine import Engine, EngineConfig, GenerationOutput
from qwenquest.model import Qwen3MoeForCausalLM, init_random_weights, load_qwen3_moe
from qwenquest.quest import QuestAlgorithm, QuestTrace

#: Upstream code this baseline mirrors.
UPSTREAM_REPO = "https://github.com/sgl-project/sglang"
UPSTREAM_BRANCH = "hisparse_quest"
UPSTREAM_COMMIT = "e568f8a362b4ae3a597b28b75716309814aa33fb"

__all__ = [
    "PRESETS",
    "QWEN3_30B_A3B",
    "QWEN3_30B_A3B_2507",
    "UPSTREAM_BRANCH",
    "UPSTREAM_COMMIT",
    "UPSTREAM_REPO",
    "Engine",
    "EngineConfig",
    "GenerationOutput",
    "QuestAlgorithm",
    "QuestTrace",
    "Qwen3MoeConfig",
    "Qwen3MoeForCausalLM",
    "SparseConfig",
    "__version__",
    "init_random_weights",
    "load_qwen3_moe",
    "parse_hisparse_config",
    "tiny_config",
]
