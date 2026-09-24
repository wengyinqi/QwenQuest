from __future__ import annotations

import json

import pytest

from qwenquest.config import (
    PRESETS,
    QWEN3_30B_A3B,
    QWEN3_30B_A3B_2507,
    Qwen3MoeConfig,
    SparseConfig,
    count_parameters,
    parse_hisparse_config,
)

# Layout of the official Qwen3-30B-A3B config.json (the 2507 refresh only
# changes rope_theta and max_position_embeddings).
OFFICIAL = {
    "architectures": ["Qwen3MoeForCausalLM"],
    "attention_bias": False,
    "bos_token_id": 151643,
    "decoder_sparse_step": 1,
    "eos_token_id": 151645,
    "head_dim": 128,
    "hidden_act": "silu",
    "hidden_size": 2048,
    "intermediate_size": 6144,
    "max_position_embeddings": 262144,
    "mlp_only_layers": [],
    "model_type": "qwen3_moe",
    "moe_intermediate_size": 768,
    "norm_topk_prob": True,
    "num_attention_heads": 32,
    "num_experts": 128,
    "num_experts_per_tok": 8,
    "num_hidden_layers": 48,
    "num_key_value_heads": 4,
    "rms_norm_eps": 1e-06,
    "rope_scaling": None,
    "rope_theta": 10000000,
    "tie_word_embeddings": False,
    "vocab_size": 151936,
}


def test_preset_parameter_counts_match_the_model_card():
    # Model card: "30.5B in total and 3.3B activated".
    total, active = count_parameters(QWEN3_30B_A3B_2507)
    assert abs(total / 1e9 - 30.53) < 0.01
    assert abs(active / 1e9 - 3.35) < 0.01
    assert count_parameters(QWEN3_30B_A3B) == (total, active)


def test_presets():
    cfg = PRESETS["qwen3-30b-a3b-thinking-2507"]
    assert cfg.num_kv_groups == 8
    assert cfg.kv_bytes_per_token(2) == 96 * 1024
    assert all(cfg.is_moe_layer(i) for i in range(cfg.num_hidden_layers))
    assert QWEN3_30B_A3B.rope_theta == 1e6 and QWEN3_30B_A3B.max_position_embeddings == 40960


def test_from_dict_official_layout(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps(OFFICIAL))
    cfg = Qwen3MoeConfig.from_pretrained(tmp_path)
    assert cfg == QWEN3_30B_A3B_2507


def test_from_dict_transformers5_layout():
    d = dict(OFFICIAL)
    d["num_local_experts"] = d.pop("num_experts")
    d.pop("rope_theta")
    d.pop("rope_scaling")
    d["rope_parameters"] = {"rope_type": "default", "rope_theta": 10000000}
    assert Qwen3MoeConfig.from_dict(d) == QWEN3_30B_A3B_2507


@pytest.mark.parametrize(
    "fields",
    [
        {
            "rope_scaling": {
                "rope_type": "yarn",
                "factor": 4.0,
                "original_max_position_embeddings": 32768,
            }
        },
        {
            "rope_scaling": {
                "type": "yarn",
                "factor": 4.0,
                "original_max_position_embeddings": 32768,
            }
        },
        {
            "rope_scaling": None,
            "rope_parameters": {
                "rope_type": "yarn",
                "rope_theta": 10000000,
                "factor": 4.0,
                "original_max_position_embeddings": 32768,
            },
        },
    ],
)
def test_yarn_is_parsed_from_every_layout(fields):
    cfg = Qwen3MoeConfig.from_dict({**OFFICIAL, **fields})
    assert cfg.rope_scaling == {
        "rope_type": "yarn",
        "factor": 4.0,
        "original_max_position_embeddings": 32768,
    }
    assert cfg.rope_theta == 1e7


def test_config_round_trips_through_dict():
    assert Qwen3MoeConfig.from_dict(QWEN3_30B_A3B_2507.to_dict()) == QWEN3_30B_A3B_2507


def test_invalid_configs_are_rejected():
    with pytest.raises(ValueError):
        Qwen3MoeConfig(num_attention_heads=30, num_key_value_heads=4)
    with pytest.raises(ValueError):
        Qwen3MoeConfig.from_dict({**OFFICIAL, "model_type": "qwen3"})
    with pytest.raises(NotImplementedError):
        Qwen3MoeConfig(rope_scaling={"rope_type": "longrope"})


# ---------------------------------------------------------------- --hisparse-config


def test_hisparse_defaults_match_upstream():
    cfg = parse_hisparse_config('{"algorithm": "quest"}')
    assert (cfg.top_k, cfg.quest_page_size, cfg.device_buffer_size) == (2048, 64, 4096)
    assert (cfg.host_to_device_ratio, cfg.swap_in_block_size) == (2, 960)
    assert cfg.avoid_recent_overlap is False
    # Without algorithm="quest" there is no quest page size (DSA-native path).
    assert parse_hisparse_config(None).quest_page_size is None


def test_hisparse_readme_example():
    cfg = parse_hisparse_config(
        '{"top_k": 2048, "device_buffer_size": 6144, "host_to_device_ratio": 10, '
        '"swap_in_block_size": 960, "algorithm": "quest", "quest_page_size": 32}'
    )
    assert cfg.device_buffer_size == 6144
    assert cfg.quest_page_size == 32
    assert parse_hisparse_config(cfg) is cfg


def test_unknown_keys_go_to_extra_config():
    cfg = parse_hisparse_config({"algorithm": "quest", "avoid_recent_overlap": True, "foo": 1})
    assert cfg.avoid_recent_overlap is True
    assert cfg.sparse_extra_config == {"avoid_recent_overlap": True, "foo": 1}


@pytest.mark.parametrize(
    "raw",
    [
        '{"algorithm": "quest", "top_k": 100}',  # 100 % 64 != 0
        '{"top_k": 2048, "device_buffer_size": 1024}',  # buffer < top_k
        '{"swap_in_block_size": 2048}',
        '{"swap_in_block_size": true}',
        "{not json",
    ],
)
def test_hisparse_validation(raw):
    with pytest.raises(ValueError):
        parse_hisparse_config(raw)


def test_sparse_config_is_frozen():
    cfg = SparseConfig()
    with pytest.raises(AttributeError):
        cfg.top_k = 1  # type: ignore[misc]
