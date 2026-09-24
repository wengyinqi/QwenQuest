from __future__ import annotations

import pytest

from qwenquest import __version__
from qwenquest.cli import main


def test_info_for_the_hisparse_setup(capsys):
    assert main(["info"]) == 0
    out = capsys.readouterr().out
    assert "30.53B total, 3.35B active" in out
    assert "31 best complete pages + last 64 tokens = 2048 tokens" in out
    assert "GQA group 8" in out


def test_info_custom_budget(capsys):
    assert main(["info", "--hisparse-config", '{"top_k": 1024, "quest_page_size": 16}']) == 0
    assert "63 best complete pages + last 16 tokens = 1024 tokens" in capsys.readouterr().out


def test_demo_runs_on_cpu(capsys):
    argv = ["demo", "--prompt-len", "80", "--new-tokens", "4", "--top-k", "32", "--page-size", "8"]
    assert main(argv) == 0
    out = capsys.readouterr().out
    assert "Quest decisions" in out
    assert "max |logit diff| = 0  (exact offload)" in out


def test_version_flag(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["--version"])
    assert exc.value.code == 0
    assert __version__ in capsys.readouterr().out
