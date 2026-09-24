# 版本管理、分支与 CI

## 1. 版本号：语义化版本（SemVer）

* 唯一来源：`src/qwenquest/_version.py` 的 `__version__`；`pyproject.toml` 通过
  `[tool.setuptools.dynamic]` 读取它，`qwenquest --version` 与安装后的包元数据都来自这里。
* 每个版本在 `CHANGELOG.md` 中有一节 `## [X.Y.Z] - YYYY-MM-DD`（[Keep a Changelog](https://keepachangelog.com/) 格式），
  开发中的改动先写在 `## [Unreleased]` 下。
* 本项目对 SemVer 的约定（1.0 之前 minor 视作「大版本」）：

| 改动 | 0.x 阶段 | ≥ 1.0 |
|---|---|---|
| 改变 Quest **选中哪些位置**或注意力数值（例如默认打开 `avoid_recent_overlap`、跟进上游算法变更） | minor | major |
| 新功能、新模式、新 CLI 子命令，不改变已有结果 | minor | minor |
| bug 修复、文档、测试、CI | patch | patch |
| 仅更新上游对照（`UPSTREAM_COMMIT`）且算法未变 | patch | patch |

## 2. 分支与提交

* `main` 为稳定分支，只接受通过 CI 的 PR；功能分支命名如 `feat/...`、`fix/...`（本次开发分支为
  `claude/modest-pasteur-cgvrgf`）。
* 提交信息使用 [Conventional Commits](https://www.conventionalcommits.org/)：
  `feat(quest): ...`、`fix(hisparse): ...`、`test: ...`、`docs: ...`、`ci: ...`、`build: ...`、`chore: ...`。
* 本地钩子：`pip install -e ".[dev]" && pre-commit install`（ruff 检查/格式化 + 版本一致性检查，
  与 CI 使用同一 ruff 版本 0.16.8）。

## 3. 发布流程

```bash
# 1. 在功能分支上：把 [Unreleased] 变成新版本（同时改 _version.py 和 CHANGELOG 链接）
python scripts/release.py bump minor          # 或 patch / major / 0.3.0
#    检查并润色 CHANGELOG 中新版本那一节
python scripts/release.py check
git commit -am "chore(release): v0.2.0"
# 2. 开 PR → CI 通过 → 合并到 main
# 3. 在 main 上打 tag 并推送，Release 工作流自动发布
git tag -a v0.2.0 -m "v0.2.0" && git push origin v0.2.0
```

`Release` 工作流（`.github/workflows/release.yml`）会：校验 tag == `v{__version__}` 且 CHANGELOG 有对应小节
→ 构建 sdist/wheel 并 `twine check` → 用 CHANGELOG 对应小节作为说明创建 GitHub Release 并附上构建产物。

## 4. CI（`.github/workflows/ci.yml`）

在每次 push（所有分支、`v*` tag）和 PR 上运行：

| Job | 内容 |
|---|---|
| Lint & format | `ruff check`、`ruff format --check` |
| Version consistency | `scripts/release.py check`：`_version.py`、CHANGELOG 最新版本/日期/链接一致；tag 推送时还要求 tag 与版本一致 |
| Type check | `mypy`（`src/qwenquest`） |
| Tests (Python 3.10–3.13) | CPU 版 PyTorch 最新版 + transformers：单元测试、上游 Quest 用例移植、HF 数值对齐、三种模式端到端等价性；再跑一遍 CLI（`info`、`demo`）；上传覆盖率 |
| Tests (oldest supported) | Python 3.10 + `torch==2.2.2` + `safetensors==0.4.0`，验证 `pyproject.toml` 中声明的最低版本 |
| Build & check distribution | `python -m build`、`twine check --strict`、在干净环境中安装 wheel 并运行 CLI |

GPU 相关测试带 `gpu` 标记，没有 CUDA 时自动跳过；在有 GPU 的机器上直接 `pytest` 即会运行。

依赖更新：`.github/dependabot.yml` 每周检查 GitHub Actions 与 pip 依赖。

## 5. 跟进上游

本仓库复现的是固定的上游版本（`qwenquest.UPSTREAM_COMMIT`）。上游 Quest 路径有变化时：

1. 对比新旧 commit 中 `quest_algorithm.py`、`hisparse_coordinator.py`、两个 FlashInfer 后端与 `hisparse.cuh`；
2. 同步修改代码与 `docs/upstream_mapping.md`，更新 `UPSTREAM_COMMIT`；
3. 若选中位置或数值改变，按第 1 节规则升级版本，并在 CHANGELOG 中写明。
