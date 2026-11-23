# genrec-ad-recall

面向工业广告场景的生成式与向量召回实验框架，基于
[TencentGR-1M](https://huggingface.co/datasets/TAAC2025) 构建。项目覆盖
数据审计、下一次点击样本构造、序列向量召回、RQ-KMeans/RQ-VAE
Semantic ID、约束自回归生成、公平评测和消融分析。

`PyTorch` · `SASRec` · `HSTU` · `InfoNCE` · `RQ-KMeans` · `RQ-VAE` ·
`Qwen2.5-0.5B` · `LoRA` · `Trie Beam Search` · `Faiss`

## Highlights

- 在 118,457 个测试用户和 660,000 个候选广告上完成统一的全候选召回评测。
- 序列向量召回融合广告 ID、13 类广告特征、用户画像、Action 与 Time-gap，
  使用 InfoNCE、混合负采样、LogQ 纠偏和难负样本训练。
- 构建 4 层 RQ-KMeans Semantic ID，通过碰撞后缀为 4,783,154 个 Item
  建立唯一映射，并实现分层自回归生成和候选前缀 Trie 约束 Beam Search。
- 实现 RQ-KMeans 与 RQ-VAE Tokenizer 对照，并完成 Qwen2.5-0.5B 的 SID
  词表扩展、Embedding 预热和 LoRA 训练。
- 提供 HSTU、Relative Attention Bias、Action/Time、负采样、模块留一、
  多随机种子、候选预算扩展和分组分析脚本。

## Recall Architecture

```text
TencentGR-1M
    │
    ├── user sequence / user profile
    ├── item ID / item features
    └── candidate set
            │
            ▼
Next-click sample construction
    ├── global time split
    ├── history cap = 100
    └── target = last candidate-covered click
            │
            ├───────────────────────────────┐
            ▼                               ▼
Sequence-to-vector retrieval        Semantic ID generation
    │                                   │
    ├── Item + profile features         ├── collaborative Item Vector
    ├── Action + Time-gap               ├── 4-level RQ-KMeans / RQ-VAE
    ├── causal Transformer              ├── collision suffix
    ├── InfoNCE + negatives             ├── hierarchical Transformer
    └── user/item vectors               ├── Qwen2.5-0.5B LoRA
            │                           └── constrained beam decoding
            ▼                               │
       ANN / exact Top-K                    │
            └──────────────┬────────────────┘
                           ▼
              same-user and same-candidate
              HR@K / NDCG@K / Recall@K
```

## Main Results

### Full-candidate vector recall

评测范围：118,457 个测试用户，660,000 个候选广告。

| Method | HR@10 | NDCG@10 | Score |
|---|---:|---:|---:|
| Official-style SASRec baseline | 0.004288 | 0.002043 | 0.002739 |
| Sequence vector recall | **0.008847** | **0.004243** | **0.005671** |

序列向量召回在统一数据、候选集和指标口径下取得约 `2.07x` 的 Score 提升，
三个随机种子的 Score 均值为 `0.005674`，样本标准差为 `0.0000569`。

### Generative retrieval comparison

同口径范围：10,000 个测试用户，511,029 个可映射候选。

| Method | Representation | Generation / Retrieval | HR@50 | HR@100 |
|---|---|---|---:|---:|
| Vector recall | continuous Item vectors | exact Top-K | **0.0293** | **0.0456** |
| RQ-KMeans SID | 4-level hierarchical codes | causal Transformer + Trie Beam | 0.0245 | 0.0354 |

RQ-KMeans SID 为两个随机种子的均值，合法候选映射率为 `100%`。该分支与向量
召回存在互补命中，随着向量候选池从 Top-100 扩展到 Top-500，SID 独有命中
由 92 个降至 11 个。

### Tokenizer comparison

| Tokenizer | Levels | Codes per level | Reconstruction MSE | Collision rate before suffix | Final SID vocabulary |
|---|---:|---:|---:|---:|---:|
| RQ-KMeans | 4 | 256 | 0.000666 | 4.65% | 1,171 |
| RQ-VAE | 4 | 256 | 0.000683 | 7.53% | 2,780 |

Qwen2.5-0.5B 探索将 RQ-VAE SID、Action、Time-gap 和用户画像映射到扩展词表，
完成 10k 样本 Head 预热和 50k 样本 LoRA 训练。诊断显示首层预测 Top-1
占比为 `63.54%`，真实目标为 `12.40%`；五层完整 SID 精度为 `0.18%`。
训练策略调整后，模型仍集中在少量热门候选，因此最终召回方案采用稳定的
序列向量召回。

## Core Modules

```text
src/taac/
├── data/          # audit, sample construction, feature cache, candidate mapping
├── models/        # SASRec/HSTU, Semantic ID generator
├── training/      # vector-recall training loop
├── evaluation/    # full-candidate retrieval and grouped analysis
├── sid/           # RQ-KMeans, RQ-VAE, Qwen SFT, constrained decoding
├── profiling/     # Faiss benchmark
└── reporting/     # experiment aggregation and plots
```

关键入口：

- `src/taac/training/train.py`: 训练序列向量召回模型。
- `src/taac/models/sasrec.py`: SASRec、HSTU 和 Relative Attention Bias。
- `src/taac/sid/tokenizer.py`: RQ-KMeans Semantic ID。
- `src/taac/sid/rqvae_tokenizer.py`: RQ-VAE Semantic ID。
- `src/taac/models/sid_generator.py`: 分层自回归 SID Generator。
- `src/taac/sid/evaluate_sid.py`: 约束 Beam Search 和向量/SID 对照。
- `src/taac/sid/qwen_sft.py`: Qwen SID 词表扩展、预热和 LoRA SFT。
- `src/taac/evaluation/retrieval.py`: 全候选 Top-K 评测。

## Quick Start

### Environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
export PYTHONPATH="${PWD}/src:${PYTHONPATH:-}"
```

CPU 环境可使用 PyTorch 官方 CPU wheel。Qwen 路线所需的
`transformers`、`peft` 和 `accelerate` 已包含在 `requirements.txt` 中。

### Data layout

运行前从官方数据源获取 TencentGR-1M，并按以下结构组织：

```text
raw/TencentGR-1M/
├── candidate/
├── item_feat/
├── seq/
├── user_feat/
└── indexer.pkl
```

示例路径位于 `configs/paths.example.yaml`。各脚本通过 `TAAC_PROJECT_ROOT`
和命令行参数接收实际数据路径。

### Core commands

```bash
# Audit data
bash scripts/run_audit.sh

# Build next-click samples
bash scripts/build_click_samples.sh

# Train the sequence vector model
bash scripts/train_b1_sasrec_unified.sh

# Evaluate on the full candidate pool
CHECKPOINT=/path/to/best.pt \
SPLIT=test \
OUTPUT=/path/to/eval.json \
bash scripts/eval_checkpoint.sh

# Build RQ-KMeans semantic IDs
bash scripts/build_sid.sh

# Train the hierarchical SID generator
bash scripts/train_sid_generator.sh

# Build RQ-VAE semantic IDs
bash scripts/build_rqvae_sid.sh
```

完整实验顺序和参数参见 `scripts/` 与 `configs/experiments/`。

## Evaluation Protocol

- 使用全局时间切分：train / valid / test = 70% / 15% / 15%。
- 每个用户最多保留目标点击前 100 条历史行为。
- 目标为用户最后一个位于公开候选集中的点击。
- 全候选质量评测使用精确打分，并过滤历史 Item。
- HR@K、NDCG@K 与 Recall@K 在同一样本和候选口径下计算。
- 向量与 SID 对照保持相同用户、候选范围和 Top-K 预算。

TAAC 官方评分使用赛事方隐藏标签。本仓库报告的是基于公开数据构造的本地
时间外推 benchmark，适合复现实验与方法对比，官方榜单成绩以赛事方结果为准。

## License

源代码使用 MIT License。TencentGR-1M 数据和第三方依赖遵循各自官方许可与
数据使用协议。
