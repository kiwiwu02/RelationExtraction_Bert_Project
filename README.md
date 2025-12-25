# RelationExtraction_Bert_Project

基于预训练大语言模型 BERT 的中文医疗关系抽取（关系分类）项目。项目以 `Chinese-RoBERTa-wwm-ext` 为主干模型，通过多种文本编码策略（实体标注、QA 模板等）与“难分类类别增强”（加权损失 + 两阶段采样）提升关系分类效果。

- 课程/项目材料：`CISC7021_Report.pdf`、`CISC7021_PPT.pdf`
- 代码入口：`Code/train.py`、`Code/test.py`

---

## 1. 任务定义（Relation Classification）

给定一条中文医疗文本 `sentence`，以及其中的一对实体（头实体 `h`、尾实体 `t`），预测它们之间的关系 `r`。

本仓库实现的是 **关系分类**（relation classification）：数据集中每条样本都已给出实体对 `(h, t)`，模型只需判别关系类别。

---

## 2. 关系标签（10 类）

项目在 `Code/data.py` 中定义了 10 个关系标签，并映射为 0-9 的分类 id：

| id | 关系（`r`） |
|---:|---|
| 0 | 临床表现 |
| 1 | 药物治疗 |
| 2 | 同义词 |
| 3 | 病因 |
| 4 | 并发症 |
| 5 | 病理分型 |
| 6 | 实验室检查 |
| 7 | 辅助治疗 |
| 8 | 相关（导致） |
| 9 | 影像学检查 |

其中项目将 `并发症(4)` 与 `相关（导致）(8)` 视作“更难/更重要”的类别，在训练时做了额外增强（见第 5 节）。

---

## 3. 数据集格式

数据位于 `Code/dataset/`，采用 **JSONL**（一行一个 JSON）。每条样本至少包含：

- `id`：样本 id
- `sentence`：原始句子/段落
- `h`：头实体（head entity）
- `t`：尾实体（tail entity）
- `r`：关系标签（上表 10 类之一）

示例（来自 `Code/dataset/train.jsonl`）：

```json
{"id": 0, "sentence": "…", "h": "先天性梅毒", "t": "脑膜炎症状", "r": "临床表现"}
```

注意：部分样本中可能出现 `object_placeholder`（见 `val.jsonl`/`test.jsonl`），属于数据本身现象；代码会把它当作普通尾实体字符串处理。

---

## 4. 文本编码策略

训练脚本 `Code/train.py` 会循环训练 5 个版本（`version = 5`），对应 5 种编码方式：

| v | 名称（`text_build_model[v]`） | 编码方式概述 |
|---:|---|---|
| 0 | `basic1` | `[CLS] h [SEP] sentence [SEP] t` |
| 1 | `basic2` | `[CLS] h [SEP] t [SEP] sentence` |
| 2 | `QA` | `[CLS] h和t之间的关系是什么？[SEP] sentence` |
| 3 | `entity_marked1` | 在句子中用 `[E1]...[/E1]`、`[E2]...[/E2]` 标注实体范围 |
| 4 | `entity_marked2` | 在句子中用 `[实体1]...[/实体1]`、`[实体2]...[/实体2]` 标注实体范围 |

实现位置：`RelationshipDataset.build_text()`（在 `Code/data.py`）。

---

## 5. 方法与训练流程

### 5.1 预训练模型

默认使用：

- `hfl/chinese-roberta-wwm-ext`

你也可以在 `Code/train.py` 中切换到（已写好注释）：

- `dmis-lab/biobert-base-cased-v1.2`（BioBERT，英文医学语料为主）
- `trueto/medbert-base-wwm-chinese`（MedBERT，中文医学语料）

### 5.2 两阶段训练（Curriculum）

`Code/train.py` 将训练分成两段：

1. **Balanced 采样阶段**（`BalancedBatchSampler`，batch 内尽量均匀覆盖类别）
2. **Priority 采样阶段**（`PiorityBatchSampler`，每个 batch 强化类别 `4(并发症)` 与 `8(相关（导致）)`）

### 5.3 加权损失（Hard-class Enhancement）

训练使用 `CrossEntropyLoss(weight=class_weights)`：

- 默认各类权重为 1
- 对 `并发症(4)` 与 `相关（导致）(8)` 设为 2.0（`class_weights[4]=2.0`, `class_weights[8]=2.0`）

> 该策略能缓解 hard class 问题，并降低类间 F1 波动范围。

### 5.4 输出内容

- 日志：`Code/output/log/results_YYYYmmdd_HHMMSS.log`
- loss 曲线（每个版本一个 CSV）：`Code/output/loss/roberta_train_losses{v}.csv`
- 模型与 tokenizer：`Code/checkpoint/model{v}/`

---

## 6. 复现实验

### 6.1 环境准备

推荐 Python 3.10+。

1) 创建环境（任选其一）：

```bash
python -m venv .venv
source .venv/bin/activate
```

或使用 conda：

```bash
conda create -n re-bert python=3.11 -y
conda activate re-bert
```

2) 安装依赖：

```bash
pip install -r requirements.txt
```

> `torch` 在不同平台/显卡上安装方式可能不同；如遇到安装/加速问题，优先参考 PyTorch 官方安装指引。

### 6.2 训练

重要：脚本使用相对路径读取 `./dataset/*.jsonl`，因此请在 `Code/` 目录下运行。

```bash
cd Code
python train.py
```

训练完成后会在 `Code/checkpoint/model0` ~ `Code/checkpoint/model4` 生成 5 个模型目录。

### 6.3 测试/评估

```bash
cd Code
python test.py
```

`test.py` 会：

- 对每个 `./checkpoint/model{v}` 载入模型
- 在 `./dataset/test.jsonl` 上评估
- 输出 `classification_report`（每类 precision/recall/f1 + macro/weighted 平均）
- 额外输出一个“加权 micro 指标”（代码中以每类 accuracy 归一化作为权重，对 TP/FP/FN 做加权）

---

## 7. 实验结果

### 7.1 预训练模型对比

来自 `CISC7021_Report.pdf`（表 5）：

| Model | 对应编码 | Macro-P | Macro-R | Macro-F1 | Micro-F1 |
|---|---|---:|---:|---:|---:|
| BioBERT | Entity_marked1 | 46.00% | 36.30% | 30.80% | 46.28% |
| Chinese-RoBERTa-wwm-ext | Entity_marked2 | **90.93%** | **90.20%** | **90.24%** | **90.46%** |
| MedBERT | QA | 89.79% | 88.70% | 88.80% | 89.25% |

更细的结论：

- **BioBERT 明显落后**：虽然是生物医学预训练，但主要为英文语料；迁移到中文医疗文本时语料/语言不匹配导致整体指标显著下降。
- **Chinese-RoBERTa-wwm-ext 最优**：中文语料与分词/字词粒度更贴合，配合实体显式标注更易聚焦实体对的关系线索。
- **MedBERT 次优但接近**：面向中文医疗语料预训练，在专业术语理解上具备优势；报告中其最优配置为 QA 编码。

### 7.2 编码策略影响（Chinese-RoBERTa）

来自 `CISC7021_Report.pdf`（表 6）：

| Encoding | Macro-P | Macro-R | Macro-F1 | Micro-F1 |
|---|---:|---:|---:|---:|
| Basic1 | 88.94% | 88.60% | 86.71% | 89.01% |
| Basic2 | 89.33% | 88.00% | 88.09% | 88.46% |
| QA | 89.23% | 86.80% | 87.06% | 89.25% |
| Entity_marked1 | 88.34% | 87.30% | 87.60% | 87.81% |
| Entity_marked2 | **90.72%** | **90.20%** | **90.24%** | **90.46%** |

更细的结论：

- **实体标注整体优于纯拼接**：实体范围显式化能减少模型对实体位置的歧义，提升注意力对齐。
- **`entity_marked2` 最优**：用更贴合中文语境的标注符号（`[实体1]...[/实体1]` 等）在报告中表现最佳。
- **顺序有影响但非决定性**：`basic2` 与 `basic1` 的差异说明实体与句子拼接顺序会影响模型提取关系线索的路径。
- **QA 编码是有效替代**：把关系分类显式提示为“问答式任务”，在报告里能获得稳定收益（尤其对未为分类做特殊优化的模型更明显）。

### 7.3 类别增强效果（Hard class：加权损失 + 采样策略）

来自 `CISC7021_Report.pdf`（表 7/8）。报告把 hard class 处理拆成两步：先做 **加权损失**，再叠加 **弱样本增强（augmentation）**。

**(1) Regular loss → Weighted loss（表 7）**

| Setting | Macro-P | Macro-R | Macro-F1 | Micro-F1 | F1 Range |
|---|---:|---:|---:|---:|---:|
| Regular Loss | 88.02% | 88.10% | 87.01% | 86.90% | 26.29% |
| Weighted Loss | 89.62% | 88.60% | 88.71% | 89.01% | 18.01% |

提升幅度（Weighted 相对 Regular）：

- Macro-F1：+1.70%
- Micro-F1：+2.11%
- F1 Range：-8.28%（类间波动显著收敛）

**(2) Weighted → Weighted + Augmentation（表 8）**

| Setting | Macro-P | Macro-R | Macro-F1 | Micro-F1 | F1 Range |
|---|---:|---:|---:|---:|---:|
| Weighted Loss | 89.62% | 88.60% | 88.71% | 89.01% | 18.01% |
| Weighted + Aug | 90.93% | 90.20% | 90.24% | 90.46% | 9.45% |

提升幅度（Weighted+Aug 相对 Weighted）：

- Macro-F1：+1.53%
- Micro-F1：+1.45%
- F1 Range：-8.56%（进一步显著收敛）

**与本仓库代码的对应关系**

- 代码已实现：**加权损失**（对类别 4/8 加权）+ **两阶段采样**（Balanced → Priority，Priority 进一步聚焦类别 4/8）。
- 报告提到但代码未单独实现：**弱样本增强（augmentation）**。因此你用本仓库当前 `train.py` 复现时，更贴近“Weighted Loss + Sampling”的设置；若要严格对齐表 8，需要额外实现数据增强流程。

---

## 8. 目录结构说明

仓库（核心部分）结构如下：

```
.
├── CISC7021_PPT.pdf
├── CISC7021_Report.pdf
└── Code/
    ├── train.py              # 训练入口（5 种编码，保存 5 个 checkpoint）
    ├── test.py               # 测试入口（逐个加载 model0..4）
    ├── data.py               # Dataset + 两种 Sampler + 标签映射
    ├── log.py                # 训练日志
    ├── max_len.py            # 统计文本 token 长度（辅助脚本）
    ├── hfd.sh                # HuggingFace 下载脚本（可选）
    ├── dataset/              # train/val/test jsonl
    ├── checkpoint/           # 训练产物（通常较大）
    └── output/
        ├── log/
        └── loss/
```

---


## 9. 常见问题（FAQ）

### Q1：运行 `train.py` 报错找不到 `./dataset/train.jsonl`？

请确认在 `Code/` 目录下运行：

```bash
cd Code
python train.py
```

### Q2：第一次运行下载 HuggingFace 模型很慢/失败？

- 确保网络可访问 Hugging Face
- 或使用 `Code/hfd.sh` 先下载到本地，再把 `model_name` 改成本地路径

### Q3：GPU/显存不足？

可以尝试：

- 降低 batch size（`BalancedBatchSampler`/`PiorityBatchSampler` 的 `batch_size`）
- 降低 `MAX_LEN`（默认 300）
- 使用 CPU（会更慢）

---

## 10. 致谢

- 课程支持：感谢 CISC7021 教学团队提供的指导与计算资源。
- 领域支持：感谢提供领域知识与验证支持的医疗专业人士。
- 开源生态：感谢开源社区维护的工具与框架（PyTorch、Transformers、scikit-learn 等）。
- 预训练模型：`hfl/chinese-roberta-wwm-ext`、`dmis-lab/biobert-base-cased-v1.2`、`trueto/medbert-base-wwm-chinese`

---

## 11. 总结

- 本项目把中文医疗关系抽取落到一个清晰的 **关系分类** 问题上，并验证了：在中文医疗语境下，选择更匹配语料的预训练模型（Chinese-RoBERTa）与更贴合任务的输入编码（实体显式标注/QA 提示）能带来显著收益。
- 对 hard class 的处理上，“提升难类权重 + 采样聚焦难类”的思路有效：既能提升整体 Macro/Micro 指标，也能降低类别间性能波动（F1 Range）。


