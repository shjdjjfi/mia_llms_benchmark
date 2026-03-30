# EM-MIAs 复现（核心方法）

本目录提供一个可直接运行的 **EM-MIAs 核心流程**：

1. 对每条文本提取 4 个特征：
   - `LOSS`：目标模型平均 token-level loss
   - `Reference`：`target_loss - reference_loss`
   - `Min-k%`：最低概率 token（等价于最高 loss token）中前 k% 的平均 log-prob
   - `zlib`：`target_loss / zlib_compressed_length`
2. 将四个特征拼接为向量。
3. 使用 `XGBoost` 训练二分类器预测成员/非成员。
4. 输出 `AUC-ROC, Accuracy, Precision, Recall, F1`。

---

## 1. 依赖安装

在仓库根目录执行：

```bash
pip install -r requirements.txt
```

> 已在 `requirements.txt` 中补充 `xgboost`。

---

## 2. 数据来源（优先使用仓库已有下载方式）

本实现支持三种数据来源，推荐顺序如下：

1. **仓库原生 MIMIR 加载方式**（`utils.load_mimir_dataset`）
2. **HuggingFace dataset**（`datasets.load_dataset`）
3. 本地 `jsonl/csv`

### 2.1 使用仓库原生 MIMIR 方式（推荐）

```bash
python em_mias_cli.py \
  --mimir-name pythia \
  --dataset-split ngram_13_0.8/train \
  --target-model EleutherAI/pythia-2.8b \
  --reference-model EleutherAI/pythia-1.4b \
  --device cpu
```

### 2.2 使用 HF dataset（需要有 text/label 列）

```bash
python em_mias_cli.py \
  --dataset-name your_dataset_name \
  --dataset-split train \
  --target-model gpt2 \
  --reference-model distilgpt2
```

### 2.3 本地文件格式

支持 `jsonl` 或 `csv`，每条记录需包含：

- `text`：文本
- `label`：`1`(member) / `0`(non-member)

### JSONL 示例

```json
{"text": "sample text 1", "label": 1}
{"text": "sample text 2", "label": 0}
```

---

## 3. 命令行运行

### 3.1 使用本地数据

```bash
python em_mias_cli.py \
  --data-path path/to/data.jsonl \
  --target-model gpt2 \
  --reference-model distilgpt2 \
  --device cpu \
  --output-json outputs/em_mias_metrics.json
```

### 3.2 无数据时自动生成示例数据（仅调试）

```bash
python em_mias_cli.py \
  --generate-example-data \
  --target-model sshleifer/tiny-gpt2 \
  --reference-model distilgpt2 \
  --device cpu \
  --output-json outputs/em_mias_example_metrics.json
```

> 如果你仍想用模块方式，也可以：`python -m em_mias.cli ...`（需在仓库根目录执行，或先把仓库根目录加入 `PYTHONPATH`）。

---

## 4. 代码结构（模块化）

- `data.py`：数据加载与自动生成示例数据
- `features.py`：基于 `transformers` 的特征提取
- `model.py`：`xgboost + sklearn` 训练与评估
- `cli.py`：命令行入口，串联全流程

---

## 5. 复现时的合理假设（论文细节缺失处）

1. `Min-k%` 按 “最低概率 token” 实现为 “最大 loss token”的 top-k%。
2. `Reference` 采用最常见定义：`target_avg_loss - reference_avg_loss`。
3. 二分类阈值固定为 0.5（可按需求后续扩展为最优阈值搜索）。
4. 默认随机划分训练/测试集（`test_size=0.3`，可通过参数调整）。

---

## 6. 输出

程序会在终端打印并在 `--output-json` 保存如下结构：

```json
{
  "dataset_path": "...",
  "generated_example_data": false,
  "target_model": "...",
  "reference_model": "...",
  "n_samples": 100,
  "assumptions": [...],
  "metrics": {
    "auc_roc": 0.90,
    "accuracy": 0.83,
    "precision": 0.84,
    "recall": 0.81,
    "f1": 0.82,
    "n_train": 70,
    "n_test": 30
  }
}
```
