# rumor-detection-model-for-AI-class

## 中文版说明

这是一个面向短文本、社交媒体风格文本的谣言检测项目，任务是二分类：

- `1` 表示 `rumor`，即谣言；
- `0` 表示 `non-rumor`，即非谣言。

项目采用一条可复现的本地 pipeline :
- 对输入文本做标准化和安全清洗，减少 zero-width 隐形字符、homoglyph 同形字符、重复字符拉长和异常空白等字符级扰动影响
- 使用调优后的 TF-IDF 集成模型输出基础谣言概率
- 从训练集中检索相似样本作为证据，抽取轻量级 claim 结构特征，并将基础模型、检索证据和结构特征融合为最终判断
  
项目提供字符级合成扰动鲁棒性评估脚本，用于报告 clean accuracy、attacked accuracy、accuracy drop 和 prediction consistency。
同时，项目支持在本地模型完成决策后，调用学校提供的大语言模型接口生成自然语言解释；LLM 只负责解释，不参与标签决策。

### 环境配置

建议使用 Python 虚拟环境：

```bash
python -m venv .venv
```

Windows PowerShell 中启用虚拟环境：

```powershell
.\.venv\Scripts\Activate.ps1
```

安装依赖：

```bash
python -m pip install -r requirements.txt
```

或者使用 Makefile：

```bash
make install
```

当前本地模型只依赖 Python 标准库和 `numpy`。如果需要启用学校 LLM 解释功能，需要额外配置环境变量：

```bash
export SCHOOL_LLM_API_KEY="your-school-api-key"
export SCHOOL_LLM_BASE_URL="https://school-api.example/v1"
export SCHOOL_LLM_MODEL="school-model-name"
```

如果学校提供的是完整 chat-completions 接口，也可以直接设置：

```bash
export SCHOOL_LLM_API_URL="https://school-api.example/v1/chat/completions"
```

Windows PowerShell 中可以使用：

```powershell
$env:SCHOOL_LLM_API_KEY="your-school-api-key"
$env:SCHOOL_LLM_BASE_URL="https://school-api.example/v1"
$env:SCHOOL_LLM_MODEL="school-model-name"
```

### 学校 LLM 接入示例

本项目默认使用上海交通大学本地大模型 API 中的 `deepseek-chat` 模型作为 LLM 解释生成模型。LLM 只用于生成 `llm_evidence` 自然语言解释，不参与谣言/非谣言标签决策，也不会修改本地模型输出的最终标签。

但我们提供可选择的模型接口。`src/llm_explainer.py` 会从环境变量读取配置，并调用 OpenAI 兼容的 chat completions 接口。因此默认模型是 `SCHOOL_LLM_MODEL=deepseek-chat`，如果需要，也可以把该环境变量改成 SJTU API 支持的其他模型。

出于安全考虑，项目不会内置作者三人的 SJTU API key，避免 key 泄露后被刷爆。使用 LLM 解释功能时，请使用者自行申请并配置自己的 `SCHOOL_LLM_API_KEY`。如果不配置 API key，本地谣言检测模型仍可正常运行，只是 `llm_evidence` 会显示为不可用。

- `SCHOOL_LLM_API_KEY`：学校 API key；
- `SCHOOL_LLM_BASE_URL`：接口 base URL；
- `SCHOOL_LLM_API_URL`：完整 chat completions URL，可替代 `SCHOOL_LLM_BASE_URL`；
- `SCHOOL_LLM_MODEL`：实际调用的模型名；
- `SCHOOL_LLM_TIMEOUT`：可选，请求超时时间；
- `SCHOOL_LLM_TEMPERATURE`：可选，解释生成温度。

如果使用上海交通大学本地大模型 API 文档中的 OpenAI 兼容格式，可以按下面方式配置。根据文档示例，base URL 为 `https://models.sjtu.edu.cn/api/v1`，chat completions endpoint 为 `/chat/completions`，默认模型调用名为 `deepseek-chat`。

Linux/macOS:

```bash
export SCHOOL_LLM_API_KEY="your-sjtu-api-key"
export SCHOOL_LLM_BASE_URL="https://models.sjtu.edu.cn/api/v1"
export SCHOOL_LLM_MODEL="deepseek-chat"
```

Windows PowerShell:

```powershell
$env:SCHOOL_LLM_API_KEY="your-sjtu-api-key"
$env:SCHOOL_LLM_BASE_URL="https://models.sjtu.edu.cn/api/v1"
$env:SCHOOL_LLM_MODEL="deepseek-chat"
```

也可以直接配置完整接口地址：

```powershell
$env:SCHOOL_LLM_API_KEY="your-sjtu-api-key"
$env:SCHOOL_LLM_API_URL="https://models.sjtu.edu.cn/api/v1/chat/completions"
$env:SCHOOL_LLM_MODEL="deepseek-chat"
```

配置完成后，运行预测命令即可在结果中的 `llm_evidence` 字段看到 LLM 生成的解释：

```powershell
python predict.py --model models/main_fusion.pkl --train train.csv --text "sample text"
```

如果暂时不想调用 LLM，可以加上 `--no-llm`：

```powershell
python predict.py --model models/main_fusion.pkl --train train.csv --text "sample text" --no-llm
```

### 数据格式

`train.csv` 和 `val.csv` 需要包含以下字段：

```text
id,text,label,event
```

字段含义：

- `id`：样本唯一标识；
- `text`：待判断的短文本；
- `label`：分类标签，取值为 `0` 或 `1`；
- `event`：事件或来源分组，用于内部分层切分和分事件评估。

### 训练方法

默认训练命令：

```bash
python train.py --train train.csv --val val.csv --model models/main_fusion.pkl --metrics outputs/metrics.json
```

或者：

```bash
make train
```

根据组内经验来看，如果使用 Mac 电脑，训练时间大约在2~3分钟，若使用 Windows 则时间或许会偏长一些，大约在7~10分钟左右。
训练阶段只使用 `train.csv` 进行模型拟合和内部调参。程序会从 `train.csv` 中构造分层内部 dev 集，用于选择 TF-IDF 分支、集成权重、分类阈值和 evidence fusion 参数。`val.csv` 只用于最终验证指标汇报，不参与调参。

训练输出：

```text
models/main_fusion.pkl
outputs/metrics.json
```

模型文件中保存了选中的子模型、集成权重、基础阈值、证据融合权重、检索 top-k、相似度门限、证据感知阈值，以及最终 pipeline 使用的检索准确性保护规则。

### 评估方法

训练完成后运行：

```bash
python evaluate.py --model models/main_fusion.pkl --data val.csv --train train.csv --out-dir outputs
```

或者：

```bash
make evaluate
```

评估输出：

```text
outputs/evaluation.json
outputs/predictions.csv
outputs/correct_cases.csv
outputs/wrong_cases.csv
outputs/explain_cases.json
```

其中 `evaluation.json` 包含整体指标和按事件划分的指标，`predictions.csv` 保存逐样本预测结果，`explain_cases.json` 保存部分正确和错误样本的结构化解释证据。

### 鲁棒性评估

项目提供自动扰动评估脚本，用于从验证集派生字符级合成对抗样本，并报告 clean/attacked/drop/consistency 指标：

```bash
python robustness_eval.py --model models/main_fusion.pkl --data val.csv --train train.csv --out-dir outputs
```

或者：

```bash
make robustness
```

当前扰动类型包括：

```text
zero_width
homoglyph
repeat_chars
whitespace
casing
punctuation
```

这些扰动主要模拟隐形字符、同形字符替换、重复字符拉长、空白变化、大小写变化和标点噪声。由于这些扰动不改变原始语义，扰动样本沿用原验证样本标签。

输出文件：

```text
outputs/robustness_report.json
outputs/robustness_predictions.csv
```

`robustness_report.json` 包含 clean accuracy、attacked accuracy、accuracy drop 和 prediction consistency。`robustness_predictions.csv` 保存每条 clean 样本在各类扰动下的预测变化。

可以使用 `--limit` 做快速检查，或用 `--attacks` 指定部分扰动：

```bash
python robustness_eval.py --limit 50 --attacks zero_width,homoglyph
```

### 单条预测

训练完成并生成模型后，可以对单条文本进行预测：

```bash
python predict.py --model models/main_fusion.pkl --train train.csv --text "Swiss museum confirms it will take on #Gurlitt collection"
```

或者：

```bash
make predict TEXT="Swiss museum confirms it will take on #Gurlitt collection"
```

预测结果以 JSON 形式输出，主要包含：

```text
label
label_name
prob_rumor
confidence
threshold
baseline_label
baseline_prob_rumor
evidence
explanation
llm_evidence
```

如果想禁用 LLM 解释，可以使用：

```bash
python predict.py --model models/main_fusion.pkl --train train.csv --text "sample text" --no-llm
```

### Web UI

本项目提供操作较为简便、用户友好的网站服务，但仅仅是本地网站，较为简陋，操作方法如下
启动本地网页服务：

```bash
python web_app.py --model models/main_fusion.pkl --train train.csv
```

或者：

```bash
make web
```

然后在浏览器打开：

```text
http://127.0.0.1:8000
```

Web UI 使用与 `predict.py` 相同的共享预测服务，会展示本地模型判断、结构化证据，以及在学校 API 配置完成时展示 LLM 生成的解释。

### 测试方法

运行单元测试：

```bash
python -m unittest discover -s tests
```

或者：

```bash
make test
```

### 项目结构

```text
.
|-- train.py                 # 训练与调参入口
|-- evaluate.py              # 评估入口
|-- robustness_eval.py       # 字符级合成扰动鲁棒性评估入口
|-- predict.py               # 单条文本预测入口
|-- web_app.py               # 本地 Web UI 服务
|-- web/                     # 浏览器端页面资源
|-- Makefile                 # 常用命令封装
|-- README.md
|-- EXPERIENTS.md
|-- requirements.txt
|-- train.csv
|-- val.csv
|-- tests/
`-- src/
    |-- text_model.py        # 文本标准化、TF-IDF 模型、指标计算
    |-- ensemble_model.py    # TF-IDF 集成模型
    |-- defense.py           # 输入清洗与防御
    |-- retriever.py         # 相似训练样本检索
    |-- evidence.py          # 检索证据特征
    |-- claim_structure.py   # 轻量级 claim 结构特征
    |-- evidence_pipeline.py # 最终证据融合推理流水线
    |-- llm_explainer.py     # 学校 LLM 解释客户端
    `-- prediction_service.py # CLI 和 Web 共用预测服务
```

生成产物默认不纳入源码管理：

```text
models/
outputs/
```

## English Version

A simple model estabiled by zsf&amp;lzh&amp;zmx, just a sample designed by 3 college students from SJTU. Maybe we'll improve it in the long run.

Our goal is to build a rumor detector for short social-media style claims.

The final system is designed to combine text classification with retrieval evidence and lightweight claim-structure signals.

This repository contains one final, reproducible rumor detection pipeline for
tweet-level binary classification. The retained pipeline is evidence-aware and
uses a retrieval accuracy guard. Prediction entry points can also attach a
school-LLM explanation layer after the local model has made its decision:

1. normalize the input text;
2. score it with a tuned TF-IDF ensemble;
3. retrieve similar training cases as evidence;
4. extract lightweight claim-structure features;
5. fuse model, retrieval, and structure signals with tuned weights;
6. apply retrieval-based probability guards for high-confidence non-rumor
   evidence and borderline rumor rescue;
7. return the final label, probability, confidence, local evidence, and local
   explanation;
8. optionally ask the configured school LLM to rewrite the same evidence into
   a natural-language `llm_evidence` explanation without changing the model
   label.

Label `1` means `rumor`; label `0` means `non-rumor`.

## Installation

```bash
python -m pip install -r requirements.txt
```

or:

```bash
make install
```

The final local pipeline only requires Python and `numpy`. The optional school
LLM explanation uses Python standard-library HTTP calls and reads API settings
from environment variables.

## Data Format

`train.csv` and `val.csv` must contain these columns:

```text
id,text,label,event
```

`label` must be `0` or `1`. `event` is used for stratified internal splitting
and per-event diagnostics.

## Train

```bash
python train.py --train train.csv --val val.csv --model models/main_fusion.pkl --metrics outputs/metrics.json
```

or:

```bash
make train
```

Training uses `train.csv` only for model fitting and internal tuning. It creates
a stratified internal dev split from `train.csv`, tunes the ensemble threshold
and evidence-fusion parameters on that internal dev split, then reports
validation metrics on `val.csv`. `val.csv` is not used for tuning. After
internal selection, the selected TF-IDF branches are refit on the full
`train.csv`.

Outputs:

```text
models/main_fusion.pkl
outputs/metrics.json
```

The model artifact stores the selected sub-models, ensemble weights, base
threshold, evidence weights, evidence top-k, retrieval similarity gate, and
evidence-aware threshold. It also stores the retrieval accuracy guard used by
the final `accuracy_rescue_fusion` pipeline.

Evidence weights are selected on the internal dev split. If a signal does not
improve the dev objective, its tuned weight can be `0.0`; the prediction JSON
always exposes the effective `decision_factors` used for the final probability.
The retrieval accuracy guard can lower probability when similar training cases
strongly support `non-rumor`, or slightly raise a borderline probability when
retrieval evidence strongly supports `rumor`.

## Evaluate

```bash
python evaluate.py --model models/main_fusion.pkl --data val.csv --train train.csv --out-dir outputs
```

or:

```bash
make evaluate
```

Evaluation always uses the final evidence-aware pipeline and the saved
evidence-aware threshold.

Outputs:

```text
outputs/evaluation.json
outputs/predictions.csv
outputs/correct_cases.csv
outputs/wrong_cases.csv
outputs/explain_cases.json
```

`evaluation.json` contains overall and per-event metrics. `predictions.csv`
contains one row per evaluated sample. `explain_cases.json` keeps structured
evidence for selected correct and wrong cases.

## Robustness Evaluation

Run the synthetic character-level robustness benchmark:

```bash
python robustness_eval.py --model models/main_fusion.pkl --data val.csv --train train.csv --out-dir outputs
```

or:

```bash
make robustness
```

The benchmark derives perturbed samples from the validation set and keeps the
original labels because the attacks are intended to preserve semantics. Current
attack types are `zero_width`, `homoglyph`, `repeat_chars`, `whitespace`,
`casing`, and `punctuation`.

Outputs:

```text
outputs/robustness_report.json
outputs/robustness_predictions.csv
```

The report includes clean accuracy, attacked accuracy, accuracy drop, and
prediction consistency. Use `--limit` for quick checks and `--attacks` to run a
subset, for example:

```bash
python robustness_eval.py --limit 50 --attacks zero_width,homoglyph
```

This is a synthetic character-level perturbation benchmark, not a manually
curated adversarial dataset or full adversarial training.

## Predict

```bash
python predict.py --model models/main_fusion.pkl --train train.csv --text "Swiss museum confirms it will take on #Gurlitt collection"
```

or:

```bash
make predict TEXT="Swiss museum confirms it will take on #Gurlitt collection"
```

Prediction returns JSON with:

```text
label
label_name
prob_rumor
confidence
threshold
baseline_label
baseline_prob_rumor
evidence
explanation
llm_evidence
```

The `evidence` object includes keyword contributions, retrieved training cases,
retrieval statistics, claim-structure features, fusion decision factors, and
the optional probability guard result.

`llm_evidence` is generated by the school LLM after the local model has already
produced its label and evidence. It is enabled by default for `predict.py` and
the web UI. If the school API is not configured, the field still appears with
`available: false` and an error message; the local model result remains valid.

学校提供的大语言模型接口只作为解释润色模块。分类标签由本地可复现模型输出，
LLM 不参与标签决策；系统将原始推文、预测标签、置信度、模型证据词和
RAG 相似样本传入学校模型，由其生成自然语言判断依据。

School LLM configuration is read from environment variables:

For safety and cost control, this repository does not include the authors'
SJTU API key. Users must configure their own `SCHOOL_LLM_API_KEY` to enable LLM
evidence generation. Without an API key, the local model still works and
`llm_evidence` is returned as unavailable.

```bash
export SCHOOL_LLM_API_KEY="your-school-api-key"
export SCHOOL_LLM_BASE_URL="https://school-api.example/v1"
export SCHOOL_LLM_MODEL="school-model-name"
```

If the school provides the complete chat-completions endpoint, use:

```bash
export SCHOOL_LLM_API_URL="https://school-api.example/v1/chat/completions"
```

Disable LLM evidence for one terminal prediction with:

```bash
python predict.py --text "sample text" --no-llm
```

## Web UI

```bash
make web
```

Then open:

```text
http://127.0.0.1:8000
```

The web UI calls the same shared prediction service as `predict.py`: it shows
the local model decision, structured model evidence, and the default LLM
evidence explanation when the school API is configured.

## Tests

```bash
python -m unittest discover -s tests
```

or:

```bash
make test
```

## Project Structure

```text
.
|-- train.py                 # train and tune the final pipeline
|-- evaluate.py              # evaluate the final pipeline
|-- robustness_eval.py       # synthetic character-level robustness evaluation
|-- predict.py               # single-text prediction with model and LLM evidence
|-- web_app.py               # local web UI server
|-- web/                     # browser UI assets
|-- Makefile                 # main commands
|-- README.md
|-- EXPERIENTS.md
|-- requirements.txt
|-- train.csv
|-- val.csv
|-- tests/
`-- src/
    |-- text_model.py        # text normalization, TF-IDF model, metrics
    |-- ensemble_model.py    # TF-IDF ensemble
    |-- defense.py           # shared input sanitization
    |-- retriever.py         # training-case retrieval
    |-- evidence.py          # retrieval evidence features
    |-- claim_structure.py   # lightweight claim-structure features
    |-- evidence_pipeline.py # final evidence-aware inference pipeline
    |-- llm_explainer.py     # school LLM explanation client
    `-- prediction_service.py # shared CLI and web prediction service
```

Generated artifacts are intentionally not part of the source workflow:

```text
models/
outputs/
```
