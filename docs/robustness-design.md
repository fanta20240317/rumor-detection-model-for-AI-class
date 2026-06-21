# 鲁棒性设计

当前项目在输入层面加入了非常非常基础的安全性和鲁棒性处理，主要逻辑位于 `src/defense.py`。该模块会在特征抽取和证据融合之前对输入文本进行标准化，减少常见字符级扰动对模型判断的影响。

目前已覆盖的输入清洗包括：

- zero-width 隐形字符清理
- homoglyph 同形字符映射，例如用西里尔/希腊字母伪装成拉丁字母
- repeated-character 拉长字符折叠，例如把过长重复字符压缩
- whitespace 异常空白合并
- Unicode NFKC 规范化

这些清洗逻辑已经集成到 evidence-aware prediction pipeline 中，因此命令行预测、评估脚本和本地 Web UI 都会使用同一套输入防御逻辑。

单元测试覆盖了可疑字符标准化，以及常见扰动下的预测稳定性检查。

## 合成扰动评估

`robustness_eval.py` 提供了一个从验证集自动派生的字符级合成对抗评估。它不需要额外人工标注，因为这些扰动的目标是保持原始语义不变，所以扰动样本沿用原验证样本标签。

当前实现的扰动类型包括：

- `zero_width`：插入 zero-width 隐形字符；
- `homoglyph`：替换为视觉相似的混淆字符；
- `repeat_chars`：拉长重复字符；
- `whitespace`：改变空白分布；
- `casing`：改变大小写模式；
- `punctuation`：添加标点噪声。

## 评估指标

脚本会报告以下鲁棒性指标：

- clean accuracy：原始验证集准确率
- attacked accuracy：扰动样本准确率
- accuracy drop：`clean accuracy - attacked accuracy`
- prediction consistency：原始样本和扰动样本预测标签一致的比例

输出文件：

```text
outputs/robustness_report.json
outputs/robustness_predictions.csv
```

`robustness_report.json` 保存总体指标、各扰动类型指标和按事件划分的 consistency。`robustness_predictions.csv` 保存每条样本在各类扰动下的预测变化。

## 使用方法

完整运行：

```bash
python robustness_eval.py --model models/main_fusion.pkl --data val.csv --train train.csv --out-dir outputs
```

或者：

```bash
make robustness
```

快速检查可以限制样本数量：

```bash
python robustness_eval.py --limit 50
```

也可以只运行部分扰动类型：

```bash
python robustness_eval.py --attacks zero_width,homoglyph
```

## 边界说明

该评估是字符级合成扰动 benchmark，可以用于说明项目考虑了输入安全性和基础对抗鲁棒性。但它不是人工标注的真实对抗样本集，也不是完整 adversarial training，依旧非常非常基础。
