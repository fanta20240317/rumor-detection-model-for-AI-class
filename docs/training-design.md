# Training Designs

选择一小组固定的双 TF-IDF 配置，目前的默认训练流程先在内部开发划分上选择模型和参数，再用完整 `train.csv` 复训最终模型。

被选中的模型概率会通过保存好的决策阈值进行校准，后续的评估和预测将会可以复用同一套标签判定规则。

最终模型还会保存 retrieval accuracy guard，用于在预测时根据高置信检索证据调整边界样本概率。

Default training entry: python train.py --train train.csv --val val.csv --model models/main_fusion.pkl --metrics outputs/metrics.json.

README documents make train, make evaluate, make predict, and make test as the normal workflow.
