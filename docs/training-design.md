# Training Designs

选择一小组固定的双 TF-IDF 配置，目前的默认训练流程会直接作用于内部开发划分的数据集上。

被选中的模型概率会通过保存好的决策阈值进行校准，后续的评估和预测将会可以复用同一套标签判定规则。

Default training entry: python train.py --train train.csv --val val.csv --model models/ensemble.pkl --metrics outputs/metrics.json.

README documents make train, make evaluate, make predict, and make test as the normal workflow.