## 环境准备
代码使用 `python3.9` 测试。

```bash
pip install -r requirements.txt
```

## 训练

```bash
python train.py
```

训练完的模型会保存在：`models/cartpole-v0-brand-new-baseline.pt`，可在 `train.py` 中修改保存位置。

## 查看效果

```bash
python eval.py [source model path (default: ./cartpole-vo-gamma-95.pt)]
```
