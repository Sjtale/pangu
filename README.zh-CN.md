# 先导杯 AI4S — Pangu-Weather 轻量化

*[English](README.md) · 简体中文*

面向先导杯 AI4S 赛题的盘古气象大模型（Pangu-Weather）轻量化与推理加速工程。
在保持组委会原始 `inference.py` 计时边界的前提下，通过结构化剪枝、全 69 通道知识蒸馏、
混合精度量化与 DCU/HIP 算子优化，压缩模型体积并降低单次预报延迟。

## 目录结构

```
.
├── AI4S_ERA5NetCDF_to_HDF5.py   # ERA5 NetCDF → 训练用 HDF5 转换
├── env.sh / earth_env.sh        # 数据集、模型路径与 DTK/Conda 环境
├── export_DDP_vars.sh           # 分布式训练环境变量
├── onedatasets/                 # ERA5 数据与归一化统计量（大文件不入库）
├── onemodels/                   # OneScience 模型目录
└── pangu_weather/               # 主工程：训练、蒸馏、量化、推理
```

`pangu_weather/` 内主要入口：

| 路径 | 说明 |
| --- | --- |
| `train.py` | 全量基准训练 |
| `scripts/prune_structured.py` | 从盘古权重生成 `pruned_96` 结构化学生初始化 |
| `distill_train.py` | 全 69 通道知识蒸馏 |
| `scripts/quantize_mixed_precision.py` | 混合精度量化（正式配置 `--keep-count 67`） |
| `scripts/compact_fuser_alias_checkpoint.py` | 无损去除 OneFuser 重复别名权重 |
| `inference.py` | 推理入口，保持模板计时边界 |
| `result.py` | 由推理输出计算 RMSE / ACC |
| `scripts/build_submission.sh` | 生成并审计提交包 |

## 快速开始

```bash
source env.sh
cd pangu_weather

# 冒烟测试：加载最终检查点跑 1 个 batch
HDF5_USE_FILE_LOCKING=FALSE \
PANGU_CHECKPOINT=model_fp16_alias_compact.pth \
PANGU_MAX_INFERENCE_BATCHES=1 \
python inference.py

# 完整推理并计算指标
HDF5_USE_FILE_LOCKING=FALSE \
PANGU_CHECKPOINT=model_fp16_alias_compact.pth \
python inference.py
python result.py
```

完整的剪枝 → 蒸馏 → 量化 → 推理命令链见
[`pangu_weather/蒸馏与推理说明.md`](pangu_weather/蒸馏与推理说明.md)。

## 测试

在仓库根目录执行（保证 `pangu_weather` 可作为包导入）：

```bash
python -m unittest discover -s pangu_weather/tests -p 'test_*.py'
```

`pangu_weather/tests/` 为基于 `unittest` 的静态与轻量运行时契约测试，不依赖 DCU；
`pangu_weather/module_test_scripts/` 中的用例需要实际 DCU 运行时环境。

## 合规说明

学生模型直接预测全部 69 个输出通道，不预测输入到真值的残差，不使用测试样本间相关性，
不使用训练后外置斜率、仿射或全局均值系数修正预测。计时区间仅包含模型前向及必要的 DCU
同步；HIP 内核随包以源码分发，首次运行时由 `hipcc` 编译，不携带闭源二进制文件。
详见 [`pangu_weather/COMPLIANCE_README.md`](pangu_weather/COMPLIANCE_README.md)。
