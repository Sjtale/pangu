# Pangu-Weather

Pangu-Weather（盘古气象大模型）是华为云提出的首个精度超过传统数值预报的 AI 气象模型，基于 3D Earth-Specific Transformer 架构，速度比传统数值预报提升 10000 倍以上。

> 论文：[Accurate medium-range global weather forecasting with 3D neural networks](https://www.nature.com/articles/s41586-023-06185-3)

## 数据准备

```bash
source ../earth_env.sh
python ../AI4S_ERA5NetCDF_to_HDF5.py 
```
真实数据的存储格式参照 `../era5_dataset_prepare/README.md`，在 `conf/config.yaml` 中修改：

```yaml
stats_dir: 均值/标准差文件路径，用于归一化
static_dir: 静态文件路径（陆地掩码等），若模型不需要可忽略
data_dir: ERA5 数据根路径，年度 h5 文件存放于 data_dir/data/{year}.h5
train_time: [1977]          # 训练年份
val_time: [2005]            # 验证年份
test_time: [2012]           # 测试年份，后台测试中，该年份只是标识，并非真实年份, 提交代码时需修改为[2050, 2052, 2054, 2056, 2058] ！！！！
```
无真实数据时，可生成虚拟数据快速验证流程(若快速验证，则需将conf/config.yaml中max_epoch设为1)：

```bash
source ../earth_env.sh
python fake_data.py         # 伪造ERA5数据，只为验证代码正确值，使用时可忽略
```

## 运行
```bash
source ../earth_env.sh

# 1. 训练
python train.py                # AI4S，单卡，赛题为单卡
python plot_loss.py            # AI4S，查看loss下降曲线

# 2. 推理（结果输出至 ./result/output/）
python inference.py

# 3. 评估 & 可视化（result.py 末尾可指定日期和变量）
python result.py
```

## 结构化剪枝

服务器上官方 FP32 权重保存在 `../pangu_backups/model_bak.pth`，不放入
`data/checkpoints/`。脚本优先使用 `data/checkpoints/model_fp16.pth`，仅在该文件
不存在时回退到官方 FP32 备份。

方向5使用显式的通道依赖迁移，将浅层宽度从 192 剪到 160、
深层宽度从 384 剪到 320，同时剪除完整注意力头。生成剪枝权重：

```bash
python scripts/prune_structured.py
```

输出 `data/checkpoints/model_pruned_fp16.pth`。`inference.py` 检测到该文件后
默认使用剪枝模型；可用以下命令强制回退到全量 FP16 模型：

```bash
PANGU_USE_PRUNED=0 python inference.py
```

首次剪枝后必须先运行推理和 `result.py` 记录未微调精度。如需恢复精度，
使用独立的剪枝微调路径：

```bash
PANGU_TRAIN_PRUNED=1 python train.py
```

微调状态保存为 `model_pruned_train.pth`，最佳验证模型同步导出为
`model_pruned_fp16.pth`，不会覆盖官方 `model_bak.pth`。

## 集群训练，提前查看slurm作业提交方式和相关指令
```bash
mkdir -p logs
sbatch work_slurm.sh    # 提交前检查分区、节点数等配置
```

## 许可证
Apache 2.0，可免费用于学术研究和商业用途。
