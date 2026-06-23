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

## FP16 权重审计与重打包

`scripts/convert_fp16.py` 会剥离优化器等非推理字段，并在转换 FP16 时保留
state dict 的共享 storage 和 tensor view。默认从官方 FP32 权重生成
`data/checkpoints/model_fp16.pth`：

```bash
python scripts/convert_fp16.py \
  --report data/checkpoints/model_fp16_audit.json
```

审计服务器上已有的 FP16 权重而不改写文件：

```bash
python scripts/convert_fp16.py \
  --source data/checkpoints/model_fp16.pth \
  --audit-only
```

脚本使用临时文件写入，并在替换目标文件前验证所有键、dtype、形状、数值和
storage 别名关系。首次在服务器运行时可通过 `--output` 生成候选文件，完成
严格加载、推理和 `result.py` 对比后再替换默认权重。

候选文件可直接用于 A/B 推理，无需覆盖当前权重：

```bash
PANGU_FP16_CHECKPOINT=model_fp16_compact.pth python inference.py
python result.py
```

## 结构化剪枝

服务器上官方 FP32 权重保存在 `./pangu_backups/model_bak.pth`，不放入
`data/checkpoints/`。脚本优先使用 `data/checkpoints/model_fp16.pth`，仅在该文件
不存在时回退到官方 FP32 备份。
`pangu_backups/` 仅供服务器调试，已加入 `.gitignore`；生成提交压缩包时必须排除，
否则会大幅增加模型大小。

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

## 知识蒸馏

当剪枝模型仅用真实标签微调仍无法恢复精度时，用官方全量模型作为
冻结教师，对 `160/320` 宽度的剪枝学生做输出蒸馏：

```bash
python scripts/prune_structured.py
python distill_train.py
```

最佳学生状态保存为 `model_distilled_train.pth`，提交用 FP16 权重保存为
`model_distilled_fp16.pth`。训练时教师不更新梯度，验证集仍使用真实标签选择
最佳 checkpoint。使用蒸馏学生推理：

```bash
PANGU_USE_DISTILLED=1 python inference.py
```

正式蒸馏默认使用官方外部 ERA5 的 `1980-1985` 年训练集和 `1986` 年
验证集。由于每个训练样本都增加一次 Teacher 前向，每个 epoch 默认随机使用
最多 2048 个样本，通过多轮 shuffle 覆盖六年数据。`ERA5_test` 配置仅保留在
`conf/config.yaml` 注释中供本地烟雾测试，不用它评估蒸馏收益。

## 集群训练，提前查看slurm作业提交方式和相关指令
```bash
mkdir -p logs
sbatch work_slurm.sh    # 提交前检查分区、节点数等配置
```

## 许可证
Apache 2.0，可免费用于学术研究和商业用途。
