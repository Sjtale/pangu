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

## 集群训练，提前查看slurm作业提交方式和相关指令
```bash
mkdir -p logs
sbatch work_slurm.sh    # 提交前检查分区、节点数等配置
```

## 许可证
Apache 2.0，可免费用于学术研究和商业用途。