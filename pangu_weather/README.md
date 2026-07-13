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

### 无损别名去重与 U/V 隔离验收

如果先不改结构、不做蒸馏，可对当前完整 `pgw_lite_pruned_96`
一次性执行存储、CPU 加载、官方计时边界、常驻/峰值显存和逐阶段
耗时归因：

```bash
PANGU_DIAG_CHECKPOINT=model_fp16.pth \
  bash scripts/run_pruned96_uv_diagnosis.sh all
```

该入口会拒绝 A80、浅层 S96 或其他非 `[2,6,6,2]` 深度的 checkpoint，
并生成 `logs/pruned96_uv_bottleneck_report.md`。逐阶段计时会在每个阶段后同步
DCU，只用于归因；V 仍以未插桩的 `inference.py` 官方计时边界为准，
U 仍以平台结果为准。

先审计并生成只保留 OneFuser 最终写入者的 checkpoint：

```bash
python scripts/compact_fuser_alias_checkpoint.py \
  --source data/checkpoints/model_fp16.pth \
  --output data/checkpoints/model_fp16_alias_compact.pth
```

脚本会输出源文件和候选文件的 SHA256，并在原子替换前校验所有保留
tensor 的键顺序、shape、dtype 和数值。不要在候选包中同时放入原 checkpoint。

直接 mask slice 和 CUDA/HIP Graph 必须分别做严格 off/on A/B：

```bash
python scripts/probe_uv_runtime_sweep.py --preset direct-mask --repeat 5
python scripts/probe_uv_runtime_sweep.py --preset cuda-graph --repeat 5
python scripts/rank_uv_candidates.py logs/uv_runtime_sweep_<timestamp>.jsonl
```

记录包含稳态延迟的均值、P50、P90、标准差，以及 allocated/reserved/
peak 显存和输出误差。Graph 只在延迟至少降低 6%、显存增量不超过
10 MiB 且输出完全一致时晋级；direct-mask 只在延迟至少降低 3% 且
输出完全一致时晋级。两个开关均默认关闭，不影响已验证提交包。

当前主线冻结所有新学生结构，只优化完整 `[2,6,6,2]` pruned_96。
buffer-intern 通过 5 次 DCU 重复验证后作为独立平台候选默认开启；
HIP、stage-wise、Graph 和 compact-mask 仍默认关闭。诊断时按以下命令隔离测量：

```bash
# 三项 HIP 控制分别 off/on，再测试三项组合
python scripts/probe_uv_runtime_sweep.py --preset hip --repeat 5 --max-batches 5

# FP16 mask/position-index 只读 storage interning
python scripts/probe_uv_runtime_sweep.py --preset buffer-intern --repeat 5 --max-batches 5

# layer1/4 固定 guardrail，仅筛 layer2/3 的三档分块
python scripts/probe_uv_runtime_sweep.py --preset stagewise --repeat 5 --max-batches 5

# 只盘点真正的 fused backend；不会重新运行已经失败的 SDPA fallback
python scripts/probe_fast_attention_capability.py \
  --output logs/fast_attention_capability.json

# 能力盘点发现 Flash 后，强制 Flash backend 验证 learned bias + shifted mask；
# 不允许回退到 math/memory-efficient SDPA
python scripts/probe_fast_attention_compatibility.py \
  --output logs/fast_attention_compatibility.json
```

兼容性报告只有在 `adapter_candidate=true` 且
`decision=PROFILE_FOR_FUSED_KERNEL` 时才进入 `hipprof` 和完整模型 A/B。
`STOP_PYTORCH_FLASH_AND_TEST_HIP_STAGEWISE` 表示组合 mask 无法使用强制
Flash 路径，应停止该路线，不得把无 mask 的基础通过误认为 Pangu 兼容。

`PANGU_ATTN_CHUNK_SIZE_LAYER{1..4}` 和
`PANGU_MLP_CHUNK_SIZE_LAYER{1..4}` 的值 `0` 表示整 stage；对应的
`PANGU_CHUNKED_QKV_LAYER{1..4}`、`PANGU_CHUNKED_PROJ_LAYER{1..4}`
控制 QKV/projection 是否继续分块。未经 DCU 和平台验收不得改变默认值。

冻结已验证 89.6297 guardrail，并审计最小提交包：

```bash
python scripts/freeze_pruned96_guardrail.py \
  --submission-zip submit_package/pangu_weather.zip \
  --checkpoint data/checkpoints/model_fp16_alias_compact.pth \
  --calibration data/checkpoints/calibration_coeffs.npy
python scripts/audit_submission_package.py \
  submit_package/pangu_weather.zip \
  --model data/checkpoints/model_fp16_alias_compact.pth
```

冻结脚本拒绝覆盖已有目录；提交包审计只允许运行所需的 7 个文件。

Graph 路径默认使用 `PANGU_GRAPH_DIRECT_INPUT=1`：Graph 捕获直接接管
example input，推理数据在计时前直接写入 Graph 固定输入。这避免捕获
和 replay 时各额外保留一整套约 150 MiB 的 GPU 输入。如需回归旧 Graph
输入路径，显式设置 `PANGU_GRAPH_DIRECT_INPUT=0`。

### 计时前无损解压 checkpoint

可对 alias-compacted checkpoint 再做 gzip 无损压缩。`inference.py` 会在官方
计时前直接在 CPU 内存中恢复，不生成解压后的磁盘权重：

```bash
python scripts/compress_checkpoint_gzip.py \
  --source data/checkpoints/model_fp16_alias_compact.pth \
  --output data/checkpoints/model_fp16_alias_compact.pth.gz
PANGU_CHECKPOINT=model_fp16_alias_compact.pth.gz python inference.py
```

脚本会验证解压后 SHA256 与源文件完全一致。提交前必须确认平台
按磁盘上的 gzip 字节数计 U，并复核 U/V/W。

### Groupwise INT4 存储、FP16 运行

如果无损 gzip 收益不足，可将当前 INT8 Linear 权重以 groupwise INT4
存储，并在计时前恢复为 FP16 模型：

```bash
python scripts/pack_groupwise_int4.py \
  --source data/checkpoints/model_fp16_alias_compact.pth \
  --output data/checkpoints/model_int4_group64.pth \
  --group-size 64
PANGU_CHECKPOINT=model_int4_group64.pth python inference.py
python result.py
```

该方案不改变 timed forward 的计算 dtype，但 INT4 重量化不是无损的。候选
必须先检查 15 个评分通道，再根据平台 `ΔU+ΔW` 决定是否晋级。

### Recovery 直接流式写入 CPU

`PANGU_CPU_RECOVERY_OUTPUT=1` 使宽度 chunk 恢复结果直接写入 pinned CPU
输出，避免在 GPU 上构造完整约 143 MiB 的 69 通道输出。该拷贝发生在
timed forward 内，必须做 U/V A/B：

```bash
python scripts/probe_uv_runtime_sweep.py --preset cpu-recovery --repeat 5
```

该路径 DCU 已测得峰值不变、稳态延迟恶化约 35.3%，保留仅用于回归，
不得重新作为提交候选。

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

### 从官方模型筛选 U/V 学生结构

> 当前战略状态：冻结。只有完整 pruned_96 的 HIP、buffer interning、
> stage-wise 分块、真实 fast-attention 和无损存储路线均到顶后，才允许
> 重新设计零随机初始化学生；不得直接恢复旧 S96/A80 训练任务。

`run_official_uv_screen.sh` 支持 `A, S96, B, E, C, D` 六个学生。
`S96` 从已训练 Width-96 checkpoint 精确选择深度；`A` 从同一
Width-96 checkpoint 做全局一致的通道、完整 attention head 和深度剪枝。
其他边界候选仍从官方 192 维 `model_bak.pth` 初始化。`S96` 是
patch `[2,8,8]`、width 96、heads `[3,6,6,3]`、depth `[1,2,2,1]`；
`A` 是同 patch/depth 的 width-80 候选。两者都要求未量化的
`model_pgw_lite_pruned_96_fp16.pth` 作为结构初始化源。

```bash
# 1. 生成未训练 FP16 结构权重
bash scripts/run_official_uv_screen.sh A prepare

# S96-Shallow 使用同一流程
bash scripts/run_official_uv_screen.sh S96 prepare
bash scripts/run_official_uv_screen.sh S96 probe
bash scripts/run_official_uv_screen.sh S96 train

# 2. 与当前提交基线做 5×4 批次 U/V A/B（至少15个稳态时间点）
bash scripts/run_official_uv_screen.sh A probe

# 3. 仅当显存下降≥15%、延迟下降≥8%或预估 ΔU+ΔV≥0.6 时执行
bash scripts/run_official_uv_screen.sh A train

# 可选第三个参数指定完整训练 epoch 数；修复前的 checkpoint 使用新前缀重训
PANGU_UV_SCREEN_PREFIX=uv_screen_a_lrfixed \
  bash scripts/run_official_uv_screen.sh A train 8

# 训练完成后做无损 OneFuser 别名去重，再对最终权重做 U/V 对比
PANGU_UV_SCREEN_PREFIX=uv_screen_a_pgw96 \
  bash scripts/run_official_uv_screen.sh A pack
PANGU_UV_SCREEN_PREFIX=uv_screen_a_pgw96 \
  bash scripts/run_official_uv_screen.sh A probe-packed
```

`train` 默认运行 1 epoch，也可通过第三个正整数参数指定多个
epoch；每个 epoch 固定 2048 steps。Warmup 只在完整协议的前 256 steps
执行一次，cosine 调度跨越全部 `epochs × 2048` steps，不在 epoch 边界
重启。训练使用
`0.45 RMSE + 0.20 ACC + 0.25 评分通道教师 + 0.10 非评分通道教师`，
并明确关闭 hint loss、量化投影和架构升级。训练每 256 个优化 step
原子更新 `${prefix}_latest.pth`；作业中断后重新执行同一 `train` 命令会恢复
模型、优化器和 scheduler，并补足当前 epoch 剩余的 step。Fresh run
拒绝覆盖任何同前缀产物；有 latest checkpoint 的 resume 可在验证改善时
原子更新 `${prefix}_train.pth` / `${prefix}_fp16.pth`。Checkpoint 会核对 profile、
epoch/step 计划、warmup、学习率和 loss 权重；修复前无此元数据的候选不允许续训。

年份分块采样仅保留为显式 I/O 诊断开关，正式架构筛选入口不强制启用。
日志中 `rolling20` 是最近 20 步真实耗时，
`data_wait` 是其中等待 DataLoader 的时间，`cumulative` 仅供长程参考。

## 集群训练，提前查看slurm作业提交方式和相关指令
```bash
mkdir -p logs
sbatch work_slurm.sh    # 提交前检查分区、节点数等配置
```

## 许可证
Apache 2.0，可免费用于学术研究和商业用途。
