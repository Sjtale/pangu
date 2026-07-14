# Pangu-Weather 轻量化合规提交

本提交对盘古全量模型进行结构化剪枝和全 69 通道知识蒸馏。
学生模型直接预测全部 69 个输出通道，不预测输入到真值的残差，不使用测试样本间相关性，
不使用训练后外置斜率、仿射或全局均值系数修正预测。

实际评分检查点 `model_fp16_alias_compact.pth` 包含 FP16 31.8988 MiB、
INT64 2.5312 MiB，没有 INT8 tensor。旧 `quantization` 字段不代表最终保留的 tensor
表示；尺寸审计以实际 dtype 和 storage 为准。

## 计时边界

提交保持组委会原始 `inference.py` 的固定计时边界：归一化、输入整理、最终 69 通道整理、
反归一化和结果导出位于模板计时区间外，计时区间仅包含模型前向及必要的 DCU 同步。
HIP 内核由随包源码在计时前或首次运行时编译，不携带共享库二进制文件。

## 复现路径

- `scripts/prune_structured.py`：从盘古权重生成固定 `pruned_96` 初始化。
- `distill_train.py`：固定 `pruned_96` 的全 69 通道知识蒸馏入口。
- `scripts/compact_fuser_alias_checkpoint.py`：无损去除重复别名权重。
- `scripts/convert_fp16.py`：保留别名的 FP16 检查点转换。
- `scripts/elide_deterministic_indices.py`：验证后省略可由固定结构重建的 position index。
- `scripts/compress_checkpoint_gzip.py`：可选的确定性无损压缩，仅在组委会确认格式后使用。
- `scripts/quantize_mixed_precision.py`：保留的后续实验工具，不是当前评分检查点的实际表示。
- `蒸馏与推理说明.md`：从剪枝、蒸馏到最终推理的完整命令。

模型压缩包必须仅在根目录包含 `model_fp16.pth`。代码包可用
`scripts/audit_submission_package.py` 执行结构、禁用系数文件与哈希审计。
