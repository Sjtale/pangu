# Pangu-Weather 轻量化合规提交

本提交对盘古全量模型进行结构化剪枝、全 69 通道知识蒸馏和混合精度量化。
学生模型直接预测全部 69 个输出通道，不预测输入到真值的残差，不使用测试样本间相关性，
不使用训练后外置斜率、仿射或全局均值系数修正预测。

## 计时边界

提交保持组委会原始 `inference.py` 的固定计时边界：归一化、输入整理、最终 69 通道整理、
反归一化和结果导出位于模板计时区间外，计时区间仅包含模型前向及必要的 DCU 同步。
HIP 内核可在首次运行时生成并编译，不携带闭源二进制文件。

## 复现路径

- `train.py`：全量基准训练入口。
- `scripts/prune_structured.py`：从盘古权重生成结构化学生初始化。
- `distill_train.py`：全 69 通道学生蒸馏入口。
- `scripts/quantize_mixed_precision.py`：混合精度量化。
- `scripts/compact_fuser_alias_checkpoint.py`：无损去除重复别名权重。
- `scripts/convert_fp16.py`：FP16 检查点转换。

模型压缩包必须仅在根目录包含 `model_fp16.pth`。代码包可用
`scripts/audit_submission_package.py` 执行结构、禁用系数文件与哈希审计。
