#!/bin/bash
# ============================================================
# Earth 模型公共环境配置
# 用法: 在各模型的 work_dcu.sh / work_slurm.sh 中 source ../env.sh
# ============================================================

echo "START TIME: $(date)"
module purge &>/dev/null || true

##### Load Conda & DTK #####
if module load sghpcdas/25.6 &>/dev/null; then
    echo "✅ HPC module sghpcdas/25.6 loaded"
else
    echo "⚠️  HPC module 'sghpcdas/25.6' not found, sourcing DTK & conda locally"
    # Source DTK (DCU Tool Kit) for ROCm/HIP runtime
    if [ -f /opt/dtk-25.04.4/env.sh ]; then
        source /opt/dtk-25.04.4/env.sh
        echo "✅ DTK 25.04.4 environment sourced"
    else
        echo "⚠️  DTK not found at /opt/dtk-25.04.4"
    fi
fi

conda init bash &>/dev/null
source ~/.bashrc &>/dev/null || true

if module load sghpc-mpi-gcc/26.3 &>/dev/null; then
    echo "✅ HPC module sghpc-mpi-gcc/26.3 loaded"
else
    echo "⚠️  HPC module 'sghpc-mpi-gcc/26.3' not found, falling back to conda openmpi"
fi

##### Activate env #####
conda activate onescience311         #AI4S, 注意变更为自己安装的conda环境
# source ../../onescience/.venv/bin/activate
source ../env.sh                     #AI4S

##### Verify env #####
echo "✅ Python: $(which python)"
which hipcc 2>/dev/null && echo "✅ hipcc found" || echo "⚠️  hipcc not found (AMD DCU/ROCm not available — GPU training disabled)"

##### Set DCU #####
export HIP_VISIBLE_DEVICES=0 2>/dev/null || true  # AI4S, 默认为0
