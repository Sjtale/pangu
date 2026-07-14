#!/usr/bin/env bash
set -euo pipefail

# ==============================================================================
# 先导杯2026-AI4S 一键自动打包脚本
# ==============================================================================
# 运行前准备：
# 1. 将 model_fp16_alias_compact.pth 重命名为 model_fp16.pth，再打包成 zip。
# 2. 生成下载链接，并将纯净的链接填写到 data/download_model_url.txt 中。
# 3. 可从任意目录运行：bash /path/to/pangu_weather/scripts/build_submission.sh
# ==============================================================================

echo "🚀 开始构建比赛提交包..."

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "$SCRIPT_DIR/.." && pwd)"
# Use an alternate directory for candidate builds so the verified baseline ZIP
# remains byte-for-byte available for rollback.
SUBMIT_DIR="${PANGU_SUBMIT_DIR:-$PROJECT_DIR/submit_package}"
# The script clears its output tree. Resolve and restrict custom destinations
# before any deletion: the normal in-project submit_package is allowed, while
# alternate candidate builds must be explicitly named pangu_* under a system
# temporary directory.
SUBMIT_DIR="$(python - "$PROJECT_DIR" "$SUBMIT_DIR" <<'PY'
from pathlib import Path
import sys
import tempfile

project_dir = Path(sys.argv[1]).resolve()
candidate = Path(sys.argv[2]).expanduser().resolve()
default = (project_dir / "submit_package").resolve()
temporary_roots = {
    Path(tempfile.gettempdir()).resolve(),
    Path("/tmp").resolve(),
    Path("/private/tmp").resolve(),
}
temporary_candidate = candidate.name.startswith("pangu_") and any(
    candidate != root and candidate.is_relative_to(root)
    for root in temporary_roots
)
if candidate != default and not temporary_candidate:
    raise SystemExit(
        "Refusing unsafe PANGU_SUBMIT_DIR; use the default submit_package or "
        "a pangu_* directory under the system temporary directory"
    )
print(candidate)
PY
)"
PANGU_DIR="$SUBMIT_DIR/pangu_weather"
MODEL_URL_FILE="$PROJECT_DIR/data/download_model_url.txt"
ZIP_FILE="$SUBMIT_DIR/pangu_weather.zip"
DISTILL_DOC="蒸馏与推理说明.md"
TEAM_PDF="侍奉部_说明文档.pdf"

ROOT_FILES=(
    compliant_inference_wrapper.py
    distill_train.py
    hip_earth_attention_tiled.py
    hip_runtime_controls.py
    inference.py
    p2_tiled_attention.py
    pangu_profile_model.py
    result.py
    score_training_utils.py
    selective_mlp96.py
    train.py
)

REPRO_SCRIPTS=(
    scripts/audit_submission_package.py
    scripts/compact_fuser_alias_checkpoint.py
    scripts/compress_checkpoint_gzip.py
    scripts/convert_fp16.py
    scripts/elide_deterministic_indices.py
    scripts/prune_structured.py
    scripts/quantize_mixed_precision.py
)

# 在删除上一个提交包前检查全部输入，避免因配置错误损失已验证包。
for relative_path in \
    conf/config.yaml \
    COMPLIANCE_README.md \
    "$DISTILL_DOC" \
    "$TEAM_PDF" \
    data/download_model_url.txt \
    hip_kernels/earth_attention_tiled_fwd.hip \
    scripts/audit_submission_package.py \
    "${REPRO_SCRIPTS[@]}" \
    "${ROOT_FILES[@]}"; do
    if [[ ! -f "$PROJECT_DIR/$relative_path" ]]; then
        echo "❌ 缺少打包必需文件：$PROJECT_DIR/$relative_path" >&2
        exit 1
    fi
done

# 1. 清理并创建干净的目录结构
rm -rf "$SUBMIT_DIR"
mkdir -p "$PANGU_DIR/conf"
mkdir -p "$PANGU_DIR/data"
mkdir -p "$PANGU_DIR/hip_kernels"
mkdir -p "$PANGU_DIR/result/output"
mkdir -p "$PANGU_DIR/scripts"

# 2. 复制官方入口、P2 运行时及在线 HIP 编译源码
echo "📋 正在复制必要文件..."
cp "$PROJECT_DIR/conf/config.yaml" "$PANGU_DIR/conf/"
cp "$PROJECT_DIR/hip_kernels/earth_attention_tiled_fwd.hip" "$PANGU_DIR/hip_kernels/"
for filename in "${ROOT_FILES[@]}"; do
    cp "$PROJECT_DIR/$filename" "$PANGU_DIR/"
done
cp "$MODEL_URL_FILE" "$PANGU_DIR/data/"
cp "$PROJECT_DIR/COMPLIANCE_README.md" "$PANGU_DIR/README.md"
cp "$PROJECT_DIR/$DISTILL_DOC" "$PANGU_DIR/"
cp "$PROJECT_DIR/$TEAM_PDF" "$PANGU_DIR/"
for relative_path in "${REPRO_SCRIPTS[@]}"; do
    cp "$PROJECT_DIR/$relative_path" "$PANGU_DIR/$relative_path"
done

# 3. 修改 config.yaml 满足官方后台测试参数
echo "🔧 正在注入官方评测专用的配置参数..."
CONFIG_FILE="$PANGU_DIR/conf/config.yaml"

sed -i.bak -E 's|stats_dir:.*$|stats_dir: "../onedatasets/ERA5_test/stats/"     # AI4S，路径不可更改, 拉取本地ERA5数据|g' "$CONFIG_FILE"
sed -i.bak -E 's|static_dir:.*$|static_dir: "../onedatasets/ERA5_test/static/"   # AI4S，路径不可更改, 拉取本地本地ERA5数据|g' "$CONFIG_FILE"
sed -i.bak -E 's|data_dir:.*$|data_dir: "../onedatasets/ERA5_test/"            # AI4S，路径不可更改 , 拉取本地本地ERA5数据|g' "$CONFIG_FILE"
sed -i.bak -E 's/train_ratio:.*$/train_ratio: [1977]/g' "$CONFIG_FILE"
sed -i.bak -E 's/val_ratio:.*$/val_ratio: [2005]/g' "$CONFIG_FILE"
sed -i.bak -E 's/test_ratio:.*$/test_ratio: [2050, 2052, 2054, 2056, 2058]/g' "$CONFIG_FILE"
sed -i.bak -E 's/checkpoint_dir:.*$/checkpoint_dir: ".\/data\/checkpoints"/g' "$CONFIG_FILE"
sed -i.bak -E 's/batch_size:.*$/batch_size: 1/g' "$CONFIG_FILE"
sed -i.bak -E 's/world_size:.*$/world_size: 1/g' "$CONFIG_FILE"
rm -f "$CONFIG_FILE.bak"

# 4. 打包并使用精确成员路径白名单审计
echo "📦 正在压缩最终提交包..."
python - "$SUBMIT_DIR" <<'PY'
from pathlib import Path
import sys
import zipfile

submit_dir = Path(sys.argv[1])
source_dir = submit_dir / "pangu_weather"
target = submit_dir / "pangu_weather.zip"
with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED) as archive:
    for path in sorted(source_dir.rglob("*")):
        member = path.relative_to(submit_dir).as_posix()
        if path.is_dir():
            archive.writestr(member.rstrip("/") + "/", b"")
        else:
            archive.write(path, member)
PY

echo "🔍 正在执行精确包结构审计..."
python "$SCRIPT_DIR/audit_submission_package.py" "$ZIP_FILE" >/dev/null
echo "✅ 打包完成！提交包路径：$ZIP_FILE"
echo "🔍 提交包内部结构预览："
python - "$ZIP_FILE" <<'PY'
import sys
import zipfile

with zipfile.ZipFile(sys.argv[1]) as archive:
    for item in archive.infolist():
        if not item.is_dir():
            print(f"{item.file_size:10d}  {item.filename}")
PY
if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$ZIP_FILE" > "$ZIP_FILE.sha256"
else
    shasum -a 256 "$ZIP_FILE" > "$ZIP_FILE.sha256"
fi

echo ""
echo "🎉 [最终检查清单]"
echo "1. 确保 SCNet 模型 ZIP 解压后直接得到 model_fp16.pth。"
echo "2. 不要同时放入 model_fp16_alias_compact.pth 和 model_fp16.pth。"
echo "3. 提交 pangu_weather.zip 给官方平台；包内已含 Markdown 和侍奉部说明文档 PDF。"
