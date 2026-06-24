#!/bin/bash
set -e

# ==============================================================================
# 先导杯2026-AI4S 一键自动打包脚本
# ==============================================================================
# 运行前准备：
# 1. 请在 SCNet 平台将选好的最终模型打包成 zip 格式（必须是 zip，例如 model_fp16.pth）。
# 2. 生成下载链接，并将纯净的链接填写到 data/download_model_url.txt 中。
# 3. 运行本脚本：bash scripts/build_submission.sh
# ==============================================================================

echo "🚀 开始构建比赛提交包..."

SUBMIT_DIR="submit_package"
PANGU_DIR="$SUBMIT_DIR/pangu-weather"

# 1. 清理并创建干净的目录结构
rm -rf $SUBMIT_DIR
mkdir -p $PANGU_DIR/conf
mkdir -p $PANGU_DIR/data/checkpoints
mkdir -p $PANGU_DIR/result/output

# 2. 复制官方要求的必需文件
echo "📋 正在复制必要文件..."
cp conf/config.yaml $PANGU_DIR/conf/
cp train.py $PANGU_DIR/
cp inference.py $PANGU_DIR/
cp README.md $PANGU_DIR/

# 3. 检查 download_model_url.txt
if [ -f "data/download_model_url.txt" ]; then
    cp data/download_model_url.txt $PANGU_DIR/data/
else
    echo "⚠️ 警告：未找到 data/download_model_url.txt！将生成空文件，请记得填写！"
    touch $PANGU_DIR/data/download_model_url.txt
fi

# 4. 修改 config.yaml 满足官方后台测试的极度严苛要求
echo "🔧 正在注入官方评测专用的配置参数..."
CONFIG_FILE="$PANGU_DIR/conf/config.yaml"

# 强制将数据集路径改为本地 ERA5_test 相对路径
sed -i.bak -E 's|stats_dir:.*$|stats_dir: "../onedatasets/ERA5_test/stats/"|g' $CONFIG_FILE
sed -i.bak -E 's|static_dir:.*$|static_dir: "../onedatasets/ERA5_test/static/"|g' $CONFIG_FILE
sed -i.bak -E 's|data_dir:.*$|data_dir: "../onedatasets/ERA5_test/"|g' $CONFIG_FILE

# 强制修改年份配置为官方指定的值
sed -i.bak -E 's/train_ratio:.*$/train_ratio: [1977]/g' $CONFIG_FILE
sed -i.bak -E 's/val_ratio:.*$/val_ratio: [2005]/g' $CONFIG_FILE
sed -i.bak -E 's/test_ratio:.*$/test_ratio: [2050, 2052, 2054, 2056, 2058]/g' $CONFIG_FILE

# 恢复官方强制不可更改的路径与参数
sed -i.bak -E 's/checkpoint_dir:.*$/checkpoint_dir: ".\/data\/checkpoints"/g' $CONFIG_FILE
sed -i.bak -E 's/batch_size:.*$/batch_size: 1/g' $CONFIG_FILE
sed -i.bak -E 's/world_size:.*$/world_size: 1/g' $CONFIG_FILE

rm -f $CONFIG_FILE.bak

# 5. 打包为最终的 ZIP 文件
echo "📦 正在压缩最终提交包..."
cd $SUBMIT_DIR
zip -r pangu_weather.zip pangu-weather/ > /dev/null
cd ..

# 6. 验证包内容
echo "✅ 打包完成！提交包路径：$SUBMIT_DIR/pangu_weather.zip"
echo "🔍 提交包内部结构预览："
unzip -l $SUBMIT_DIR/pangu_weather.zip | grep -v "/$"

echo ""
echo "🎉 [最终检查清单]"
echo "1. 确保 download_model_url.txt 里的链接没有多余的空格。"
echo "2. 确保 SCNet 上的模型压缩包是 .zip 格式，且解压后不要带一层冗余文件夹。"
echo "3. 提交 pangu_weather.zip 给官方平台，外加一份你的优化文档 PDF！"
