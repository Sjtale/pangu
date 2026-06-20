import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import glob
import h5py
from datetime import datetime
from tqdm import tqdm
from onescience.utils.fcn.YParams import YParams
from matplotlib import rcParams

# rcParams['font.family'] = 'serif'
# rcParams['font.serif'] = ['DejaVu Serif']
rcParams['mathtext.fontset'] = 'stix'
rcParams['axes.linewidth'] = 0.9
rcParams['xtick.major.width'] = 0.9
rcParams['ytick.major.width'] = 0.9

def get_metadata(data_dir, test_years, channels):
    """Multi-file HDF5 format: data/{year}/*.h5"""
    h5_files = []
    for year in test_years:
        year_files = sorted(glob.glob(os.path.join(data_dir, "data", str(year), "*.h5")))
        h5_files.extend(year_files)

    if not h5_files:
        raise FileNotFoundError(f"No HDF5 files found in {data_dir}/data/{{year}}/")
    channel_indices = list(range(len(channels)))

    total_files = [f for f in os.listdir('./result/output/') if f.endswith('.npy')]
    total_files.sort()

    # Build file map: YYYYMMDDHH → HDF5 path
    h5_map = {}
    for h5f in h5_files:
        basename = os.path.basename(h5f).replace('.h5', '')
        h5_map[basename] = h5f
    return total_files, channel_indices, h5_map

def get_result(total_files, channel_indices, h5_map, clim_mean):
    channel_rmse = np.zeros(len(channel_indices))
    channel_acc = np.zeros(len(channel_indices))
    clim_mean = clim_mean[0, :, :, :]
    numerator = np.zeros(len(channel_indices))
    pred_sq_sum = np.zeros(len(channel_indices))
    label_sq_sum = np.zeros(len(channel_indices))
    for file in tqdm(total_files, unit="files"):
        fname = file[:-4]  # 去掉 .npy
        h5_path = h5_map[fname]
        with h5py.File(h5_path, "r") as f:
            label = f["fields"][:].squeeze()  # [C, H, W]
            label = label[channel_indices]
        pred = np.load(f'result/output/{file}').squeeze()

        label_anom = label - clim_mean
        pred_anom = pred - clim_mean
        # 累加
        numerator += np.sum(pred_anom * label_anom, axis=(1, 2))
        pred_sq_sum += np.sum(pred_anom ** 2, axis=(1, 2))
        label_sq_sum += np.sum(label_anom ** 2, axis=(1, 2))

        channel_rmse += np.sqrt(np.mean((label - pred) ** 2, axis=(1, 2)))
    channel_rmse /= len(total_files)
    channel_acc = numerator / (np.sqrt(pred_sq_sum * label_sq_sum) + 1e-8)
    np.save('./result/acc.npy', channel_acc)
    np.save('./result/rmse.npy', channel_rmse)


def show_result():
    channel_rmse = np.load('./result/rmse.npy')
    channel_acc = np.load('./result/acc.npy')

    channels = [cfg_data.dataset.channels[i] for i in range(len(channel_indices))]
    w = 24  # 最长 channel 名宽度

    # 表头
    print(f"┌{'─' * (w + 2)}┬{'─' * 14}┬{'─' * 14}┐")
    print(f"│ {'Channel':<{w}} │ {'RMSE':>12} │ {'ACC':>12} │")
    print(f"├{'─' * (w + 2)}┼{'─' * 14}┼{'─' * 14}┤")
    # 数据行
    for i, ch in enumerate(channels):
        print(f"│ {ch:<{w}} │ {channel_rmse[i]:>12.4f} | {channel_acc[i]:>12.4f} |")
    print(f"├{'─' * (w + 2)}┼{'─' * 14}┼{'─' * 14}┤")
    print(f"│ {'Average':<{w}} │ {np.mean(channel_rmse):>12.4f} │ {np.mean(channel_acc):>12.4f} │")
    print(f"└{'─' * (w + 2)}┴{'─' * 14}┴{'─' * 14}┘")

def plot(label, pred, var, filename):
    # 基础设置
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # 坐标轴标签
    xtick_labels = ['180°W', '90°W', '0°', '90°E', '180°E']
    ytick_labels = ['90°S', '45°S', '0°', '45°N', '90°N']
    xticks = np.linspace(0, label.shape[-1] - 1, 5)
    yticks = np.linspace(0, label.shape[-2] - 1, 5)

    # 计算统一色条范围
    vmin = min(label.min(), pred.min())
    vmax = max(label.max(), pred.max())

    # 计算差异和 RMSE
    diff = label - pred
    rmse = np.sqrt(np.mean(diff ** 2))
    diff_abs_max = np.abs(diff).max()

    # 绘图配置
    plot_configs = [
        {'data': label, 'title': 'Truth', 'cmap': 'viridis', 'vmin': vmin, 'vmax': vmax},
        {'data': pred,  'title': 'Prediction', 'cmap': 'viridis', 'vmin': vmin, 'vmax': vmax},
        {'data': diff,  'title': f'Difference (RMSE={rmse:.2f})', 'cmap': 'RdBu_r', 'vmin': -diff_abs_max, 'vmax': diff_abs_max},
    ]

    # 统一绘制
    for ax, cfg in zip(axes, plot_configs):
        im = ax.imshow(cfg['data'], cmap=cfg['cmap'], vmin=cfg['vmin'], vmax=cfg['vmax'])
        ax.set_title(cfg['title'], fontsize=12, pad=4)
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        ax.set_xticks(xticks)
        ax.set_xticklabels(xtick_labels)
        ax.set_yticks(yticks)
        ax.set_yticklabels(ytick_labels)
        plt.colorbar(im, ax=ax, orientation='horizontal')

    # 总标题
    fig.suptitle(var, fontsize=14, fontweight='bold', y=0.98)

    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

def plot_loss(train_loss, valid_loss):

    mask = ~(np.isnan(train_loss) | np.isnan(valid_loss))
    train_loss = train_loss[mask]
    valid_loss = valid_loss[mask]

    fig, ax = plt.subplots(figsize=(5, 3.5))
    # 配置
    colors = {'train': '#2563EB', 'valid': '#EA580C'}
    epochs = np.arange(1, len(train_loss) + 1)

    # 绑定曲线
    ax.plot(epochs, train_loss, color=colors['train'], linewidth=1.5, label='Train')
    ax.plot(epochs, valid_loss, color=colors['valid'], linewidth=1.5, label='Valid', linestyle='--')
    # 标注最小值
    min_idx = np.argmin(valid_loss)
    ax.scatter(epochs[min_idx], valid_loss[min_idx],
               color=colors['valid'], s=40, zorder=5, edgecolors='white')
    ax.annotate(f'Best: {valid_loss[min_idx]:.3f}',
                xy=(epochs[min_idx], valid_loss[min_idx]),
                xytext=(10, 10), textcoords='offset points', fontsize=8, color=colors['valid'],
                arrowprops=dict(arrowstyle='-', color=colors['valid'], lw=0.5))

    # 坐标轴
    ax.set(xlabel='Epoch', ylabel='Loss', xlim=(0, len(train_loss) + 1))

    # 样式
    ax.legend(frameon=False, loc='upper right')
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.spines[['top', 'right']].set_visible(False)

    plt.tight_layout()
    plt.savefig('./result/loss.png', dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    current_path = os.getcwd()
    sys.path.append(current_path)
    config_file_path = os.path.join(current_path, 'conf/config.yaml')
    cfg = YParams(config_file_path, 'model')
    cfg_data = YParams(config_file_path, "datapipe")

    train_loss = np.load('./data/checkpoints/trloss.npy')
    valid_loss = np.load('./data/checkpoints/valoss.npy')
    plot_loss(train_loss, valid_loss)

    data_dir = cfg_data.dataset.data_dir
    test_years = cfg_data.dataset.test_ratio
    total_files, channel_indices, h5_map = get_metadata(data_dir, test_years, cfg_data.dataset.channels)

    # Load data & Compute RMSE/ACC per channel
    stats_dir = os.path.join(data_dir, "stats")
    mu = np.load(os.path.join(stats_dir, "global_means.npy"))

    clim_mean = mu[:, channel_indices, :, :]
    get_result(total_files, channel_indices, h5_map, clim_mean)
    show_result()


    ##### 默认绘制 test_time 第一年的第一个时间步，用户可自行指定日期和变量 #####
    # test_year = cfg_data.dataset.test_time[0]
    # eg_files = [f'2012071606']
    # channel_index = [cfg_data.dataset.channels.index(v) for v in ['2m_temperature', 'geopotential_500', 'temperature_500']]

    # selected_var = [cfg_data.dataset.channels[int(i)] for i in channel_index]
    # print(f"seleted date: {eg_files}")
    # print(f"selected channels: {selected_var}")
    # for file in eg_files:
    #     h5_path = h5_map.get(file)
    #     if h5_path is None:
    #         print(f"⚠️  No HDF5 file for {file}, skipping plot")
    #         continue
    #     with h5py.File(h5_path, "r") as f:
    #         label = f["fields"][:].squeeze()  # [C, H, W]
    #         label = label[channel_indices]
    #     pred = np.load(f'result/output/{file}.npy').squeeze()
    #     for i in range(len(selected_var)):
    #         filename = f'./result/{file}_{selected_var[i]}.png'
    #         plot(label[channel_index[i]], pred[channel_index[i]], selected_var[i], filename)
    #         print(f'✅plot {filename}')
