import torch
import os
import sys
import glob
import numpy as np
import h5py
from tqdm import tqdm
import time
import json
from onescience.models.pangu import Pangu
from onescience.utils.YParams import YParams
from onescience.datapipes.climate import ERA5Datapipe


def get_stats(data_dir, channels):
    """从 metadata.json 读取变量列表，提取归一化参数"""
    with open(os.path.join(data_dir, "metadata.json"), "r") as f:
        metadata = json.load(f)
    all_variables = metadata["variables"]

    channel_indices = [all_variables.index(v) for v in channels]
    stats_dir = os.path.join(data_dir, "stats")
    mu = np.load(os.path.join(stats_dir, "global_means.npy"))   # [1, C, 1, 1]
    std = np.load(os.path.join(stats_dir, "global_stds.npy"))
    means = mu[:, channel_indices, :, :]
    stds = std[:, channel_indices, :, :]
    return means, stds


if __name__ == "__main__":
    current_path = os.getcwd()
    sys.path.append(current_path)

    ## Model config init
    config_file_path = os.path.join(current_path, "conf/config.yaml")
    cfg = YParams(config_file_path, "model")
    ## DataLoader init
    cfg_data = YParams(config_file_path, "datapipe")

    means, stds = get_stats(cfg_data.dataset.data_dir, cfg_data.dataset.channels)

    datapipe = ERA5Datapipe(params=cfg_data, distributed=False)
    test_dataloader = datapipe.test_dataloader()

    land_mask = torch.from_numpy(np.load(os.path.join(cfg_data.dataset.static_dir, "land_mask.npy")).astype(np.float32))
    soil_type = torch.from_numpy(np.load(os.path.join(cfg_data.dataset.static_dir, "soil_type.npy")).astype(np.float32))
    topography = torch.from_numpy(np.load(os.path.join(cfg_data.dataset.static_dir, "topography.npy")).astype(np.float32))
    topography = (topography - topography.mean()) / (topography.std(unbiased=False) + 1e-6)
    surface_mask = torch.stack([land_mask, soil_type, topography], dim=0).to('cuda:0')
    surface_mask = surface_mask.unsqueeze(0).repeat(cfg_data.dataloader.batch_size, 1, 1, 1)
    surface_mask = surface_mask.half()  # FP16: static mask 也转为半精度

    # ---- FP16: 优先加载 FP16 权重，回退到原始 FP32 权重 ----
    fp16_ckpt_path = f"{cfg.checkpoint_dir}/model_fp16.pth"
    fp32_ckpt_path = f"{cfg.checkpoint_dir}/model_bak.pth"
    if os.path.exists(fp16_ckpt_path):
        print(f"⚡ 加载 FP16 权重: {fp16_ckpt_path}")
        ckpt = torch.load(fp16_ckpt_path, map_location="cuda:0")
    else:
        print(f"ℹ️  未找到 FP16 权重，回退加载 FP32: {fp32_ckpt_path}")
        ckpt = torch.load(fp32_ckpt_path, map_location="cuda:0")

    model = Pangu(img_size=cfg_data.dataset.img_size,
                  patch_size=cfg.patch_size,
                  embed_dim=cfg.embed_dim,
                  num_heads=cfg.num_heads,
                  window_size=cfg.window_size,
                  ).to('cuda:0')
    model.load_state_dict(ckpt["model_state_dict"])
    model.half()   # FP16: 确保整个模型在半精度下运行

    model.eval()
    os.makedirs('result/output/', exist_ok=True)                          # AI4S, 输出路径不可更改
    print(f"📂 samples will be generated to './result/output/'")

    time_list = []
    first = True
    with torch.no_grad():
        for data in tqdm(test_dataloader, desc="Inferring testset", unit="batch"):
            invar = data[0]
            outvar = data[1]
            filename = data[4][-1][0]
            if first:
                first = False
                print(f"  invar  shape: {list(invar.shape)}   ← [batch, channels, H, W]")
                print(f"  outvar shape: {list(outvar.shape)}  ← [batch, channels, H, W]")
                print(f"  the first filename: {filename}")

            # FP16: 输入数据直接转为半精度
            invar_surface = invar[:, :4, :, :].to("cuda:0", dtype=torch.float16)
            invar_upper_air = invar[:, 4:, :, :].to("cuda:0", dtype=torch.float16)
            invar = torch.concat([invar_surface, surface_mask, invar_upper_air], dim=1)
            
            #----------------------AI4S(时间度量不可更改)---------------------------
            start_time = time.perf_counter()      # AI4S(时间度量，位置不可更改)
            out_surface, out_upper_air = model(invar)
            end_time = time.perf_counter()        # AI4S(时间度量，位置不可更改)
            time_list.append(end_time-start_time) # AI4S(时间度量，位置不可更改)
            #---------------------------------------------------------------------

            out_upper_air = out_upper_air.reshape(invar_upper_air.shape)
            # FP16: 输出转回 float32 再做反归一化，避免半精度下乘法精度损失
            pred_var = torch.concat([out_surface, out_upper_air], dim=1).float().cpu().numpy()
            pred_var = pred_var * stds + means
            np.save(f"result/output/{filename}.npy", pred_var)


        #----------------------AI4S(时间度量不可更改)---------------------------
        # 保存到 time_list.json 文件
        with open("result/time_record.json", "w", encoding="utf-8") as f:
            json.dump(time_list, f, ensure_ascii=False, indent=4)
        #---------------------------------------------------------------------

    if torch.cuda.is_available():
        print(f"Max VRAM: {torch.cuda.max_memory_allocated() / 1024**2:.1f} MB")
        print(f"Current VRAM: {torch.cuda.memory_allocated() / 1024**2:.1f} MB")

