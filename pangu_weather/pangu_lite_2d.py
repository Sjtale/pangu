"""2D no-bias Pangu-Lite student used by the long-run KD pipeline."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class DropPath(nn.Module):
    def __init__(self, probability=0.0):
        super().__init__()
        self.probability = float(probability)

    def forward(self, x):
        if not self.training or self.probability == 0.0:
            return x
        keep = 1.0 - self.probability
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        return x * torch.empty(shape, device=x.device).bernoulli_(keep) / keep


class Mlp(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fc1 = nn.Linear(dim, 4 * dim)
        self.fc2 = nn.Linear(4 * dim, dim)

    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x)))


def _window_partition(x, window_size):
    batch, height, width, channels = x.shape
    wh, ww = window_size
    return (
        x.view(batch, height // wh, wh, width // ww, ww, channels)
        .permute(0, 1, 3, 2, 4, 5)
        .reshape(-1, wh * ww, channels)
    )


def _window_reverse(windows, window_size, batch, height, width):
    wh, ww = window_size
    return (
        windows.view(batch, height // wh, width // ww, wh, ww, -1)
        .permute(0, 1, 3, 2, 4, 5)
        .reshape(batch, height, width, -1)
    )


class EarthAttention2DNoBias(nn.Module):
    """Window attention with no Earth-specific relative/absolute bias."""

    def __init__(self, dim, num_heads):
        super().__init__()
        if dim % num_heads:
            raise ValueError("dim must be divisible by num_heads")
        self.num_heads = int(num_heads)
        self.scale = (dim // num_heads) ** -0.5
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x, mask=None):
        batch_windows, tokens, channels = x.shape
        qkv = self.qkv(x).reshape(
            batch_windows, tokens, 3, self.num_heads, channels // self.num_heads
        ).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        attention = (q * self.scale) @ k.transpose(-2, -1)
        if mask is not None:
            window_count = mask.shape[0]
            attention = attention.view(
                batch_windows // window_count,
                window_count,
                self.num_heads,
                tokens,
                tokens,
            )
            attention = attention + mask.unsqueeze(0).unsqueeze(2)
            attention = attention.view(batch_windows, self.num_heads, tokens, tokens)
        attention = attention.softmax(dim=-1)
        x = (attention @ v).transpose(1, 2).reshape(batch_windows, tokens, channels)
        return self.proj(x)


class EarthSpecificBlock2DNoBias(nn.Module):
    def __init__(self, dim, resolution, num_heads, window_size, shift, drop_path):
        super().__init__()
        self.resolution = tuple(resolution)
        self.window_size = tuple(window_size)
        self.shift = tuple(shift)
        self.norm1 = nn.LayerNorm(dim)
        self.attn = EarthAttention2DNoBias(dim, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(dim)
        self.drop_path = DropPath(drop_path)
        self.register_buffer("attn_mask", self._make_mask(), persistent=False)

    def _padded_resolution(self):
        height, width = self.resolution
        wh, ww = self.window_size
        return math.ceil(height / wh) * wh, math.ceil(width / ww) * ww

    def _make_mask(self):
        if self.shift == (0, 0):
            return None
        height, width = self._padded_resolution()
        wh, ww = self.window_size
        sh, sw = self.shift
        labels = torch.zeros((1, height, width, 1))
        hs = (slice(0, -wh), slice(-wh, -sh), slice(-sh, None))
        ws = (slice(0, -ww), slice(-ww, -sw), slice(-sw, None))
        counter = 0
        for h_slice in hs:
            for w_slice in ws:
                labels[:, h_slice, w_slice] = counter
                counter += 1
        windows = _window_partition(labels, self.window_size).squeeze(-1)
        difference = windows.unsqueeze(1) - windows.unsqueeze(2)
        return difference.masked_fill(difference != 0, -100.0)

    def forward(self, x):
        height, width = self.resolution
        batch, tokens, channels = x.shape
        if tokens != height * width:
            raise ValueError(f"expected {height * width} tokens, got {tokens}")
        shortcut = x
        x = self.norm1(x).view(batch, height, width, channels)
        padded_height, padded_width = self._padded_resolution()
        x = F.pad(x, (0, 0, 0, padded_width - width, 0, padded_height - height))
        if self.shift != (0, 0):
            x = torch.roll(x, shifts=(-self.shift[0], -self.shift[1]), dims=(1, 2))
        windows = _window_partition(x, self.window_size)
        windows = self.attn(windows, self.attn_mask)
        x = _window_reverse(windows, self.window_size, batch, padded_height, padded_width)
        if self.shift != (0, 0):
            x = torch.roll(x, shifts=self.shift, dims=(1, 2))
        x = x[:, :height, :width].reshape(batch, tokens, channels)
        x = shortcut + self.drop_path(x)
        return x + self.drop_path(self.mlp(self.norm2(x)))


class EarthSpecificLayer2DNoBias(nn.Module):
    def __init__(self, depth, dim, resolution, num_heads, window_size, drop_path):
        super().__init__()
        half_window = (window_size[0] // 2, window_size[1] // 2)
        self.blocks = nn.ModuleList(
            EarthSpecificBlock2DNoBias(
                dim,
                resolution,
                num_heads,
                window_size,
                (0, 0) if index % 2 == 0 else half_window,
                drop_path[index],
            )
            for index in range(depth)
        )

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x


class DownSample2D(nn.Module):
    def __init__(self, dim, input_resolution):
        super().__init__()
        self.input_resolution = tuple(input_resolution)
        self.norm = nn.LayerNorm(4 * dim)
        self.linear = nn.Linear(4 * dim, 2 * dim, bias=False)

    def forward(self, x):
        height, width = self.input_resolution
        batch, _, channels = x.shape
        x = x.view(batch, height, width, channels)
        x = F.pad(x, (0, 0, 0, width % 2, 0, height % 2))
        x = torch.cat((x[:, 0::2, 0::2], x[:, 1::2, 0::2], x[:, 0::2, 1::2], x[:, 1::2, 1::2]), dim=-1)
        return self.linear(self.norm(x.reshape(batch, -1, 4 * channels)))


class UpSample2D(nn.Module):
    def __init__(self, input_dim, output_dim, input_resolution, output_resolution):
        super().__init__()
        self.input_resolution = tuple(input_resolution)
        self.output_resolution = tuple(output_resolution)
        self.output_dim = int(output_dim)
        self.linear1 = nn.Linear(input_dim, 4 * output_dim, bias=False)
        self.norm = nn.LayerNorm(output_dim)
        self.linear2 = nn.Linear(output_dim, output_dim, bias=False)

    def forward(self, x):
        batch = x.shape[0]
        height, width = self.input_resolution
        x = self.linear1(x).view(batch, height, width, 2, 2, self.output_dim)
        x = x.permute(0, 1, 3, 2, 4, 5).reshape(batch, 2 * height, 2 * width, self.output_dim)
        target_height, target_width = self.output_resolution
        x = x[:, :target_height, :target_width].reshape(batch, -1, self.output_dim)
        return self.linear2(self.norm(x))


class PanguLite2DAttentionPosEmbed(nn.Module):
    def __init__(self, img_size=(721, 1440), patch_size=(8, 8), dim=288):
        super().__init__()
        if tuple(patch_size) != (8, 8) or int(dim) != 288:
            raise ValueError("KD student is locked to patch_size=(8,8), dim=288")
        self.img_size = tuple(img_size)
        self.patch_size = tuple(patch_size)
        self.dim = int(dim)
        high = (math.ceil(img_size[0] / 8), math.ceil(img_size[1] / 8))
        low = (math.ceil(high[0] / 2), math.ceil(high[1] / 2))
        if high != (91, 180):
            raise ValueError("absolute_pos_embed contract requires img_size=(721,1440)")
        self.patchembed = nn.Conv2d(72, dim, kernel_size=patch_size, stride=patch_size)
        self.absolute_pos_embed = nn.Parameter(torch.zeros(1, 91, dim))
        drop = torch.linspace(0, 0.2, 16).tolist()
        self.layer1 = EarthSpecificLayer2DNoBias(2, dim, high, 6, (6, 12), drop[:2])
        self.downsample = DownSample2D(dim, high)
        self.layer2 = EarthSpecificLayer2DNoBias(6, 2 * dim, low, 12, (6, 12), drop[2:8])
        self.layer3 = EarthSpecificLayer2DNoBias(6, 2 * dim, low, 12, (6, 12), drop[8:14])
        self.upsample = UpSample2D(2 * dim, dim, low, high)
        self.layer4 = EarthSpecificLayer2DNoBias(2, dim, high, 6, (6, 12), drop[14:])
        self.patchrecovery = nn.ConvTranspose2d(2 * dim, 69, kernel_size=patch_size, stride=patch_size)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.trunc_normal_(self.absolute_pos_embed, std=0.02)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x):
        if isinstance(x, (tuple, list)):
            surface, upper = x
            x = torch.cat((surface, upper.flatten(1, 2)), dim=1)
        if x.shape[1] != 72:
            raise ValueError(f"expected 72 input channels, got {x.shape[1]}")
        original_height, original_width = x.shape[-2:]
        x = F.pad(x, (0, (-original_width) % 8, 0, (-original_height) % 8))
        x = self.patchembed(x).permute(0, 2, 3, 1)
        x = x + self.absolute_pos_embed.unsqueeze(2)
        x = x.reshape(x.shape[0], -1, self.dim)
        x = self.layer1(x)
        skip = x
        x = self.layer2(self.downsample(x))
        x = self.layer3(x)
        x = self.layer4(self.upsample(x))
        x = torch.cat((skip, x), dim=-1).transpose(1, 2).reshape(x.shape[0], 2 * self.dim, 91, 180)
        output = self.patchrecovery(x)[:, :, :original_height, :original_width]
        return output[:, :4], output[:, 4:]


PanguModel = PanguLite2DAttentionPosEmbed
