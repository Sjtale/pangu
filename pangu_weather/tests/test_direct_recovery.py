import importlib
import sys
import types
import unittest

import torch
import torch.nn as nn


def _install_onescience_stubs():
    onescience = types.ModuleType("onescience")
    models = types.ModuleType("onescience.models")
    pangu = types.ModuleType("onescience.models.pangu")
    modules = types.ModuleType("onescience.modules")
    attention = types.ModuleType("onescience.modules.attention")
    earthattention3d = types.ModuleType(
        "onescience.modules.attention.earthattention3d"
    )

    class Pangu(nn.Module):
        pass

    class OneRecovery(nn.Module):
        pass

    class OneFuser(nn.Module):
        pass

    class EarthAttention3D(nn.Module):
        pass

    pangu.Pangu = Pangu
    modules.OneRecovery = OneRecovery
    modules.OneFuser = OneFuser
    earthattention3d.EarthAttention3D = EarthAttention3D

    sys.modules.setdefault("onescience", onescience)
    sys.modules.setdefault("onescience.models", models)
    sys.modules.setdefault("onescience.models.pangu", pangu)
    sys.modules.setdefault("onescience.modules", modules)
    sys.modules.setdefault("onescience.modules.attention", attention)
    sys.modules.setdefault(
        "onescience.modules.attention.earthattention3d", earthattention3d
    )


_install_onescience_stubs()
pangu_profile_model = importlib.import_module("pangu_weather.pangu_profile_model")


class TinyPanguPatchRecovery(nn.Module):
    def __init__(self, img_size, patch_size, in_chans, out_chans):
        super().__init__()
        if len(img_size) == 2:
            img_size = (1, *img_size)
        if len(patch_size) == 2:
            patch_size = (1, *patch_size)
        self.img_size = tuple(img_size)
        self.patch_size = tuple(patch_size)
        self.in_chans = in_chans
        self.out_chans = out_chans
        self.proj = nn.ConvTranspose3d(
            in_chans,
            out_chans,
            kernel_size=self.patch_size,
            stride=self.patch_size,
        )

    def forward(self, x):
        squeeze_pressure_dim = False
        if x.ndim == 4:
            x = x.unsqueeze(2)
            squeeze_pressure_dim = True
        output = self.proj(x)
        output = pangu_profile_model._crop_recovery_output(self, output)
        if squeeze_pressure_dim:
            output = output.squeeze(2)
        return output


class DirectRecoveryTests(unittest.TestCase):
    def test_direct_recovery_matches_convtranspose3d_with_crop(self):
        torch.manual_seed(7)
        recovery = TinyPanguPatchRecovery(
            img_size=(3, 5, 7),
            patch_size=(2, 3, 2),
            in_chans=3,
            out_chans=4,
        )
        x = torch.randn(2, 3, 2, 2, 4)

        expected = recovery(x)
        actual = pangu_profile_model._direct_patch_recovery(
            recovery, x, width_chunk_size=2
        )

        self.assertEqual(tuple(actual.shape), tuple(expected.shape))
        self.assertLessEqual((actual - expected).abs().max().item(), 1e-6)

    def test_direct_recovery_matches_2d_surface_recovery_with_crop(self):
        torch.manual_seed(11)
        recovery = TinyPanguPatchRecovery(
            img_size=(5, 7),
            patch_size=(3, 2),
            in_chans=3,
            out_chans=2,
        )
        x = torch.randn(2, 3, 2, 4)

        expected = recovery(x)
        actual = pangu_profile_model._direct_patch_recovery(
            recovery, x, width_chunk_size=1
        )

        self.assertEqual(tuple(actual.shape), tuple(expected.shape))
        self.assertLessEqual((actual - expected).abs().max().item(), 1e-6)


if __name__ == "__main__":
    unittest.main()
