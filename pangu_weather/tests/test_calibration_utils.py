import tempfile
import unittest
from pathlib import Path
import sys

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[2]))

from pangu_weather.calibration_utils import (
    AffineCalibration,
    GlobalMeanCorrection,
    apply_affine_calibration,
    apply_global_mean_correction,
    fit_affine_from_sums,
    load_affine_calibration,
    save_affine_calibration,
    weighted_channel_mean,
)


class CalibrationUtilsTests(unittest.TestCase):
    def test_fit_affine_recovers_scale_and_bias(self):
        x = np.array([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]], dtype=np.float64)
        y0 = 1.2 * x[0] + 0.5
        y1 = 0.8 * x[1] - 0.25
        y = np.stack([y0, y1])

        affine = fit_affine_from_sums(
            sum_x=np.sum(x, axis=1),
            sum_y=np.sum(y, axis=1),
            sum_xx=np.sum(x * x, axis=1),
            sum_xy=np.sum(x * y, axis=1),
            count=x.shape[1],
            channel_stds=np.array([10.0, 10.0]),
        )

        self.assertTrue(np.allclose(affine.scale, [1.2, 0.8], atol=1e-6))
        self.assertTrue(np.allclose(affine.bias, [0.5, -0.25], atol=1e-6))

    def test_affine_roundtrip_and_apply_shape(self):
        pred = np.ones((1, 2, 3, 4), dtype=np.float32) * 3.0
        clim = np.ones((1, 2, 1, 1), dtype=np.float32)
        affine = AffineCalibration(
            scale=np.array([2.0, 0.5], dtype=np.float32),
            bias=np.array([1.0, -1.0], dtype=np.float32),
        )

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "calibration_affine.npz"
            save_affine_calibration(str(path), affine)
            loaded = load_affine_calibration(str(path), 2)

        actual = apply_affine_calibration(pred, clim, loaded)
        self.assertEqual(actual.shape, pred.shape)
        self.assertFalse(np.isnan(actual).any())
        self.assertTrue(np.allclose(actual[:, 0], 6.0))
        self.assertTrue(np.allclose(actual[:, 1], 1.0))

    def test_global_mean_correction_only_changes_masked_channels(self):
        pred = np.zeros((1, 2, 3, 4), dtype=np.float32)
        pred[:, 0] = 2.0
        pred[:, 1] = 5.0
        correction = GlobalMeanCorrection(
            target_mean=np.array([4.0, 1.0], dtype=np.float32),
            channel_mask=np.array([True, False]),
        )

        actual = apply_global_mean_correction(pred, correction)
        means = weighted_channel_mean(actual, np.cos(np.deg2rad(np.linspace(-90, 90, 3))))
        self.assertAlmostEqual(float(means[0]), 4.0, places=5)
        self.assertAlmostEqual(float(means[1]), 5.0, places=5)


if __name__ == "__main__":
    unittest.main()
