import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[2]))

from pangu_weather.calibration_utils import (
    SCORED_CHANNELS,
    blocked_slope_calibration,
    scored_channel_indices,
)


class BlockedCalibrationTests(unittest.TestCase):
    def test_official_scored_channel_order(self):
        channels = ["unscored_a", *reversed(SCORED_CHANNELS), "unscored_b"]
        indices = scored_channel_indices(channels)
        self.assertEqual([channels[i] for i in indices], list(SCORED_CHANNELS))

    def test_recovers_scored_slopes_and_preserves_unscored_coefficients(self):
        rng = np.random.default_rng(7)
        samples, channels, height, width = 8, 69, 3, 4
        climatology = rng.normal(size=(channels, height, width))
        target_anom = rng.normal(size=(samples, channels, height, width))
        targets = climatology + target_anom
        accepted = np.linspace(0.8, 1.2, channels, dtype=np.float32)
        scored = np.arange(15)
        desired_adjustment = np.linspace(0.9, 1.1, 15)
        predictions = targets.copy()
        predictions[:, scored] = climatology[scored] + (
            target_anom[:, scored] / desired_adjustment.reshape(1, -1, 1, 1)
        )

        result = blocked_slope_calibration(
            predictions,
            targets,
            climatology,
            accepted,
            scored,
            num_blocks=4,
            minimum_block_gain=0.0,
        )

        expected = accepted[scored] * desired_adjustment
        self.assertTrue(np.allclose(result.candidate_coeffs[scored], expected, atol=1e-6))
        self.assertTrue(np.array_equal(result.candidate_coeffs[15:], accepted[15:]))
        self.assertTrue(result.promotion_eligible)
        self.assertGreater(result.worst_relative_w, 40.0)

    def test_perfect_baseline_fails_positive_gain_gate(self):
        rng = np.random.default_rng(11)
        targets = rng.normal(size=(4, 69, 2, 3))
        climatology = np.zeros((69, 2, 3))
        accepted = np.ones(69, dtype=np.float32)

        result = blocked_slope_calibration(
            targets,
            targets,
            climatology,
            accepted,
            np.arange(15),
            num_blocks=2,
            minimum_block_gain=0.001,
        )

        self.assertFalse(result.promotion_eligible)
        self.assertAlmostEqual(result.worst_relative_w, 40.0, places=6)


if __name__ == "__main__":
    unittest.main()
