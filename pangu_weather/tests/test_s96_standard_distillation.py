import ast
import itertools
import os
import unittest
from collections import OrderedDict
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F


PANGU = Path(__file__).parents[1]


def load_functions(path, names, namespace):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    selected = [
        node
        for node in tree.body
        if (isinstance(node, ast.FunctionDef) and node.name in names)
        or (isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id in names
            for target in node.targets
        ))
    ]
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), "exec"), namespace)
    return namespace


class S96StandardDistillationTests(unittest.TestCase):
    def test_checkpoint_resolution_does_not_duplicate_checkpoint_dir(self):
        namespace = load_functions(
            PANGU / "distill_train.py",
            {"checkpoint_path", "resolve_checkpoint_arg"},
            {"os": os},
        )
        resolve = namespace["resolve_checkpoint_arg"]
        cfg = SimpleNamespace(checkpoint_dir="./data/checkpoints")
        self.assertEqual(
            resolve(cfg, "student.pth"), "./data/checkpoints/student.pth"
        )
        self.assertEqual(
            resolve(cfg, "./data/checkpoints/student.pth"),
            "data/checkpoints/student.pth",
        )
        self.assertEqual(resolve(cfg, "/tmp/student.pth"), "/tmp/student.pth")

    def test_exact_depth_mapping_uses_entrance_and_exit_blocks(self):
        namespace = load_functions(
            PANGU / "scripts" / "prune_structured.py",
            {
                "S96_SOURCE_DEPTHS",
                "S96_TARGET_DEPTHS",
                "get_depth_block_map",
                "get_source_key_for_target",
                "_state_depths",
                "_exact_s96_depth_state",
            },
            {"OrderedDict": OrderedDict, "itertools": itertools},
        )
        source_depths = namespace["S96_SOURCE_DEPTHS"]
        target_depths = namespace["S96_TARGET_DEPTHS"]
        source_state = OrderedDict({"patchembed.marker": torch.tensor([99.0])})
        target_state = OrderedDict({"patchembed.marker": torch.tensor([0.0])})
        for layer, (source_depth, target_depth) in enumerate(
            zip(source_depths, target_depths), start=1
        ):
            for block in range(source_depth):
                source_state[f"layer{layer}.blocks.{block}.marker"] = torch.tensor(
                    [float(block)]
                )
            for block in range(target_depth):
                target_state[f"layer{layer}.blocks.{block}.marker"] = torch.tensor([0.0])

        migrated, source_keys, block_map = namespace["_exact_s96_depth_state"](
            source_state, target_state
        )
        self.assertEqual(block_map, [[0], [0, 5], [0, 5], [0]])
        self.assertEqual(float(migrated["layer2.blocks.1.marker"]), 5.0)
        self.assertEqual(float(migrated["layer3.blocks.1.marker"]), 5.0)
        self.assertEqual(
            source_keys["layer4.blocks.0.marker"], "layer4.blocks.0.marker"
        )
        self.assertEqual(set(migrated), set(target_state))
        wrapped_state = {
            f"layer{layer}.fuser.blocks.{block}.marker": torch.tensor([0.0])
            for layer, depth in enumerate(source_depths, start=1)
            for block in range(depth)
        }
        self.assertEqual(namespace["_state_depths"](wrapped_state), source_depths)

    def test_all_69_channels_and_equal_teacher_label_mix(self):
        namespace = load_functions(
            PANGU / "distill_train.py",
            {"forecast_loss", "distillation_loss"},
            {"F": F},
        )
        forecast_loss = namespace["forecast_loss"]
        distillation_loss = namespace["distillation_loss"]

        zeros_surface = torch.zeros(1, 4, 1, 1)
        zeros_upper = torch.zeros(1, 65, 1, 1)
        student_surface = torch.ones_like(zeros_surface)
        student_upper = torch.ones_like(zeros_upper)
        expected_branch_loss = 1.0 + 0.25
        self.assertAlmostEqual(
            float(
                forecast_loss(
                    student_surface,
                    student_upper,
                    zeros_surface,
                    zeros_upper,
                )
            ),
            expected_branch_loss,
        )

        target = (zeros_surface, zeros_upper)
        teacher = (torch.full_like(zeros_surface, 3.0), torch.full_like(zeros_upper, 3.0))
        total, hard, teacher_loss = distillation_loss(
            (student_surface, student_upper),
            target,
            teacher,
            ground_truth_weight=0.5,
            teacher_weight=0.5,
        )
        self.assertAlmostEqual(float(hard), 1.25)
        self.assertAlmostEqual(float(teacher_loss), 2.5)
        self.assertAlmostEqual(float(total), 0.5 * 1.25 + 0.5 * 2.5)

        one_channel = torch.zeros_like(zeros_upper)
        one_channel[:, -1] = 65.0
        self.assertAlmostEqual(
            float(forecast_loss(zeros_surface, one_channel, zeros_surface, zeros_upper)),
            1.0,
        )


if __name__ == "__main__":
    unittest.main()
