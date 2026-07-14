import importlib.util
import tempfile
import unittest
from collections import OrderedDict
from pathlib import Path

import torch
import torch.nn.functional as F


PANGU = Path(__file__).parents[1]
SCRIPT = PANGU / "scripts" / "initialize_selective_mlp96.py"
SPEC = importlib.util.spec_from_file_location("initialize_selective_mlp96", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class SelectiveMLP96SamplingTests(unittest.TestCase):
    def test_public_protocol_constants_are_exact(self):
        self.assertEqual(MODULE.PROFILE_NAME, "selective_mlp96")
        self.assertEqual(MODULE.HUMAN_LABEL, "SelectiveMLP-96")
        self.assertEqual(
            MODULE.INITIALIZATION_METHOD,
            "pruned96_activation_aware_mlp_pair_selection",
        )
        self.assertEqual(MODULE.OFFICIAL_TRAIN_INPUT_COUNT, 32)
        self.assertEqual(MODULE.TOKEN_SAMPLE_COUNT, 4096)
        self.assertEqual(MODULE.TOKENS_PER_INPUT, 4096)
        self.assertEqual(MODULE.TOTAL_SAMPLED_TOKENS, 131072)
        self.assertEqual(MODULE.TOKEN_SAMPLE_SEED, 20260713)
        self.assertEqual(MODULE.SELECTED_NEURON_COUNT, 384)
        self.assertEqual(len(MODULE.SELECTIVE_MLP_PREFIXES), 11)
        self.assertEqual(
            MODULE.SELECTIVE_MLP_PREFIXES,
            tuple(
                [
                    f"layer2.Fuser.blocks.{block}.transformer.mlp"
                    for block in range(1, 6)
                ]
                + [
                    f"layer3.Fuser.blocks.{block}.transformer.mlp"
                    for block in range(6)
                ]
            ),
        )

    def test_official_train_indices_are_even_unique_and_endpoint_aligned(self):
        first = MODULE.official_train_input_indices(320)
        second = MODULE.official_train_input_indices(320)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 32)
        self.assertEqual(first[0], 0)
        self.assertEqual(first[-1], 319)
        self.assertEqual(len(set(first)), 32)
        self.assertTrue(set(b - a for a, b in zip(first, first[1:])) <= {10, 11})
        with self.assertRaisesRegex(ValueError, "32 are required"):
            MODULE.official_train_input_indices(31)

    def test_4096_tokens_per_each_input_are_streamed_repeatably(self):
        prefix = MODULE.SELECTIVE_MLP_PREFIXES[0]
        collectors = [
            MODULE.ActivationRMSCollector(prefixes=(prefix,)),
            MODULE.ActivationRMSCollector(prefixes=(prefix,)),
            MODULE.ActivationRMSCollector(prefixes=(prefix,), seed=20260714),
        ]
        for sample_index in range(32):
            activation = (
                torch.arange(5000 * 3, dtype=torch.float32).reshape(1, 5000, 3)
                / 10000.0
                + sample_index
            )
            for collector in collectors:
                collector.add(prefix, activation)
        statistics = [collector.finalize()[prefix] for collector in collectors]
        self.assertEqual(tuple(statistics[0].shape), (3,))
        self.assertTrue(torch.equal(statistics[0], statistics[1]))
        self.assertFalse(torch.equal(statistics[0], statistics[2]))
        self.assertEqual(collectors[0].total_tokens, 131072)

    def test_collector_rejects_incomplete_capture(self):
        prefix = MODULE.SELECTIVE_MLP_PREFIXES[0]
        collector = MODULE.ActivationRMSCollector(prefixes=(prefix,))
        collector.add(prefix, torch.zeros(4096, 2))
        with self.assertRaisesRegex(RuntimeError, "1/32"):
            collector.finalize()

    def test_selected_blocks_force_one_complete_hook_call(self):
        class FakeMlp:
            pass

        class FakeBlock:
            def __init__(self, mlp):
                self.mlp = mlp
                self._pangu_mlp_chunk_size = 32768

        class FakeModel:
            def __init__(self):
                self.modules = []
                self.blocks = []
                self.unselected = FakeBlock(FakeMlp())
                self.modules.extend(
                    [
                        ("layer2.Fuser.blocks.0.transformer", self.unselected),
                        (
                            "layer2.Fuser.blocks.0.transformer.mlp",
                            self.unselected.mlp,
                        ),
                    ]
                )
                for prefix in MODULE.SELECTIVE_MLP_PREFIXES:
                    mlp = FakeMlp()
                    block = FakeBlock(mlp)
                    self.blocks.append(block)
                    self.modules.extend(
                        [(prefix.rsplit(".mlp", 1)[0], block), (prefix, mlp)]
                    )

            def named_modules(self, remove_duplicate=False):
                del remove_duplicate
                return iter(self.modules)

        model = FakeModel()
        MODULE.force_single_call_activation_capture(model)
        self.assertTrue(
            all(block._pangu_mlp_chunk_size == 0 for block in model.blocks)
        )
        self.assertEqual(model.unselected._pangu_mlp_chunk_size, 32768)


class SelectiveMLP96ImportanceTests(unittest.TestCase):
    def test_importance_matches_required_formula(self):
        inputs = torch.tensor([[1.0, -1.0], [2.0, 0.5], [-0.5, 1.5]])
        fc1_weight = torch.tensor([[1.0, 2.0], [-1.0, 0.5], [0.25, -0.75]])
        fc1_bias = torch.tensor([0.5, -0.25, 1.0])
        fc2_weight = torch.tensor([[1.0, 2.0, 3.0], [4.0, 0.5, -2.0]])
        actual = MODULE.mlp_neuron_importance(
            inputs, fc1_weight, fc1_bias, fc2_weight
        )
        activated = F.gelu(F.linear(inputs, fc1_weight, fc1_bias))
        expected = activated.square().mean(0).sqrt() * torch.linalg.vector_norm(
            fc2_weight, dim=0
        )
        self.assertTrue(torch.allclose(actual, expected, rtol=1e-6, atol=1e-7))
        streamed = MODULE.importance_from_activation_rms(
            activated.square().mean(0).sqrt(), fc2_weight
        )
        self.assertTrue(torch.allclose(streamed, expected, rtol=1e-6, atol=1e-7))

    def test_top_indices_have_deterministic_low_index_tie_break(self):
        scores = torch.ones(768)
        selected = MODULE.deterministic_top_indices(scores, 384)
        self.assertTrue(torch.equal(selected, torch.arange(384)))

    def test_paired_top384_copy_and_exact_untouched_state(self):
        prefix = MODULE.SELECTIVE_MLP_PREFIXES[0]
        hidden = 768
        fc1_weight = torch.arange(hidden * 2, dtype=torch.float32).reshape(hidden, 2)
        fc1_bias = torch.arange(hidden, dtype=torch.float32)
        fc2_weight = torch.arange(2 * hidden, dtype=torch.float32).reshape(2, hidden)
        fc2_bias = torch.tensor([7.0, 8.0])
        untouched = torch.tensor([3.0, 4.0, 5.0])
        source = OrderedDict(
            [
                (prefix + ".fc1.weight", fc1_weight),
                (prefix + ".fc1.bias", fc1_bias),
                (prefix + ".fc2.weight", fc2_weight),
                (prefix + ".fc2.bias", fc2_bias),
                ("layer1.norm.weight", untouched),
            ]
        )
        target = OrderedDict(
            [
                (prefix + ".fc1.weight", torch.empty(384, 2)),
                (prefix + ".fc1.bias", torch.empty(384)),
                (prefix + ".fc2.weight", torch.empty(2, 384)),
                (prefix + ".fc2.bias", torch.empty(2)),
                ("layer1.norm.weight", torch.empty(3)),
            ]
        )
        activation_rms = {prefix: torch.arange(hidden, dtype=torch.float32)}
        migrated, indices, source_map = MODULE.initialize_selective_mlp_state(
            source,
            target,
            activation_rms,
            prefixes=(prefix,),
            selected_count=384,
        )
        expected_indices = torch.arange(384, 768)
        self.assertTrue(torch.equal(indices[prefix], expected_indices))
        self.assertTrue(
            torch.equal(migrated[prefix + ".fc1.weight"], fc1_weight[expected_indices])
        )
        self.assertTrue(
            torch.equal(migrated[prefix + ".fc1.bias"], fc1_bias[expected_indices])
        )
        self.assertTrue(
            torch.equal(
                migrated[prefix + ".fc2.weight"], fc2_weight[:, expected_indices]
            )
        )
        self.assertTrue(torch.equal(migrated[prefix + ".fc2.bias"], fc2_bias))
        self.assertTrue(torch.equal(migrated["layer1.norm.weight"], untouched))
        self.assertNotEqual(
            migrated["layer1.norm.weight"].data_ptr(), untouched.data_ptr()
        )
        self.assertEqual(set(migrated), set(target))
        self.assertEqual(set(source_map), set(target))


class SelectiveMLP96AuditTests(unittest.TestCase):
    def test_quantized_and_incomplete_sources_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "unquantized"):
            MODULE.reject_quantized_source(
                {"quantization": {"method": "mixed_int8_fp16"}},
                OrderedDict(weight=torch.ones(2)),
            )
        with self.assertRaisesRegex(ValueError, "unquantized"):
            MODULE.reject_quantized_source(
                {}, OrderedDict(weight=torch.ones(2), weight_scale=torch.ones(1))
            )
        with self.assertRaisesRegex(ValueError, "unquantized"):
            MODULE.reject_quantized_source(
                {}, OrderedDict(weight=torch.ones(2, dtype=torch.int8))
            )
        expected = OrderedDict(weight=torch.zeros(2, 3), bias=torch.zeros(2))
        MODULE.validate_complete_source(OrderedDict(expected), expected)
        with self.assertRaisesRegex(ValueError, "Incomplete source"):
            MODULE.validate_complete_source(
                OrderedDict(weight=torch.zeros(2, 3)), expected
            )
        with self.assertRaisesRegex(ValueError, "shape_mismatch"):
            MODULE.validate_complete_source(
                OrderedDict(weight=torch.zeros(3, 2), bias=torch.zeros(2)), expected
            )

    def test_logical_state_hash_is_order_independent_and_content_sensitive(self):
        first = OrderedDict(
            weight=torch.arange(6, dtype=torch.float32).reshape(2, 3),
            index=torch.tensor([2, 1], dtype=torch.int64),
        )
        reordered = OrderedDict(reversed(list(first.items())))
        changed = OrderedDict(first)
        changed["weight"] = changed["weight"].clone()
        changed["weight"][0, 0] = 99
        self.assertEqual(
            MODULE.state_dict_sha256(first), MODULE.state_dict_sha256(reordered)
        )
        self.assertNotEqual(
            MODULE.state_dict_sha256(first), MODULE.state_dict_sha256(changed)
        )

    def test_metadata_records_all_hashes_indices_and_zero_random_parameters(self):
        state = OrderedDict(weight=torch.arange(8, dtype=torch.float32))
        neurons = OrderedDict(
            (prefix, torch.arange(384)) for prefix in MODULE.SELECTIVE_MLP_PREFIXES
        )
        train_indices = MODULE.official_train_input_indices(320)
        metadata = MODULE.build_initialization_metadata(
            source_sha256="a" * 64,
            teacher_sha256="b" * 64,
            initialized_state=state,
            neuron_indices=neurons,
            train_indices=train_indices,
            state_key_count=17,
            parameter_key_count=13,
        )
        self.assertEqual(metadata["source_sha256"], "a" * 64)
        self.assertEqual(metadata["teacher_sha256"], "b" * 64)
        self.assertEqual(metadata["init_sha256"], MODULE.state_dict_sha256(state))
        self.assertEqual(metadata["method"], MODULE.INITIALIZATION_METHOD)
        self.assertEqual(metadata["activation_calibration"]["input_indices"], train_indices)
        self.assertEqual(metadata["activation_calibration"]["tokens_per_input"], 4096)
        self.assertEqual(metadata["activation_calibration"]["tokens_per_mlp"], 131072)
        self.assertEqual(metadata["activation_calibration"]["seed"], 20260713)
        self.assertEqual(len(metadata["neuron_indices"]), 11)
        self.assertEqual(metadata["covered_state_tensor_keys"], 17)
        self.assertEqual(metadata["covered_parameter_state_keys"], 13)
        self.assertTrue(metadata["strict_coverage"])
        self.assertEqual(metadata["random_initialized_parameters"], 0)

    def test_atomic_save_refuses_overwrite_and_preserves_init_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "selective_mlp96.pth"
            state = OrderedDict(weight=torch.arange(4))
            payload = {"model_state_dict": state}
            output_hash = MODULE.atomic_save(payload, output)
            self.assertEqual(output_hash, MODULE.sha256_file(output))
            loaded = torch.load(output, map_location="cpu", weights_only=False)
            self.assertEqual(
                MODULE.state_dict_sha256(loaded["model_state_dict"]),
                MODULE.state_dict_sha256(state),
            )
            with self.assertRaises(FileExistsError):
                MODULE.atomic_save(payload, output)


if __name__ == "__main__":
    unittest.main()
