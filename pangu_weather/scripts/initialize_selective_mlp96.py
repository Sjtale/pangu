#!/usr/bin/env python3
"""Build SelectiveMLP-96 from an unquantized full-depth pruned-96 source.

Only eleven deep MLPs are changed.  For each changed MLP, 384 neurons are
selected with official-training activations and copied as an inseparable
``fc1`` row/bias + ``fc2`` column group.  Every other state tensor is copied
bit-for-bit from the source checkpoint.

The reusable functions in this file intentionally depend only on PyTorch.
Project/data imports are delayed until the CLI is executed so the selection,
coverage, and provenance rules can be tested with synthetic tensors.
"""

import argparse
import hashlib
import os
import random
import re
import sys
from collections import OrderedDict
from pathlib import Path

import torch
import torch.nn.functional as F


PANGU_DIR = Path(__file__).resolve().parents[1]

PROFILE_NAME = "selective_mlp96"
HUMAN_LABEL = "SelectiveMLP-96"
INITIALIZATION_METHOD = "pruned96_activation_aware_mlp_pair_selection"

OFFICIAL_TRAIN_INPUT_COUNT = 32
# Required independently for every target MLP and every selected input.
TOKEN_SAMPLE_COUNT = 4096
TOKEN_SAMPLE_SEED = 20260713
TOKENS_PER_INPUT = TOKEN_SAMPLE_COUNT
TOTAL_SAMPLED_TOKENS = OFFICIAL_TRAIN_INPUT_COUNT * TOKENS_PER_INPUT
SELECTED_NEURON_COUNT = 384

SOURCE_PATCH_SIZE = [2, 8, 8]
SOURCE_EMBED_DIM = 96
SOURCE_NUM_HEADS = [3, 6, 6, 3]
SOURCE_DEPTHS = [2, 6, 6, 2]
SOURCE_WINDOW_SIZE = [2, 6, 12]
SOURCE_MLP_RATIO_BLOCKS = [[4, 4], [4, 4, 4, 4, 4, 4], [4, 4, 4, 4, 4, 4], [4, 4]]
TARGET_MLP_RATIO_BLOCKS = [[4, 4], [4, 2, 2, 2, 2, 2], [2, 2, 2, 2, 2, 2], [4, 4]]
EXPECTED_PARAMETER_COUNT = 14_768_265

# ``Fuser`` is the canonical name used by compact checkpoints.  Helpers below
# also accept the model's lower-case alias without creating a second schedule.
SELECTIVE_MLP_PREFIXES = tuple(
    [f"layer2.Fuser.blocks.{block}.transformer.mlp" for block in range(1, 6)]
    + [f"layer3.Fuser.blocks.{block}.transformer.mlp" for block in range(6)]
)

IMPORTANCE_FORMULA = "sqrt(mean(GELU(fc1(x))^2))*l2_norm(fc2[:,neuron])"
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")


def canonical_state_key(key):
    """Normalize the duplicate OneFuser registration to its compact alias."""

    return str(key).replace(".fuser.", ".Fuser.")


def official_train_input_indices(dataset_length, count=OFFICIAL_TRAIN_INPUT_COUNT):
    """Return ``count`` endpoint-inclusive, evenly spaced training indices."""

    dataset_length = int(dataset_length)
    count = int(count)
    if count < 2:
        raise ValueError("At least two official-train inputs are required")
    if dataset_length < count:
        raise ValueError(
            f"Official training dataset has {dataset_length} inputs; {count} are required"
        )
    denominator = count - 1
    # Integer round-to-nearest avoids platform-dependent floating-point linspace.
    indices = [
        (position * (dataset_length - 1) + denominator // 2) // denominator
        for position in range(count)
    ]
    if len(set(indices)) != count or indices[0] != 0 or indices[-1] != dataset_length - 1:
        raise AssertionError("Evenly spaced official-train indices are not unique/end-aligned")
    return indices


class ActivationRMSCollector:
    """Stream GELU(fc1(x)) RMS from 4096 sampled tokens for each input/MLP."""

    def __init__(
        self,
        prefixes=SELECTIVE_MLP_PREFIXES,
        *,
        seed=TOKEN_SAMPLE_SEED,
        input_count=OFFICIAL_TRAIN_INPUT_COUNT,
        tokens_per_input=TOKENS_PER_INPUT,
    ):
        prefixes = tuple(canonical_state_key(prefix) for prefix in prefixes)
        if len(set(prefixes)) != len(prefixes):
            raise ValueError("MLP prefixes must be unique after alias canonicalization")
        if input_count <= 0 or tokens_per_input <= 0:
            raise ValueError("input_count and tokens_per_input must be positive")
        self.prefixes = prefixes
        self.seed = int(seed)
        self.input_count = int(input_count)
        self.tokens_per_input = int(tokens_per_input)
        self.total_tokens = self.input_count * self.tokens_per_input
        self._rng = {prefix: random.Random(self.seed) for prefix in prefixes}
        self._sum_squares = {prefix: None for prefix in prefixes}
        self._calls = {prefix: 0 for prefix in prefixes}

    def add(self, prefix, fc1_output):
        prefix = canonical_state_key(prefix)
        if prefix not in self._sum_squares:
            raise KeyError(f"Activation was captured for an unscheduled MLP: {prefix}")
        if self._calls[prefix] >= self.input_count:
            raise RuntimeError(f"Too many activation inputs captured for {prefix}")
        if not isinstance(fc1_output, torch.Tensor) or fc1_output.ndim < 2:
            raise TypeError("fc1 output must be a rank-two-or-higher tensor")
        flat = fc1_output.detach().reshape(-1, fc1_output.shape[-1])
        if flat.shape[0] < self.tokens_per_input:
            raise ValueError(
                f"{prefix} has only {flat.shape[0]} tokens; "
                f"{self.tokens_per_input} are required per input"
            )
        indices = sorted(
            self._rng[prefix].sample(range(flat.shape[0]), self.tokens_per_input)
        )
        index = torch.tensor(indices, dtype=torch.long, device=flat.device)
        selected = flat.index_select(0, index).to(dtype=torch.float32)
        sum_squares = F.gelu(selected, approximate="none").square().sum(dim=0)
        sum_squares = sum_squares.to(device="cpu", dtype=torch.float64)
        if self._sum_squares[prefix] is None:
            self._sum_squares[prefix] = sum_squares
        else:
            self._sum_squares[prefix].add_(sum_squares)
        self._calls[prefix] += 1

    def finalize(self):
        activation_rms = OrderedDict()
        for prefix in self.prefixes:
            if self._calls[prefix] != self.input_count:
                raise RuntimeError(
                    f"Incomplete activation capture for {prefix}: "
                    f"{self._calls[prefix]}/{self.input_count} inputs"
                )
            sum_squares = self._sum_squares[prefix]
            if sum_squares is None:
                raise AssertionError(f"Missing activation statistics for {prefix}")
            activation_rms[prefix] = (sum_squares / self.total_tokens).sqrt().float()
        return activation_rms


def mlp_neuron_importance(sampled_inputs, fc1_weight, fc1_bias, fc2_weight):
    """Compute the specified activation-output importance for every neuron."""

    tensors = (sampled_inputs, fc1_weight, fc2_weight)
    if not all(isinstance(tensor, torch.Tensor) for tensor in tensors):
        raise TypeError("sampled_inputs, fc1_weight, and fc2_weight must be tensors")
    if sampled_inputs.ndim != 2 or fc1_weight.ndim != 2 or fc2_weight.ndim != 2:
        raise ValueError("MLP importance expects rank-two input and weight tensors")
    hidden, input_width = fc1_weight.shape
    if sampled_inputs.shape[1] != input_width:
        raise ValueError("Sampled MLP input width does not match fc1 input width")
    if fc2_weight.shape[1] != hidden:
        raise ValueError("fc2 columns do not pair one-to-one with fc1 rows")
    if fc1_bias is not None:
        if not isinstance(fc1_bias, torch.Tensor) or tuple(fc1_bias.shape) != (hidden,):
            raise ValueError("fc1 bias does not pair one-to-one with fc1 rows")
        bias = fc1_bias.detach().to(device="cpu", dtype=torch.float32)
    else:
        bias = None

    inputs = sampled_inputs.detach().to(device="cpu", dtype=torch.float32)
    weight1 = fc1_weight.detach().to(device="cpu", dtype=torch.float32)
    weight2 = fc2_weight.detach().to(device="cpu", dtype=torch.float32)
    activated = F.gelu(F.linear(inputs, weight1, bias), approximate="none")
    activation_rms = activated.square().mean(dim=0).sqrt()
    output_norm = torch.linalg.vector_norm(weight2, ord=2, dim=0)
    importance = activation_rms * output_norm
    if not torch.isfinite(importance).all():
        raise ValueError("Non-finite activation-aware MLP importance")
    return importance


def importance_from_activation_rms(activation_rms, fc2_weight):
    """Finish the importance formula from streamed activation RMS statistics."""

    if not isinstance(activation_rms, torch.Tensor) or activation_rms.ndim != 1:
        raise TypeError("activation_rms must be a rank-one tensor")
    if not isinstance(fc2_weight, torch.Tensor) or fc2_weight.ndim != 2:
        raise TypeError("fc2_weight must be a rank-two tensor")
    if fc2_weight.shape[1] != activation_rms.numel():
        raise ValueError("Activation RMS neurons do not pair with fc2 columns")
    output_norm = torch.linalg.vector_norm(
        fc2_weight.detach().to(device="cpu", dtype=torch.float32), ord=2, dim=0
    )
    importance = activation_rms.detach().to(device="cpu", dtype=torch.float32) * output_norm
    if not torch.isfinite(importance).all():
        raise ValueError("Non-finite activation-aware MLP importance")
    return importance


def deterministic_top_indices(scores, count=SELECTED_NEURON_COUNT):
    """Select by descending score, break ties by source index, return sorted ids."""

    if not isinstance(scores, torch.Tensor) or scores.ndim != 1:
        raise TypeError("Neuron scores must be a rank-one tensor")
    count = int(count)
    if count <= 0 or count > scores.numel():
        raise ValueError(f"Cannot select {count} neurons from {scores.numel()}")
    cpu_scores = scores.detach().to(device="cpu", dtype=torch.float64)
    ranking = sorted(range(cpu_scores.numel()), key=lambda index: (-float(cpu_scores[index]), index))
    return torch.tensor(sorted(ranking[:count]), dtype=torch.long)


def clean_state_dict(state):
    if not isinstance(state, dict):
        raise TypeError("Checkpoint must contain a tensor state dict")
    cleaned = OrderedDict()
    for key, value in state.items():
        key = str(key)
        clean_key = key[len("module.") :] if key.startswith("module.") else key
        if clean_key in cleaned:
            raise ValueError(f"Duplicate state key after module-prefix removal: {clean_key}")
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"State value is not a tensor: {key}")
        cleaned[clean_key] = value
    return cleaned


def reject_quantized_source(checkpoint, state):
    """Reject weight-only/packed checkpoints rather than dequantizing silently."""

    if not isinstance(checkpoint, dict):
        raise TypeError("Checkpoint payload must be a mapping")
    forbidden_metadata = sorted(
        key for key in ("quantization", "int4_storage") if key in checkpoint
    )
    quantized_keys = []
    for key, value in state.items():
        lower = str(key).lower()
        if (
            lower.endswith("_scale")
            or ".int4_" in lower
            or value.dtype in {torch.int8, torch.uint8}
            or bool(getattr(value, "is_quantized", False))
        ):
            quantized_keys.append(str(key))
    if forbidden_metadata or quantized_keys:
        raise ValueError(
            "SelectiveMLP-96 source must be complete unquantized weights: "
            f"metadata={forbidden_metadata}, tensors={quantized_keys[:5]}"
        )


def validate_complete_source(source_state, expected_state, label="source"):
    """Require an exact, full state structure; compact/partial inputs are refused."""

    source_keys = set(source_state)
    expected_keys = set(expected_state)
    missing = sorted(expected_keys - source_keys)
    unexpected = sorted(source_keys - expected_keys)
    mismatched = sorted(
        key
        for key in expected_keys & source_keys
        if tuple(source_state[key].shape) != tuple(expected_state[key].shape)
    )
    if missing or unexpected or mismatched:
        raise ValueError(
            f"Incomplete {label} state structure: missing={missing[:5]}, "
            f"unexpected={unexpected[:5]}, shape_mismatch={mismatched[:5]}"
        )


def _resolve_alias_key(state, canonical_key):
    if canonical_key in state:
        return canonical_key
    lower_alias = canonical_key.replace(".Fuser.", ".fuser.")
    if lower_alias in state:
        return lower_alias
    raise KeyError(f"State is missing required tensor {canonical_key}")


def _selected_tensor(source, indices, dimension):
    return torch.index_select(source, dimension, indices.to(source.device)).clone()


def initialize_selective_mlp_state(
    source_state,
    target_state,
    activation_rms,
    *,
    prefixes=SELECTIVE_MLP_PREFIXES,
    selected_count=SELECTED_NEURON_COUNT,
):
    """Return a strictly covered state plus the selected source neuron ids."""

    source_state = clean_state_dict(source_state)
    target_state = clean_state_dict(target_state)
    prefixes = tuple(canonical_state_key(prefix) for prefix in prefixes)
    statistics = {
        canonical_state_key(prefix): value for prefix, value in activation_rms.items()
    }
    if set(statistics) != set(prefixes):
        raise ValueError(
            "Activation statistics do not exactly cover the SelectiveMLP schedule: "
            f"missing={sorted(set(prefixes) - set(statistics))}, "
            f"unexpected={sorted(set(statistics) - set(prefixes))}"
        )

    neuron_indices = OrderedDict()
    for prefix in prefixes:
        fc2_weight = source_state[_resolve_alias_key(source_state, prefix + ".fc2.weight")]
        rms = statistics[prefix]
        scores = importance_from_activation_rms(rms, fc2_weight)
        neuron_indices[prefix] = deterministic_top_indices(scores, selected_count)

    migrated = OrderedDict()
    source_key_map = OrderedDict()
    for target_key, target_tensor in target_state.items():
        canonical_key = canonical_state_key(target_key)
        source_key = target_key if target_key in source_state else _resolve_alias_key(
            source_state, canonical_key
        )
        source_tensor = source_state[source_key]
        selected_prefix = next(
            (prefix for prefix in prefixes if canonical_key.startswith(prefix + ".")),
            None,
        )
        suffix = canonical_key[len(selected_prefix) + 1 :] if selected_prefix else ""
        if selected_prefix is not None and suffix == "fc1.weight":
            value = _selected_tensor(source_tensor, neuron_indices[selected_prefix], 0)
        elif selected_prefix is not None and suffix == "fc1.bias":
            value = _selected_tensor(source_tensor, neuron_indices[selected_prefix], 0)
        elif selected_prefix is not None and suffix == "fc2.weight":
            value = _selected_tensor(source_tensor, neuron_indices[selected_prefix], 1)
        else:
            if tuple(source_tensor.shape) != tuple(target_tensor.shape):
                raise ValueError(
                    f"Untouched tensor shape differs for {target_key}: "
                    f"{tuple(source_tensor.shape)} != {tuple(target_tensor.shape)}"
                )
            value = source_tensor.detach().clone()
        if tuple(value.shape) != tuple(target_tensor.shape):
            raise ValueError(
                f"Paired MLP migration shape mismatch for {target_key}: "
                f"{tuple(value.shape)} != {tuple(target_tensor.shape)}"
            )
        migrated[target_key] = value
        source_key_map[target_key] = source_key

    if set(migrated) != set(target_state) or len(migrated) != len(target_state):
        raise RuntimeError("Strict target state coverage failed")
    for key, target_tensor in target_state.items():
        canonical_key = canonical_state_key(key)
        is_selected_shape = any(
            canonical_key in {
                prefix + ".fc1.weight",
                prefix + ".fc1.bias",
                prefix + ".fc2.weight",
            }
            for prefix in prefixes
        )
        if not is_selected_shape:
            source_key = source_key_map[key]
            if not torch.equal(migrated[key], source_state[source_key]):
                raise AssertionError(f"Untouched tensor was not copied exactly: {key}")
    return migrated, neuron_indices, source_key_map


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def state_dict_sha256(state):
    """Hash tensor names, dtypes, shapes, and logical bytes in canonical order."""

    digest = hashlib.sha256()
    for key in sorted(state):
        tensor = state[key]
        if not isinstance(tensor, torch.Tensor) or tensor.is_sparse:
            raise TypeError(f"State hash requires a dense tensor: {key}")
        cpu = tensor.detach().cpu().contiguous()
        header = f"{key}\0{cpu.dtype}\0{tuple(cpu.shape)}\0".encode("utf-8")
        digest.update(len(header).to_bytes(8, "little"))
        digest.update(header)
        if cpu.numel():
            digest.update(cpu.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def build_initialization_metadata(
    *,
    source_sha256,
    teacher_sha256,
    initialized_state,
    neuron_indices,
    train_indices,
    state_key_count,
    parameter_key_count,
):
    for label, value in (
        ("source_sha256", source_sha256),
        ("teacher_sha256", teacher_sha256),
    ):
        if not isinstance(value, str) or SHA256_PATTERN.fullmatch(value) is None:
            raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    train_indices = [int(index) for index in train_indices]
    if len(train_indices) != OFFICIAL_TRAIN_INPUT_COUNT:
        raise ValueError("Metadata must record exactly 32 official-train inputs")
    normalized_indices = OrderedDict(
        (
            canonical_state_key(prefix),
            [int(index) for index in indices.detach().cpu().tolist()],
        )
        for prefix, indices in neuron_indices.items()
    )
    if set(normalized_indices) != set(SELECTIVE_MLP_PREFIXES):
        raise ValueError("Metadata neuron indices do not cover the fixed 11-MLP schedule")
    if any(len(indices) != SELECTED_NEURON_COUNT for indices in normalized_indices.values()):
        raise ValueError("Every selective MLP must record exactly 384 neuron indices")
    state_key_count = int(state_key_count)
    parameter_key_count = int(parameter_key_count)
    return {
        "method": INITIALIZATION_METHOD,
        "profile_name": PROFILE_NAME,
        "human_label": HUMAN_LABEL,
        "source": "full_depth_ratio4_pruned96",
        "teacher": "official_full192",
        "source_sha256": source_sha256,
        "teacher_sha256": teacher_sha256,
        # This is the non-circular logical state hash, not the outer .pth bytes.
        "init_sha256": state_dict_sha256(initialized_state),
        "mlp_ratio_blocks": TARGET_MLP_RATIO_BLOCKS,
        "neuron_indices": normalized_indices,
        "activation_calibration": {
            "dataset": "official_train",
            "input_indices": train_indices,
            "input_count": OFFICIAL_TRAIN_INPUT_COUNT,
            "tokens_per_input": TOKENS_PER_INPUT,
            "tokens_per_mlp": TOTAL_SAMPLED_TOKENS,
            "seed": TOKEN_SAMPLE_SEED,
            "importance": IMPORTANCE_FORMULA,
            "aggregation": "streamed_float64_sum_of_squares",
        },
        "state_tensor_keys": state_key_count,
        "covered_state_tensor_keys": state_key_count,
        "parameter_state_keys": parameter_key_count,
        "covered_parameter_state_keys": parameter_key_count,
        "strict_coverage": True,
        "random_initialized_parameters": 0,
    }


def _named_modules_with_aliases(model):
    try:
        iterator = model.named_modules(remove_duplicate=False)
    except TypeError:  # Older torch versions do not expose remove_duplicate.
        iterator = model.named_modules()
    modules = {}
    for name, module in iterator:
        canonical = canonical_state_key(name)
        existing = modules.get(canonical)
        if existing is not None and existing is not module:
            raise ValueError(f"Conflicting model modules canonicalize to {canonical}")
        modules[canonical] = module
    return modules


def force_single_call_activation_capture(model):
    """Make each selected deep MLP expose one complete activation tensor.

    The inference builder enables chunked MLPs by default.  At patch8 the deep
    stages contain 33120 tokens, so the default 32768 chunk emits two ``fc1``
    hook calls (32768 + 352) for one input.  The sampling contract is defined
    over the complete input, not independently over runtime chunks.  Setting
    only the eleven selected blocks to a non-positive chunk size preserves the
    memory-saving forward everywhere else and makes the existing uniform
    4096-of-33120 sampling exact.
    """

    modules = _named_modules_with_aliases(model)
    for prefix in SELECTIVE_MLP_PREFIXES:
        block_name = prefix.rsplit(".mlp", 1)[0]
        block = modules.get(block_name)
        if block is None:
            raise KeyError(f"Source model has no scheduled block {block_name}")
        mlp = modules.get(prefix)
        if mlp is None or getattr(block, "mlp", None) is not mlp:
            raise KeyError(f"Source model has no scheduled MLP {prefix}")
        # ``_forward_chunked_mlp_block`` interprets <=0 as the full token count.
        # Native OneScience forwards ignore this private runtime-only attribute.
        block._pangu_mlp_chunk_size = 0


def capture_official_train_mlp_activation_rms(
    model, dataset, train_indices, input_builder
):
    """Run 32 source forwards and stream sampled GELU(fc1(x)) RMS."""

    if [int(index) for index in train_indices] != official_train_input_indices(len(dataset)):
        raise ValueError("Capture must use the fixed 32 evenly spaced official-train indices")
    force_single_call_activation_capture(model)
    collector = ActivationRMSCollector()
    modules = _named_modules_with_aliases(model)
    handles = []
    for prefix in SELECTIVE_MLP_PREFIXES:
        module_name = prefix + ".fc1"
        if module_name not in modules:
            raise KeyError(f"Source model has no scheduled module {module_name}")

        def hook(_module, _inputs, output, selected_prefix=prefix):
            collector.add(selected_prefix, output)

        handles.append(modules[module_name].register_forward_hook(hook))
    try:
        with torch.inference_mode():
            for index in train_indices:
                model(input_builder(dataset[int(index)]))
    finally:
        for handle in handles:
            handle.remove()
    return collector.finalize()


def _load_checkpoint(path, label):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"{label} checkpoint not found: {path}")
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict):
        raise TypeError(f"{label} checkpoint must be a mapping")
    state = clean_state_dict(checkpoint.get("model_state_dict", checkpoint))
    reject_quantized_source(checkpoint, state)
    return checkpoint, state


def _build_model(build_pangu_model, profile):
    return build_pangu_model(
        img_size=[721, 1440],
        patch_size=profile["patch_size"],
        embed_dim=profile["embed_dim"],
        num_heads=profile["num_heads"],
        window_size=profile["window_size"],
        depth_blocks=profile["depth_blocks"],
        mlp_ratio_blocks=profile["mlp_ratio_blocks"],
        use_swiglu=False,
        use_rmsnorm=False,
        use_gqa=False,
        share_deep_blocks=False,
        chunked_attention=False,
    )


def _profile(patch_size, embed_dim, num_heads, depths, mlp_ratio_blocks):
    return {
        "patch_size": list(patch_size),
        "embed_dim": int(embed_dim),
        "num_heads": list(num_heads),
        "depth_blocks": list(depths),
        "window_size": list(SOURCE_WINDOW_SIZE),
        "mlp_ratio_blocks": [list(stage) for stage in mlp_ratio_blocks],
    }


def _source_float_dtype(state):
    dtypes = {
        tensor.dtype
        for key, tensor in state.items()
        if key.endswith(".weight") and tensor.is_floating_point()
    }
    if len(dtypes) != 1:
        raise ValueError(f"Source weights must use one floating dtype, got {sorted(map(str, dtypes))}")
    dtype = next(iter(dtypes))
    if dtype not in {torch.float16, torch.float32, torch.bfloat16}:
        raise ValueError(f"Unsupported source floating dtype: {dtype}")
    return dtype


def _surface_mask(cfg_data, device, dtype):
    import numpy as np

    static_dir = Path(cfg_data.dataset.static_dir)
    land = torch.from_numpy(np.load(static_dir / "land_mask.npy").astype("float32"))
    soil = torch.from_numpy(np.load(static_dir / "soil_type.npy").astype("float32"))
    topo = torch.from_numpy(np.load(static_dir / "topography.npy").astype("float32"))
    topo = (topo - topo.mean()) / (topo.std(unbiased=False) + 1e-6)
    return torch.stack((land, soil, topo)).to(device=device, dtype=dtype)


def _official_input_builder(surface_mask, device, dtype):
    def build(sample):
        if not isinstance(sample, (tuple, list)) or not sample:
            raise TypeError("Official training sample must contain an input tensor")
        raw = torch.as_tensor(sample[0])
        if raw.ndim == 3:
            raw = raw.unsqueeze(0)
        if raw.ndim != 4 or raw.shape[1] != 69:
            raise ValueError(f"Expected official [B,69,H,W] input, got {tuple(raw.shape)}")
        raw = raw.to(device=device, dtype=dtype)
        mask = surface_mask.unsqueeze(0).expand(raw.shape[0], -1, -1, -1)
        return torch.cat((raw[:, :4], mask, raw[:, 4:]), dim=1)

    return build


def _parameter_state_keys(model):
    try:
        iterator = model.named_parameters(remove_duplicate=False)
    except TypeError:
        iterator = model.named_parameters()
    return [name for name, _parameter in iterator]


def atomic_save(payload, output_path):
    output_path = Path(output_path)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    if output_path.exists() or temporary.exists():
        raise FileExistsError(f"Refusing to overwrite output or temporary file: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        torch.save(payload, temporary)
        os.replace(temporary, output_path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return sha256_file(output_path)


def initialize(args):
    """CLI path: audit sources, collect official activations, and save one init."""

    sys.path.insert(0, str(PANGU_DIR))
    from onescience.datapipes.climate import ERA5Datapipe
    from onescience.utils.YParams import YParams
    from pangu_profile_model import build_pangu_model, selective_mlp_96_profile

    source_path = Path(args.source)
    teacher_path = Path(args.teacher)
    _source_checkpoint, source_state = _load_checkpoint(source_path, "pruned96 source")
    _teacher_checkpoint, teacher_state = _load_checkpoint(teacher_path, "official teacher")

    teacher_profile = _profile(
        [2, 4, 4], 192, [6, 12, 12, 6], SOURCE_DEPTHS, SOURCE_MLP_RATIO_BLOCKS
    )
    teacher_model = _build_model(build_pangu_model, teacher_profile)
    validate_complete_source(teacher_state, teacher_model.state_dict(), "official teacher")
    del teacher_state, teacher_model, _teacher_checkpoint

    source_profile = _profile(
        SOURCE_PATCH_SIZE,
        SOURCE_EMBED_DIM,
        SOURCE_NUM_HEADS,
        SOURCE_DEPTHS,
        SOURCE_MLP_RATIO_BLOCKS,
    )
    source_model = _build_model(build_pangu_model, source_profile)
    validate_complete_source(source_state, source_model.state_dict(), "pruned96 source")
    source_dtype = _source_float_dtype(source_state)
    device = torch.device(args.device)
    source_model.to(device=device, dtype=source_dtype).eval()
    source_model.load_state_dict(source_state, strict=True)

    cfg_data = YParams(args.config, "datapipe")
    datapipe = ERA5Datapipe(params=cfg_data, distributed=False)
    train_loader, _sampler = datapipe.train_dataloader()
    train_indices = official_train_input_indices(len(train_loader.dataset))
    mask = _surface_mask(cfg_data, device, source_dtype)
    activation_rms = capture_official_train_mlp_activation_rms(
        source_model,
        train_loader.dataset,
        train_indices,
        _official_input_builder(mask, device, source_dtype),
    )
    source_model.to("cpu")
    del source_model, mask, datapipe, train_loader
    if device.type == "cuda":
        torch.cuda.empty_cache()

    target_profile = selective_mlp_96_profile()
    if target_profile.get("name") != PROFILE_NAME:
        raise ValueError(f"Project builder returned the wrong profile: {target_profile}")
    target_model = _build_model(build_pangu_model, target_profile)
    parameter_count = sum(parameter.numel() for parameter in target_model.parameters())
    if parameter_count != EXPECTED_PARAMETER_COUNT:
        raise ValueError(
            f"SelectiveMLP-96 parameter count mismatch: "
            f"{parameter_count} != {EXPECTED_PARAMETER_COUNT}"
        )
    target_state = target_model.state_dict()
    parameter_keys = _parameter_state_keys(target_model)
    initialized_state, neuron_indices, _source_key_map = initialize_selective_mlp_state(
        source_state, target_state, activation_rms
    )
    target_model.to(dtype=source_dtype)
    target_model.load_state_dict(initialized_state, strict=True)

    metadata = build_initialization_metadata(
        source_sha256=sha256_file(source_path),
        teacher_sha256=sha256_file(teacher_path),
        initialized_state=initialized_state,
        neuron_indices=neuron_indices,
        train_indices=train_indices,
        state_key_count=len(target_state),
        parameter_key_count=len(parameter_keys),
    )
    payload = {
        "model_state_dict": initialized_state,
        "model_profile": target_profile,
        "initialization": metadata,
    }
    output_sha = atomic_save(payload, args.output)
    saved = torch.load(args.output, map_location="cpu", weights_only=False)
    if state_dict_sha256(saved["model_state_dict"]) != metadata["init_sha256"]:
        raise RuntimeError("Saved SelectiveMLP-96 state hash verification failed")
    print(f"profile={PROFILE_NAME}")
    print(f"method={INITIALIZATION_METHOD}")
    print(f"parameters={parameter_count}")
    print(f"state_coverage={len(target_state)}/{len(target_state)}")
    print("random_initialized_parameters=0")
    print(f"source_sha256={metadata['source_sha256']}")
    print(f"teacher_sha256={metadata['teacher_sha256']}")
    print(f"init_sha256={metadata['init_sha256']}")
    print(f"checkpoint_sha256={output_sha}")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, help="Unquantized full-depth pruned96 .pth")
    parser.add_argument("--teacher", required=True, help="Official full192 model_bak.pth")
    parser.add_argument("--output", required=True, help="New SelectiveMLP-96 init .pth")
    parser.add_argument("--config", default=str(PANGU_DIR / "conf" / "config.yaml"))
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args(argv)


if __name__ == "__main__":
    initialize(parse_args())
