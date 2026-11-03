import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file, save_file

from qwenputer_handover.training.export import ExportError, export

Q = "model.language_model.layers.0.self_attn.q_proj.weight"
UP = "model.language_model.layers.0.mlp.up_proj.weight"
NORM = "model.language_model.layers.0.input_layernorm.weight"
VISUAL = "model.visual.blocks.0.attn.qkv.weight"


def make_base(root: Path) -> Path:
    base = root / "base"
    base.mkdir()
    save_file(
        {Q: torch.eye(2, dtype=torch.bfloat16), NORM: torch.ones(2, dtype=torch.bfloat16)},
        base / "model-00001-of-00002.safetensors",
        metadata={"format": "pt"},
    )
    save_file(
        {
            UP: torch.zeros(3, 2, dtype=torch.bfloat16),
            VISUAL: torch.full((2, 2), 7.0, dtype=torch.bfloat16),
        },
        base / "model-00002-of-00002.safetensors",
        metadata={"format": "pt"},
    )
    index = {
        "metadata": {},
        "weight_map": {
            Q: "model-00001-of-00002.safetensors",
            NORM: "model-00001-of-00002.safetensors",
            UP: "model-00002-of-00002.safetensors",
            VISUAL: "model-00002-of-00002.safetensors",
        },
    }
    (base / "model.safetensors.index.json").write_text(json.dumps(index))
    (base / "config.json").write_text('{"model_type": "qwen3_5"}')
    (base / "tokenizer.json").write_text("{}")
    return base


def make_adapter(
    root: Path,
    cfg: dict[str, object],
    modules: dict[str, tuple[list[list[float]], list[list[float]]]],
) -> Path:
    adapter = root / "adapter"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text(
        json.dumps({"peft_type": "LORA", "r": 1, "lora_alpha": 2, **cfg})
    )
    tensors = {}
    for m, (a, b) in modules.items():
        tensors[f"base_model.model.{m}.lora_A.weight"] = torch.tensor(a)
        tensors[f"base_model.model.{m}.lora_B.weight"] = torch.tensor(b)
    save_file(tensors, adapter / "adapter_model.safetensors")
    return adapter


def test_adapted_weights_get_w_plus_scaled_ba(tmp_path: Path) -> None:
    base = make_base(tmp_path)
    adapter = make_adapter(
        tmp_path,
        {},
        {
            Q.removesuffix(".weight"): ([[1.0, 2.0]], [[0.5], [0.25]]),  # A (1x2), B (2x1)
        },
    )
    keys = export(base, adapter, tmp_path / "out")
    assert keys == [Q]
    merged = load_file(tmp_path / "out" / "model-00001-of-00002.safetensors")
    # scale = alpha / r = 2; B @ A = [[0.5, 1], [0.25, 0.5]]
    assert merged[Q].dtype == torch.bfloat16
    assert merged[Q].float().tolist() == [[2.0, 2.0], [0.5, 2.0]]
    assert merged[NORM].float().tolist() == [1.0, 1.0]
    # the untouched shard and every other file are copied byte for byte
    for name in (
        "model-00002-of-00002.safetensors",
        "model.safetensors.index.json",
        "config.json",
        "tokenizer.json",
    ):
        assert (tmp_path / "out" / name).read_bytes() == (base / name).read_bytes()


def test_rslora_scale(tmp_path: Path) -> None:
    base = make_base(tmp_path)
    adapter = make_adapter(
        tmp_path,
        {"r": 4, "lora_alpha": 4, "use_rslora": True},
        {UP.removesuffix(".weight"): ([[1.0, 0.0]] * 4, [[1.0, 0.0, 0.0, 0.0]] * 3)},
    )
    export(base, adapter, tmp_path / "out")
    merged = load_file(tmp_path / "out" / "model-00002-of-00002.safetensors")
    assert merged[UP].float().tolist() == [[2.0, 0.0]] * 3  # alpha / sqrt(r) = 2


def test_key_absent_from_base_is_an_error(tmp_path: Path) -> None:
    base = make_base(tmp_path)
    adapter = make_adapter(
        tmp_path,
        {},
        {"model.language_model.layers.9.mlp.up_proj": ([[1.0, 0.0]], [[1.0], [1.0], [1.0]])},
    )
    with pytest.raises(ExportError, match="not in the base checkpoint"):
        export(base, adapter, tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_dora_is_rejected(tmp_path: Path) -> None:
    base = make_base(tmp_path)
    adapter = make_adapter(tmp_path, {"use_dora": True}, {})
    with pytest.raises(ExportError, match="use_dora"):
        export(base, adapter, tmp_path / "out")
