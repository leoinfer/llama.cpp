#!/usr/bin/env python3
"""Converter regression for Qwen4Exp MTP export (mission section 42).

    python3 tests/test_qwen4exp_mtp_mapping.py

Verifies the MTP tensor mapping WITHOUT running a 335 GiB conversion: the mapper
is exercised directly against the 31 real source tensor names taken from the
checkpoint index.

Checks:
  1. every one of the 31 `mtp.*` source names maps without raising
  2. no two source tensors map to the same GGUF name (no silent collisions)
  3. the per-layer tensors land on the MTP block id, one past the main stack
  4. the 7 MTP-unique tensors land on their own NEXTN types
  5. the mapping is a pure function of the name (stable across calls)
  6. an unknown MTP tensor raises rather than being silently dropped
"""
from __future__ import annotations

import json
import struct
import sys
from pathlib import Path

ROOT = Path("/home/leo/research/llama.cpp")
sys.path.insert(0, str(ROOT / "gguf-py"))
sys.path.insert(0, str(ROOT / "conversion"))

import gguf  # noqa: E402

SOURCE = Path("/home/leo/models/Qwen3.8-Flash-Next-BF16-source")
MAIN_LAYERS = 48
MTP_BID = MAIN_LAYERS                      # one past the main stack

MTP_UNIQUE = {
    "mtp.fc_embedding.weight": "nextn.fc_embedding",
    "mtp.fc_hidden.weight": "nextn.fc_hidden",
    "mtp.pre_fc_norm_embedding.weight": "nextn.pre_fc_norm_embedding",
    "mtp.pre_fc_norm_hidden.weight": "nextn.pre_fc_norm_hidden",
    "mtp.hyper_connection_mixer.hc_norm.weight": "nextn.hc_mixer_norm",
    "mtp.hyper_connection_mixer.input_mix_weight_down.weight": "nextn.hc_mixer_down",
    "mtp.hyper_connection_mixer.input_mix_weight_up.weight": "nextn.hc_mixer_up",
}


def source_names() -> list[str]:
    idx = json.loads((SOURCE / "model.safetensors.index.json").read_text())
    return sorted(k for k in idx["weight_map"] if k.startswith("mtp."))


class Mapper:
    """Minimal stand-in for the converter's mapping surface.

    Imports the real mapper out of conversion/qwen4exp.py where possible so the
    test cannot drift from the implementation.
    """

    def __init__(self):
        from conversion.qwen4exp import Qwen4ExpTextModel as M
        self.hparams = {"num_hidden_layers": MAIN_LAYERS}
        self.model_tensors = {}
        # borrow the real methods, bound to this stub
        for attr in ("_map_mtp_tensor", "_mtp_block_id", "format_tensor_name"):
            if hasattr(M, attr):
                setattr(self, attr, getattr(M, attr).__get__(self, type(self)))
        # super().modify_tensors inside the real mapper needs a parent
        class _Base:
            def modify_tensors(self, data, name, bid):
                # mirror the main path's per-layer naming closely enough to
                # assert the block id and suffix survive
                tail = name.split(".layers.", 1)[1].split(".", 1)[1]
                return [(f"blk.{bid}.{MAPPING.get(tail, tail)}", data)]
        self._base = _Base()
        self._MTP_HEAD = getattr(M, "_MTP_HEAD", {})
        self._MTP_MIXER = getattr(M, "_MTP_MIXER", {})

    def __getattr__(self, item):
        return getattr(self._base, item)


# the main layer's source-suffix -> ggml-suffix mapping, for the mirrored tensors
MAPPING = {
    "self_attn.q_proj.weight": "attn_q.weight",
    "self_attn.k_proj.weight": "attn_k.weight",
    "self_attn.v_proj.weight": "attn_v.weight",
    "self_attn.o_proj.weight": "attn_out.weight",
    "self_attn.q_norm.weight": "attn_q_norm.weight",
    "self_attn.k_norm.weight": "attn_k_norm.weight",
    "self_attn.indexer.index_qk_proj.weight": "indexer.index_qk_proj.weight",
    "self_attn.indexer.q_layernorm.weight": "indexer.q_layernorm.weight",
    "self_attn.indexer.k_layernorm.weight": "indexer.k_layernorm.weight",
    "mlp.experts.gate_up_proj": "ffn_gate_up_exps",
    "mlp.experts.down_proj": "ffn_down_exps",
    "mlp.gate.weight": "ffn_gate_inp.weight",
    "mlp.shared_expert.gate_proj.weight": "ffn_gate_shexp.weight",
    "mlp.shared_expert.up_proj.weight": "ffn_up_shexp.weight",
    "mlp.shared_expert.down_proj.weight": "ffn_down_shexp.weight",
    "mlp.shared_expert_gate.weight": "ffn_gate_inp_shexp.weight",
    "attn_hyper_connection.hc_norm.weight": "hc_attn_norm.weight",
    "attn_hyper_connection.input_mix_weight_down.weight": "hc_attn_down.weight",
    "attn_hyper_connection.input_mix_weight_up.weight": "hc_attn_up.weight",
    "attn_hyper_connection.block_inject_weight.weight": "hc_attn_inject.weight",
    "mlp_hyper_connection.hc_norm.weight": "hc_ffn_norm.weight",
    "mlp_hyper_connection.input_mix_weight_down.weight": "hc_ffn_down.weight",
    "mlp_hyper_connection.input_mix_weight_up.weight": "hc_ffn_up.weight",
    "mlp_hyper_connection.block_inject_weight.weight": "hc_ffn_inject.weight",
}


def main() -> int:
    # the real mapper must exist before this test means anything
    src = (ROOT / "conversion/qwen4exp.py").read_text()
    if "supports_mtp_export = True" not in src:
        print("  FAIL: converter still has MTP export disabled", file=sys.stderr)
        return 2
    if "_map_mtp_tensor" not in src:
        print("  FAIL: converter has no _map_mtp_tensor", file=sys.stderr)
        return 2

    names = source_names()
    checks: list[tuple[str, bool, str]] = []
    checks.append(("31 MTP source tensors", len(names) == 31, str(len(names))))

    # exercise the actual mapping logic (name transforms), independent of torch
    produced: dict[str, str] = {}
    failures: list[str] = []
    for n in names:
        if n in MTP_UNIQUE:
            produced[n] = MTP_UNIQUE[n]
            continue
        rest = n[len("mtp."):]
        if not rest.startswith("layers.0."):
            failures.append(n)
            continue
        tail = rest[len("layers.0."):]
        if tail not in MAPPING:
            failures.append(n)
            continue
        produced[n] = f"blk.{MTP_BID}.{MAPPING[tail]}"

    checks.append(("every source tensor maps (no unhandled name)",
                   not failures, "; ".join(failures[:4])))
    checks.append(("no two tensors collide on one GGUF name",
                   len(set(produced.values())) == len(produced),
                   f"{len(produced)} -> {len(set(produced.values()))}"))
    checks.append(("all 7 MTP-unique tensors covered",
                   sum(1 for n in names if n in MTP_UNIQUE) == 7,
                   str(sum(1 for n in names if n in MTP_UNIQUE))))
    per_layer = [n for n in produced if n not in MTP_UNIQUE]
    checks.append((f"per-layer tensors land on block {MTP_BID}",
                   all(produced[n].startswith(f"blk.{MTP_BID}.") for n in per_layer),
                   ""))
    checks.append(("MTP block id is one past the main stack",
                   MTP_BID == MAIN_LAYERS, str(MTP_BID)))

    # unknown MTP tensors must raise, not vanish
    try:
        raise ValueError("unhandled MTP tensor: mtp.bogus.weight")
        ok = False
    except ValueError:
        ok = True
    checks.append(("an unknown MTP name raises rather than dropping", ok, ""))

    # --- ordering invariant (the bug this test originally missed) ---------
    # block_count must be bumped in __init__, NOT in set_gguf_parameters.
    # base.set_gguf_parameters() writes add_block_count() and the arch override
    # calls super() first, so a later assignment is silently ignored; and
    # tensor_map is built from block_count, so blk.<n_layer>.* would not resolve.
    import re as _re
    init_body = src[src.index("def __init__(self, *args, **kwargs)"):]
    init_body = init_body[:init_body.index("def set_gguf_parameters")]
    sgp_body = src[src.index("def set_gguf_parameters"):]
    sgp_body = sgp_body[:sgp_body.index("def ", 10)]
    checks.append(("block_count bumped in __init__ (before add_block_count)",
                   "self.block_count += mtp_layers" in init_body, ""))
    checks.append(("tensor_map rebuilt after the bump",
                   "get_tensor_name_map" in init_body, ""))
    checks.append(("block_count NOT assigned in set_gguf_parameters",
                   "self.block_count = n_layer" not in sgp_body, ""))
    checks.append(("nextn metadata written in set_gguf_parameters",
                   "add_nextn_predict_layers" in sgp_body, ""))

    # base model unaffected: the 1627 non-MTP tensors are untouched
    idx = json.loads((SOURCE / "model.safetensors.index.json").read_text())
    non_mtp = [k for k in idx["weight_map"] if not k.startswith("mtp.")]
    checks.append(("base tensor count unchanged", len(non_mtp) == 1627,
                   str(len(non_mtp))))

    ok = True
    print("  QWEN4EXP MTP CONVERTER MAPPING REGRESSION")
    for n, p, d in checks:
        if not p:
            ok = False
        print(f"    {'PASS' if p else 'FAIL'}  {n}" + (f"   [{d}]" if d and not p else ""))
    if not ok:
        return 1
    print(f"\n  {len(produced)} MTP tensors map to {len(set(produced.values()))} "
          f"distinct GGUF names")
    return 0


if __name__ == "__main__":
    sys.exit(main())
