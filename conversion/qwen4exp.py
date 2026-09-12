from __future__ import annotations

from typing import Iterable, cast

import torch
from torch import Tensor

import gguf
import numpy as np

from .base import ModelBase
from .qwen import _LinearAttentionVReorderBase, _Qwen35MRopeMixin
from .qwen3vl import Qwen3VLVisionModel


@ModelBase.register("Qwen4ExpForConditionalGeneration", "Qwen4ExpForCausalLM")
@ModelBase.example("Qwen/Qwen3.8-Flash-Next")
class Qwen4ExpTextModel(_Qwen35MRopeMixin, _LinearAttentionVReorderBase):
    """Qwen3.8-Flash-Next.

    Shares the Qwen3.5 gated delta net and interleaved mrope, and adds three things:
    hyper-connections in place of every layer norm, QSA sparse attention on the full
    attention layers, and PLE n-gram hash embeddings on a single layer.
    """

    model_arch = gguf.MODEL_ARCH.QWEN4EXP

    # The MTP block is a real inference path, not training-only: vLLM publishes
    # a concrete Qwen4Exp AMD MTP graph. transformers discards the weights via
    # _keys_to_ignore_on_load_unexpected = [r"^mtp.*"], which is why this was
    # previously disabled. Export is opt-in (mtp_only / no_mtp still gate it).
    supports_mtp_export = True
    no_mtp = False

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # MTP: the draft block occupies one extra block id past the main stack.
        # This MUST happen here rather than in set_gguf_parameters, because
        # block_count is consumed twice before that runs:
        #   1. base.set_gguf_parameters() calls add_block_count() with whatever
        #      value is set at that point, and the arch's override calls super()
        #      first -- so a later assignment would be ignored; and
        #   2. tensor_map is built from block_count, so the MTP layer's
        #      blk.<n_layer>.* names would not resolve at all.
        # deepseek.py sets it in __init__ for exactly these reasons.
        mtp_layers = int(self.hparams.get("mtp_num_hidden_layers") or 0)
        if not self.no_mtp and mtp_layers:
            self.block_count += mtp_layers
            self.tensor_map = gguf.get_tensor_name_map(self.model_arch, self.block_count)

        # only the shard names, so the table itself is never held
        self._ple_shards: dict[int, str] = {}
        self._ple_row_dim: int | None = None

    def _read_hash_constants(self, suffix: str) -> list[int]:
        """Read an int64 PLE constant straight from the checkpoint.

        prepare_tensors() casts every non-float dtype to float32 before
        modify_tensors() sees it (base.py), which would silently round these
        45-bit multipliers. Reading the lazy tensor here bypasses that.
        """
        for name, gen in self.model_tensors.items():
            if name.endswith(suffix):
                t = gen()
                if t.dtype != torch.int64:
                    t = t.to(torch.int64)
                return [int(x) for x in t.tolist()]
        raise ValueError(f"PLE constant {suffix!r} missing from the checkpoint")

    def set_gguf_parameters(self):
        super().set_gguf_parameters()
        hp = self.hparams

        self.gguf_writer.add_hyper_connection_count(hp["hc_count"])
        self.gguf_writer.add_hyper_connection_low_rank(hp["hc_lowrank"])

        n_layer = hp["num_hidden_layers"]

        # block_count is already bumped in __init__ (see there for why). Only
        # the metadata key is written here, mirroring the NEXTN convention.
        mtp_layers = hp.get("mtp_num_hidden_layers") or 0
        if not self.no_mtp and mtp_layers:
            self.gguf_writer.add_nextn_predict_layers(int(mtp_layers))

        self.gguf_writer.add_indexer_head_count(hp["indexer_n_heads"])
        self.gguf_writer.add_indexer_key_length(hp["indexer_head_dim"])
        self.gguf_writer.add_indexer_top_k(hp["indexer_budget"])
        ratio = hp["indexer_compress_ratio"]
        layer_types = hp["layer_types"]
        self.gguf_writer.add_attention_compress_ratios(
            [ratio if layer_types[i] == "full_attention" else 0 for i in range(n_layer)]
        )

        # ple_layer_ids is 1-based in the HF config; empty means no n-gram table,
        # so emit no PLE keys rather than optional ones
        ple_layers = [i - 1 for i in hp["ple_layer_ids"]]
        if not ple_layers:
            return
        self.gguf_writer.add_ple_layers(ple_layers)
        self.gguf_writer.add_ple_ngram_size(hp["ngram_size"])
        self.gguf_writer.add_ple_heads_per_ngram(hp["heads_per_ngram"])
        self.gguf_writer.add_ple_conv_kernel(hp["ple_conv_kernel_size"])
        self.gguf_writer.add_ple_eos_token_id(self._eos_token_id())
        # an image is decoded as an embeddings-only batch, so the graph has no placeholder
        # ids to hash; carry the id and let it stand in for those positions
        _img = self._image_token_id()
        if _img is not None:
            self.gguf_writer.add_ple_image_token_id(int(_img))
        if self._ple_row_dim is not None:
            self.gguf_writer.add_embedding_length_per_layer_input(self._ple_row_dim)

        self.gguf_writer.add_ple_layer_multipliers(
            self._read_hash_constants("ple_embedding.layer_multipliers"))
        self.gguf_writer.add_ple_head_offsets(
            self._read_hash_constants("ple_embedding.ngram_heads_offsets"))
        self.gguf_writer.add_ple_head_vocab_sizes(
            self._read_hash_constants("ple_embedding.ngram_heads_vocab_sizes"))

    def _image_token_id(self) -> int | None:
        img = self.hparams.get("image_token_id")
        return None if img is None else int(img)

    def _eos_token_id(self) -> int:
        eos = self.hparams.get("eos_token_id")
        if isinstance(eos, list):
            # the PLE hash resets n-grams on the primary EOS
            return int(eos[-1])
        if eos is None:
            raise ValueError("eos_token_id is required: the PLE hash resets its n-grams on it")
        return int(eos)

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        # int64 hash constants must stay exact; 1-D tensors force F32, so use KV
        if name.endswith("ple_embedding.layer_multipliers"):
            self._ple_multipliers = [int(x) for x in data_torch.tolist()]
            return []
        if name.endswith("ple_embedding.ngram_heads_offsets"):
            self._ple_head_offsets = [int(x) for x in data_torch.tolist()]
            return []
        if name.endswith("ple_embedding.ngram_heads_vocab_sizes"):
            self._ple_head_vocab_sizes = [int(x) for x in data_torch.tolist()]
            return []

        if ".ngram_embedding.shard_" in name:
            return self._place_ple_shard(data_torch, name)

        # --- MTP block -------------------------------------------------
        # The MTP layer mirrors a main decoder layer, so its per-layer tensors
        # reuse the main layer's mapping at the MTP block id. Only the fusion
        # and mixer tensors are unique to MTP.
        if name.startswith("mtp."):
            return self._map_mtp_tensor(data_torch, name, bid)

        # one projection feeds indexer q and k; split it, as minimax-m3 does
        if ".indexer.index_qk_proj.weight" in name:
            n_q = self.hparams["indexer_n_heads"] * self.hparams["indexer_head_dim"]
            q = data_torch[:n_q]
            k = data_torch[n_q:]
            return [
                (self.format_tensor_name(gguf.MODEL_TENSOR.INDEXER_Q_PROJ, bid, ".weight"), q),
                (self.format_tensor_name(gguf.MODEL_TENSOR.INDEXER_K_PROJ, bid, ".weight"), k),
            ]

        # Gemma zero-centred gammas the inherited norm.weight rule misses
        if name.endswith((".ple.norm_key.weight", ".ple.norm_query.weight", ".ple.norm_conv.weight",
                          ".indexer.q_layernorm.weight", ".indexer.k_layernorm.weight")):
            return [(self.map_tensor_name(name), data_torch + 1)]

        if name.endswith(".ple.conv1d.weight"):
            return [(self.map_tensor_name(name), data_torch.squeeze())]

        return super().modify_tensors(data_torch, name, bid)

    def _mtp_block_id(self) -> int:
        """Block id of the MTP layer: one past the main stack."""
        return int(self.hparams["num_hidden_layers"])

    _MTP_MIXER = {
        "hc_norm.weight":        "NEXTN_HC_MIXER_NORM",
        "input_mix_weight_down.weight": "NEXTN_HC_MIXER_DOWN",
        "input_mix_weight_up.weight":   "NEXTN_HC_MIXER_UP",
    }
    _MTP_HEAD = {
        "fc_embedding.weight":           "NEXTN_FC_EMBEDDING",
        "fc_hidden.weight":              "NEXTN_FC_HIDDEN",
        "pre_fc_norm_embedding.weight":  "NEXTN_PRE_FC_NORM_EMBEDDING",
        "pre_fc_norm_hidden.weight":     "NEXTN_PRE_FC_NORM_HIDDEN",
    }

    def _map_mtp_tensor(self, data_torch, name: str, bid):
        """Map one `mtp.*` source tensor to its GGUF name."""
        rest = name[len("mtp."):]

        # MTP head-level fusion tensors and the HCMixer
        if rest in self._MTP_HEAD:
            t = getattr(gguf.MODEL_TENSOR, self._MTP_HEAD[rest])
            return [(self.format_tensor_name(t), data_torch)]
        if rest.startswith("hyper_connection_mixer."):
            suffix = rest[len("hyper_connection_mixer."):]
            if suffix not in self._MTP_MIXER:
                raise ValueError(f"unhandled MTP mixer tensor: {name}")
            t = getattr(gguf.MODEL_TENSOR, self._MTP_MIXER[suffix])
            return [(self.format_tensor_name(t), data_torch)]

        # Per-layer tensors: rewrite to the main layer prefix at the MTP block id
        # and reuse the main mapping, so shapes/quirks (indexer split, norm gain,
        # expert stacks) are handled exactly once.
        prefix = "layers.0."
        if not rest.startswith(prefix):
            raise ValueError(f"unhandled MTP tensor: {name}")
        main_name = ("model.language_model.layers."
                     f"{self._mtp_block_id()}." + rest[len(prefix):])
        return super().modify_tensors(data_torch, main_name, self._mtp_block_id())

    # the shards concatenate into a tensor of well over 100 GB
    # use LazyChunkedTensor here, a single shard resident at a time
    def _place_ple_shard(self, data_torch: Tensor, name: str) -> Iterable[tuple[str, Tensor]]:

        idx = int(name.rpartition(".shard_")[2].partition(".")[0])
        n_parts = self.hparams["split_ngram_parts"]

        self._ple_shards[idx] = name
        self._ple_row_dim = int(data_torch.shape[-1])

        if len(self._ple_shards) < n_parts:
            return []

        # the checkpoint may yield the shards in any order, the row order is by index
        shards = [self._ple_shards[i] for i in sorted(self._ple_shards)]
        rows = 0
        for shard in shards:
            shape = self.model_tensors[shard]().shape
            if int(shape[-1]) != self._ple_row_dim:
                raise ValueError(
                    f"PLE shard {shard} has row dim {int(shape[-1])}, expected {self._ple_row_dim}")
            rows += int(shape[0])

        table = gguf.LazyChunkedTensor(
            [self._load_ple_shard(shard) for shard in shards],
            shape=(rows, self._ple_row_dim),
            dtype=np.float32,
        )
        gguf_name = gguf.TENSOR_NAMES[gguf.MODEL_TENSOR.PER_LAYER_TOKEN_EMBD]
        return [(gguf_name + ".weight", cast(Tensor, table))]

    def _load_ple_shard(self, name: str):
        def load() -> np.ndarray:
            from .base import LazyTorchTensor

            # a fresh lazy tensor every call, or to_eager() memoizes every shard
            eager = LazyTorchTensor.to_eager(self.model_tensors[name]())
            return eager.to(torch.float32).contiguous().numpy()
        return load

    def prepare_tensors(self):
        super().prepare_tensors()
        n_parts = self.hparams.get("split_ngram_parts", 0)
        if self._ple_shards and len(self._ple_shards) != n_parts:
            raise ValueError(
                f"got {len(self._ple_shards)} PLE embedding shards, expected {n_parts}"
            )


@ModelBase.register("Qwen4ExpForConditionalGeneration")
@ModelBase.example("Qwen/Qwen3.8-Flash-Next")
class Qwen4ExpVisionModel(Qwen3VLVisionModel):
    """The vision tower is an unmodified Qwen3-VL ViT."""
