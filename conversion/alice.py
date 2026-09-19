from __future__ import annotations

from typing import Iterable, TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from torch import Tensor

from .base import ModelBase, TextModel, gguf, logger


@ModelBase.register("AliceAIForCausalLM")
class AliceAIModel(TextModel):
    """Yandex AliceAI-Foundation-80B-A3B: hybrid gated-full-attention + split-projection
    KDA (Korean-style Delta Attention path reused via kimi-linear primitives), deep
    sigmoid-router MoE with bias correction + shared expert, and a split
    block-attention-residual (depth-softmax-mix over completed blocks)."""
    model_arch = gguf.MODEL_ARCH.ALICE_AI

    def set_vocab(self):
        self._set_vocab_sentencepiece()

    def set_gguf_parameters(self):
        super().set_gguf_parameters()
        self.gguf_writer.add_block_count(self.hparams["num_hidden_layers"])
        self.gguf_writer.add_context_length(self.hparams["max_position_embeddings"])
        self.gguf_writer.add_embedding_length(self.hparams["hidden_size"])
        self.gguf_writer.add_feed_forward_length(self.hparams.get(
            "intermediate_size", self.hparams["hidden_size"] * 2))

        # hybrid layer marking: honor config layer_types (fall back to the 1-in-4 pattern)
        block = self.hparams["num_hidden_layers"]
        ltypes = self.hparams.get("layer_types")
        if ltypes is not None and len(ltypes) == block:
            recurrent = [t == "linear_attention" for t in ltypes]
        else:
            recurrent = [(i + 1) % 4 != 0 for i in range(block)]
        self.gguf_writer.add_recurrent_layers(recurrent)
        self.gguf_writer.add_full_attention_interval(4)

        # attention geometry
        self.gguf_writer.add_head_count(self.hparams["num_attention_heads"])
        self.gguf_writer.add_head_count_kv(self.hparams["num_key_value_heads"])
        if (rope_dim := self.hparams.get("head_dim")) is None:
            rope_dim = self.hparams["hidden_size"] // self.hparams["num_attention_heads"]
        self.gguf_writer.add_rope_dimension_count(int(rope_dim * self.hparams.get("partial_rotary_factor", 0.25)))
        self.gguf_writer.add_rope_freq_base(self.hparams.get("rope_theta", 1000000.0))

        # KDA geometry
        self.gguf_writer.add_ssm_conv_kernel(self.hparams["linear_conv_kernel_dim"])
        self.gguf_writer.add_kda_head_dim(self.hparams["linear_key_head_dim"])
        # KDA head count drives conv/recurrent state sizing and differs from attn heads
        self.gguf_writer.add_ssm_group_count(self.hparams["linear_num_key_heads"])

        self.gguf_writer.add_expert_count(self.hparams["num_experts"])
        self.gguf_writer.add_expert_used_count(self.hparams["num_experts_per_tok"])
        self.gguf_writer.add_expert_feed_forward_length(self.hparams["moe_intermediate_size"])
        self.gguf_writer.add_expert_shared_count(1)
        self.gguf_writer.add_expert_shared_feed_forward_length(
            self.hparams["shared_expert_intermediate_size"])
        self.gguf_writer.add_expert_gating_func(gguf.ExpertGatingFuncType.SIGMOID)
        self.gguf_writer.add_expert_weights_norm(True)
        self.gguf_writer.add_expert_weights_scale(1.0)

        # block attention residual (depth-softmax-mix over completed blocks)
        self.gguf_writer.add_attn_res_block_size(self.hparams.get("block_attn_res_block_size", 4))

        if (rope_dim := self.hparams.get("head_dim")) is None:
            rope_dim = self.hparams["hidden_size"] // self.hparams["num_attention_heads"]
        self.gguf_writer.add_rope_dimension_count(int(rope_dim * self.hparams.get("partial_rotary_factor", 0.25)))

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        # zero-centred RMSNorm gains: fold +1 at conversion (matches proven converter table)
        if name.endswith((
            "input_layernorm.weight",
            "post_attention_layernorm.weight",
            "self_attn.q_norm.weight",
            "self_attn.k_norm.weight",
            "model.norm.weight",
        )):
            data_torch = data_torch + 1

        # KDA conv weights: HF [C, 1, K] -> numpy (1, C, 1, K) = ggml [K, 1, C, 1]
        if name.endswith((".q_conv1d.weight", ".k_conv1d.weight", ".v_conv1d.weight")):
            if data_torch.ndim == 3:
                c, _, k = data_torch.shape
                data_torch = data_torch.reshape(1, c, 1, k)
            elif data_torch.ndim == 2:
                c, k = data_torch.shape
                data_torch = data_torch.reshape(1, c, 1, k)

        # a_log_bias: converter pre-negates to -exp(a_log), matching llama.cpp ssm_a convention
        if name.endswith(".a_log_bias"):
            data_torch = -torch.exp(data_torch)

        # fused expert tensor -> keep fused (loader creates split + fused handles)
        if name.endswith("mlp.experts.gate_up_proj"):
            import os
            if os.environ.get("ALICE_SPLIT_EXPERTS") == "1":
                # HF layout is [E, 2I, H] with gate in the first half (chunk(2, dim=-1))
                n_ff = data_torch.shape[1] // 2
                base = name.removesuffix(".gate_up_proj")
                yield (self.tensor_name(base + ".gate_proj.weight", bid), data_torch[:, :n_ff].contiguous())
                yield (self.tensor_name(base + ".up_proj.weight",   bid), data_torch[:, n_ff:].contiguous())
                return
            yield (self.format_tensor_name(gguf.MODEL_TENSOR.FFN_GATE_UP_EXP, bid), data_torch)
            return

        yield (self.tensor_name(name, bid), data_torch)

    @classmethod
    def filter_tensors(cls, item):
        name, gen = item
        if name.startswith("mtp."):
            # MTP draft head excluded from the trunk GGUF for bring-up
            return None
        return super().filter_tensors(item)

    def tensor_name(self, name: str, bid: int | None) -> str:
        # tensor_mapping entries carry no suffix; most GGUF tensor names need .weight/.bias,
        # but ssm_a / ssm_dt match the loader's bare names (kimi-linear precedent)
        # ssm_a is loaded bare (SSM_A_NOSCAN has no suffix), ssm_dt is loaded with ".bias"
        mapped = self.map_tensor_name(name)
        if bid is not None and mapped == f"blk.{bid}.ssm_a":
            return mapped
        if not mapped.endswith((".weight", ".bias")):
            # exp_probs_b is the router's bias-correction vector even though the HF
            # name ends in "_bias", not ".bias"; the loader expects ".bias".
            bias = name.endswith(".bias") or mapped.endswith(".ssm_dt") or mapped.endswith(".exp_probs_b")
            mapped += ".bias" if bias else ".weight"
        return mapped
