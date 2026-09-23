# RDNA4 / ROCm research path

This branch is the consolidated research fork of llama.cpp used by the REMORA
Lab local-inference work. It exists so that a visitor can build and run the
same RDNA4/RDNA4-class path the research used, instead of reconstructing it
from a dozen divergent branches.

It is a **research branch**, not a proposed upstream change. Expect
experimental flags, campaign instrumentation, and knobs that exist to make a
measurement reproducible rather than to be a good default.

## What is in this branch

| Area | What it adds |
| --- | --- |
| Backend policy | ROCm/HIP as the production/performance backend with Vulkan kept as the parity oracle, debug path, and mechanism donor. Both backends build from this tree. |
| `alice_ai` architecture | A custom hybrid KDA linear-attention + MoE model family (48 blocks: 36 linear-attention + 12 full-attention, 512 routed experts top-10 plus one shared, `expert_ff` 512, hidden 2048, 262144 context, 1 declared MTP layer). |
| Recurrent-state correctness | The Alice-local recurrent conv snapshot plane fix: the snapshot offset must be `n_seq_tokens - slot`, not `min(slot, n_seq_tokens)`. Without it, `K > 1` speculative rollback restores conv state and S-state from different steps. A model-free known-answer test covers the plane-equals-rollback-depth contract. |
| Host expert tier and arena | An explicit host expert arena over a packed store, plus the host-buffer plumbing that lets device-resident execution read host-tier weights. The arena is what moves decode from the cold-mmap class into the double-digit tokens/s class on the reference machine. |
| Readback/submission batching | Vulkan→host split-input readbacks batched into a single synchronization boundary (the change behind the historical 18.430 tokens/s decode record), plus per-copy submit/wait counters so the scheduler's copy behaviour can be attributed. |
| MIX34 (type 44) | A 640-value block layout (20 × 32-value subblocks: 12 IQ4_NL + 8 D32A3 plus a selector word, 4.15 bpw) with CPU dot/type-trait support and a Vulkan dequant/mat-vec path. |
| Expert residency and feed | Freeze-first expert residency controls, owned-queue fills with readahead on registered mappings, and the routed-expert slab prefetch engine measured by the Flash-Next campaign. |
| Routing instrumentation | A route-trace probe and an eval-callback tensor dump for capturing real routing decisions. |

## What is deliberately not in this branch

- No model weights, tokenizer payloads, or datasets.
- No campaign receipts, logs, or local machine identifiers.
- No harness scripts from the agent side of the research.

## Build

Both backends in one tree:

```sh
cmake -B build -S . \
  -DGGML_HIP=ON -DAMDGPU_TARGETS=gfx1200 \
  -DGGML_VULKAN=ON \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLAMA_CURL=OFF
cmake --build build -j
```

ROCm/HIP-only or Vulkan-only builds work the same way by dropping the other
flag. `gfx1200` is the reference target; substitute your own GPU target, and
expect measured behaviour to differ — every number in the REMORA Lab research
records is a single-machine phenotype, not a portability claim.

## Using it

The branch is the runtime side of the research. The measured results, the
negative results, and the exact configurations live in the REMORA Lab
repository, not here:

- `research/alice/README.md` — the Alice campaign: architecture, artifact,
  arena, MTP correctness, backend records, prefill.
- `research/host-kv/README.md` — host-KV block reuse and huge-context behaviour.
- `research/qwen27b/README.md` — the Qwen3.8-27B ROCm side campaign.
- `research/falsified/ALICE_CAMPAIGN_NEGATIVES.md` — what was measured and
  rejected, so it is not re-derived.

Read those before quoting a number from a run of this branch: several
directions in this tree are instrumented but **retired**, and the research
records say which.

## Provenance and licence

This tree is a fork of llama.cpp and inherits its MIT licence and its authors.
The research additions are published under the same terms. See `LICENSE` and
`AUTHORS` in this repository.
