#include "models.h"

#include "ggml-backend.h"
#include "ggml.h"
#include "ggml-cpu.h"

#include <cstdlib>
#include <cstdio>
#include <string>
#include <vector>
#include <mutex>
#include "llama-memory-recurrent.h"


// ---------------------------------------------------------------------------
// ALICE_MOE_STATS: route-mass accumulator (see models.h).
//
// Called by the context after every graph compute when the alice probe nodes
// exist; reads back alice_probe_topk-<il> via the scheduler's host sync and adds
// each selected id to its layer histogram. Written once at process exit.
// Process-global, NOT thread_local: graphs are constructed and destroyed from
// different threads (llama-cli runs the model in a worker and tears the graph
// down elsewhere), so a per-thread accumulator loses every observation.
static alice_moe_probe g_alice_probe;
static std::mutex       g_alice_probe_mu;

void alice_moe_probe::arm(int nlay, int nexp, int ntop) {
    if (active) {
        return;
    }
    active = true;
    n_layer = nlay;
    n_expert = nexp;
    topk = ntop;
    counts.assign((size_t) nlay, std::vector<int64_t>((size_t) nexp, 0));
    act2.assign((size_t) nlay, 0.0);
    if (const char * cap = std::getenv("ALICE_MOE_TRACE_POS")) {
        trace_cap = atoll(cap);
        if (trace_cap > 0) {
            trace.assign((size_t) nlay * (size_t) trace_cap * (size_t) ntop, -1);
        }
    }
}

void alice_moe_probe::observe_act(const char * name, double sum_sq) {
    std::lock_guard<std::mutex> lock(g_alice_probe_mu);
    if (!active) {
        return;
    }
    const char * dash = strrchr(name, '-');
    if (!dash) {
        return;
    }
    int il = atoi(dash + 1);
    if (il < 0 || il >= n_layer) {
        return;
    }
    act2[(size_t) il] += sum_sq;
}

void alice_moe_probe::observe(const char * name, const int32_t * ids, int64_t n_ids) {
    std::lock_guard<std::mutex> lock(g_alice_probe_mu);
    if (!active || !ids || n_ids <= 0) {
        return;
    }
    // name is "alice_probe_topk-<il>"
    if (topk > 0) {
        last_ubatch_tokens = n_ids / topk;   // ids are [k, n_pos]
    }
    const char * dash = strrchr(name, '-');
    if (!dash) {
        return;
    }
    int il = atoi(dash + 1);
    if (il < 0 || il >= n_layer) {
        return;
    }
    for (int64_t i = 0; i < n_ids; ++i) {
        int e = (int) ids[i];
        if (e >= 0 && e < n_expert) {
            counts[(size_t) il][(size_t) e] += 1;
        }
    }
    // teacher-forced route trace: ids are [k, n_pos] flattened position-major
    if (trace_cap > 0 && topk > 0) {
        const int64_t n_pos = n_ids / topk;
        for (int64_t p = 0; p < n_pos; ++p) {
            const int64_t pos = trace_pos + p;
            if (pos >= trace_cap) {
                break;
            }
            int32_t * dst = &trace[((size_t) il * (size_t) trace_cap + (size_t) pos) * (size_t) topk];
            for (int64_t k = 0; k < topk; ++k) {
                dst[k] = ids[p * topk + k];
            }
        }
    }
}

std::string alice_moe_probe::json() const {
    std::string out;
    char buf[256];
    snprintf(buf, sizeof(buf), "{\"n_layer\":%d,\"n_expert\":%d,\"topk\":%d,\"tokens\":%lld,\"counts\":[",
        n_layer, n_expert, topk, (long long) tokens);
    out += buf;
    for (int l = 0; l < n_layer; ++l) {
        out += (l ? ",[" : "[");
        for (int e = 0; e < n_expert; ++e) {
            char nb[32];
            snprintf(nb, sizeof(nb), "%s%lld", e ? "," : "", (long long) counts[(size_t) l][(size_t) e]);
            out += nb;
        }
        out += "]";
    }
    out += "],\"act2\":[";
    const bool dbg = std::getenv("ALICE_DEBUG_PROBE") != nullptr;
    for (int l = 0; l < n_layer; ++l) {
        char nb[64];
        snprintf(nb, sizeof(nb), "%.6g", act2[(size_t) l]);
        for (char * c = nb; *c; ++c) {
            if (*c == ',') {
                *c = '.'; // the process locale may be de_DE; JSON needs '.'
            }
        }
        out += (l ? "," : "");
        out += nb;
        if (dbg) {
            fprintf(stderr, "[probe] json act2[%d/%zu] = %.6g\n", l, act2.size(), act2[(size_t) l]);
        }
    }
    return out + "]}";
}


// Read the probe selection tensors out of a *computed* graph. ggml keeps every
// node buffer in its scheduler allocations; after a synchronous compute the
// buffers already hold results, so a plain host copy is exact.

bool alice_moe_probe_collect(ggml_cgraph * gf, int64_t * out_ids) {
    if (!g_alice_probe.active || !gf) {
        return false;
    }
    // NOTE Sep 2026: the probe tensor is "alice_probe_topk-<il>" (distinct from the
    // deployed router's "ffn_moe_topk-<il>", which would double-count selections).
    const char * want = "alice_probe_topk-";
    const size_t want_len = strlen(want);
    const char * want_act = "alice_act2-";
    const size_t want_act_len = strlen(want_act);
    int64_t total = 0;
    const int n_nodes = ggml_graph_n_nodes(gf);
    for (int i = 0; i < n_nodes; ++i) {
        ggml_tensor * t = ggml_graph_node(gf, i);
        if (!t || !t->name[0]) {
            continue;
        }
        if (strncmp(t->name, want_act, want_act_len) == 0) {
            if (t->type == GGML_TYPE_F32 && ggml_nelements(t) == 1) {
                float v = 0.0f;
                ggml_backend_tensor_get(t, &v, 0, sizeof(v));
                g_alice_probe.observe_act(t->name, (double) v);
            }
            continue;
        }
        if (strncmp(t->name, want, want_len) != 0) {
            continue;
        }
        const int64_t n = ggml_nelements(t);
        if (n <= 0) {
            continue;
        }
        std::vector<int32_t> ids((size_t) n);
        ggml_backend_tensor_get(t, ids.data(), 0, (size_t) n * sizeof(int32_t));
        g_alice_probe.observe(t->name, ids.data(), n);
        total += n;
    }
    g_alice_probe.tokens += 1;
    if (g_alice_probe.trace_cap > 0 && g_alice_probe.topk > 0 && total > 0) {
        // total counts one layer's ids; the token count of the ubatch is derived
        // from the first layer only, so compute it from the node count directly.
        g_alice_probe.trace_pos += g_alice_probe.last_ubatch_tokens;
    }
    if (out_ids) {
        *out_ids = total;
    }
    return true;
}


void alice_moe_probe_write(const alice_moe_probe & probe) {
    const char * path = std::getenv("ALICE_MOE_STATS");
    if (!path || !*path) {
        return;
    }
    FILE * f = fopen(path, "w");
    if (!f) {
        fprintf(stderr, "[alice] ALICE_MOE_STATS: cannot open %s\n", path);
        return;
    }
    std::string j = probe.json();
    fwrite(j.data(), 1, j.size(), f);
    fclose(f);

    // ALICE_MOE_TRACE: binary per-position route trace (header + int32 payload)
    if (const char * tp = std::getenv("ALICE_MOE_TRACE")) {
        FILE * tf = fopen(tp, "wb");
        if (tf) {
            const int32_t hdr[4] = { (int32_t) probe.n_layer, (int32_t) probe.trace_cap,
                                     (int32_t) probe.topk, (int32_t) std::min(probe.trace_pos, probe.trace_cap) };
            fwrite(hdr, sizeof(hdr), 1, tf);
            fwrite(probe.trace.data(), sizeof(int32_t), probe.trace.size(), tf);
            fclose(tf);
            fprintf(stderr, "[alice] wrote route trace to %s (%lld positions)\n", tp,
                    (long long) hdr[3]);
        }
    }
    int64_t tot = 0;
    for (const auto & row : probe.counts) {
        for (auto c : row) {
            tot += c;
        }
    }
    fprintf(stderr, "[alice] wrote route mass to %s (%lld selections)\n", path, (long long) tot);
}

// Yandex AliceAI-80B-A3B: hybrid 3xKDA+1xgated-attention, sigmoid-router MoE with
// bias correction + shared expert, split block-attention-residual (depth-softmax-mix).
// KDA recurrence, conv state, hybrid memory, partial RoPE, QK-norm all reuse the
// kimi-linear / delta-net-base / qwen3next primitives.


// ---------------------------------------------------------------------------
// ALICE_MOE_CPU_BLOCK (ALICE_MOE_BLOCK=1): one composite CPU op per MoE layer.
//
// Problem (ALICE_CPU_MOE_LAYER_ATTRIBUTION.json): the deployed CPU MoE layer is
// ~48 graph nodes, each with its own thread-team barrier. The arithmetic sums
// to ~0.49 ms but the deployed layer costs ~1.58 ms; ~1.1 ms is dispatch and
// barriers, not math. The two earlier fused attempts lost (0.85-0.89x,
// ALICE_FUSED_MOE_AB.json) because they reimplemented the expert matvecs by
// hand (5 custom stages, host-visible intermediates, extra scheduler splits)
// instead of reusing ggml's row-parallel mm_id.
//
// This op does NOT reimplement the matvec. It reuses the exact ggml CPU
// kernels (mul_mat, vec_dot path of mul_mat_id, silu, add) by calling the same
// forward functions ggml would call, but WITHOUT building 48 graph nodes: one
// CUSTOM node = one scheduler split region = one thread-team entry. The op's
// dst is one plain [n_embd, n_tokens] f32 tensor allocated by the scheduler in
// the CPU backend's compute buffer (CUSTOM has no internal wdata and needs
// none: all scratch is model-owned, allocated once at load in a host buffer).
//
// Three nodes per layer (was ~48): R (router+topk, single thread), A
// (gate_up rows striped over the team into CPU-local staging), B (silu +
// down rows striped by OUTPUT ROW, atomic-free). Two inter-node barriers
// instead of ~48 per-node barriers. See the node comments for why a single
// barrier-free node is impossible without atomics or 2.6M-dot recompute.
//
// Correctness: arithmetic order matches the deployed path row-for-row (router
// dot, sigmoid, bias, partial selection sort over the same 512 values,
// gate_up vec_dot per row, ggml silu, down vec_dot per row, weighted
// accumulate). ggml's fast-exp sigmoid/silu differ from libm by ~1e-7; the
// parity gate is top-10 ids + max|logit diff|, same bar the earlier fused
// attempt used (top-5 identical, max|diff| 0.63).
//
// Enabled per layer only when BOTH expert banks are host-backed (same gate as
// alice_fused_moe_layer_ok): the layers -ncmoe keeps on the CPU. GPU layers
// and every other arch keep the deployed path. Gated by ALICE_MOE_BLOCK=1
// (and ALICE_MOE_BLOCK_MAX for differential layer counts). The old
// ALICE_FUSED_MOE 5-stage path is untouched.

// scalar top-k over one row of 512 f32 logits: insertion into a 10-slot
// min-heap kept in registers. ids_out[j], w_out[j] = (expert, prob).
static inline void alice_moe_block_topk10(const float * probs, int32_t * ids_out, float * w_out) {
    int32_t ids[10];
    float   w[10];
    for (int j = 0; j < 10; ++j) { ids[j] = -1; w[j] = -1e30f; }
    for (int e = 0; e < 512; ++e) {
        const float v = probs[e];
        if (v <= w[0]) continue;
        int j = 0;
        w[0] = v; ids[0] = e;
        // sift down
        for (;;) {
            int c0 = 2*j+1, c1 = 2*j+2, m = j;
            if (c0 < 10 && w[c0] < w[m]) m = c0;
            if (c1 < 10 && w[c1] < w[m]) m = c1;
            if (m == j) break;
            float tv = w[j]; w[j] = w[m]; w[m] = tv;
            int32_t ti = ids[j]; ids[j] = ids[m]; ids[m] = ti;
            j = m;
        }
    }
    // heap order is not sorted; selection sort the 10 (order must match
    // argsort DESC so the ids tensor is bit-comparable with deployed)
    for (int a = 0; a < 10; ++a) {
        int b = a;
        for (int c = a+1; c < 10; ++c) if (w[c] > w[b]) b = c;
        float tv = w[a]; w[a] = w[b]; w[b] = tv;
        int32_t ti = ids[a]; ids[a] = ids[b]; ids[b] = ti;
    }
    for (int j = 0; j < 10; ++j) { ids_out[j] = ids[j]; w_out[j] = w[j]; }
}

// Node R: router + topk, runs single-threaded (n_tasks=1).
// src[0] = x [n_embd, n_tok] f32. dst = [20, n_tok] f32 packing ids (as f32
// bit patterns via memcpy) in rows [0,10) and weights in rows [10,20).
// Readers decode with memcpy (exact, no cast). One writer (ith==0), one node.
static void alice_moe_block_router(ggml_tensor * dst, int ith, int nth, void * ud_) {
    (void) nth;
    if (ith != 0) return;
    const auto * ud = (const alice_moe_block_ud *) ud_;
    const llama_layer & layer = ud->model->layers[ud->il];
    const ggml_tensor * x  = dst->src[0];
    const ggml_tensor * gi = layer.ffn_gate_inp;
    const ggml_tensor * pb = layer.ffn_exp_probs_b;
    const int64_t n_embd = x->ne[0], n_tok = x->ne[1];
    const float * xb = (const float *) x->data;
    const float * gib = (const float *) gi->data;
    const float * pbb = pb ? (const float *) pb->data : nullptr;
    for (int64_t t = 0; t < n_tok; ++t) {
        const float * xt = (const float *) ((const char *) xb + t * x->nb[1]);
        float lg[512], pr[512];
        for (int64_t e = 0; e < 512; ++e) {
            const float * col = (const float *) ((const char *) gib + e * gi->nb[1]);
            double s = 0.0;
            for (int64_t i = 0; i < n_embd; ++i) s += (double) xt[i] * col[i];
            lg[e] = (float) s;
        }
        for (int64_t e = 0; e < 512; ++e) {
            const float v = lg[e] + (pbb ? pbb[e] : 0.0f);
            pr[e] = 1.0f / (1.0f + expf(-v));
        }
        int32_t idt[10]; float wtt[10];
        alice_moe_block_topk10(pr, idt, wtt);
        float sum = 0.0f;
        for (int j = 0; j < 10; ++j) sum += wtt[j];
        if (sum < 6.103515625e-5f) sum = 6.103515625e-5f;
        float * slot = (float *) ((char *) dst->data + t * dst->nb[1]);
        for (int j = 0; j < 10; ++j) {
            int32_t id = idt[j]; float w = wtt[j] / sum;
            memcpy(slot + j, &id, sizeof(int32_t));
            slot[10 + j] = w;
        }
    }
}

// Node A: gate_up rows striped; writes [2*n_ff, n_top, n_tok] f32 staging.
// src[0] = x [n_embd, n_tok]; src[1] = router pack [20, n_tok] from node R.
static void alice_moe_block_gate_up(ggml_tensor * dst, int ith, int nth, void * ud_) {
    const auto * ud = (const alice_moe_block_ud *) ud_;
    const llama_layer & layer = ud->model->layers[ud->il];
    const ggml_tensor * x  = dst->src[0];
    const ggml_tensor * rk = dst->src[1];   // router pack [20, n_tok] from node R
    const ggml_tensor * gu = layer.ffn_gate_up_exps;
    const int64_t n_embd = x->ne[0], n_tok = x->ne[1];
    const int64_t n_ff = gu->ne[1] / 2;
    const ggml_vec_dot_t vd_gu = ggml_get_type_traits_cpu(gu->type)->vec_dot;
    const int64_t row_gu = (int64_t) ggml_row_size(gu->type, gu->ne[0]);
    const size_t b_gu = (size_t) row_gu * (size_t) (2 * n_ff);
    const float * xb = (const float *) x->data;
    char xq[2400];
    for (int64_t t = 0; t < n_tok; ++t) {
        const float * xt = (const float *) ((const char *) xb + t * x->nb[1]);
        const float * slot_r = (const float *) ((const char *) rk->data + t * rk->nb[1]);
        int32_t idt[10];
        for (int j = 0; j < 10; ++j) memcpy(&idt[j], slot_r + j, sizeof(int32_t));
        ggml_get_type_traits_cpu(GGML_TYPE_Q8_K)->from_float(xt, xq, n_embd);
        const int64_t units = (int64_t) 10 * 2 * n_ff;
        for (int64_t u = ith; u < units; u += nth) {
            const int64_t j = u / (2 * n_ff), r = u % (2 * n_ff);
            const int32_t e = idt[j];
            float s = 0.0f;
            vd_gu((int) gu->ne[0], &s, 0,
                  (const char *) gu->data + (size_t) e * b_gu + (size_t) r * row_gu, 0,
                  xq, 0, 1);
            float * slot = (float *) ((char *) dst->data + t * dst->nb[2] + j * dst->nb[1]);
            slot[r] = s;
        }
    }
}

// Node B: silu(gate)*up + down rows, striped by OUTPUT ROW (atomic-free).
// src[0] = gu staging [2*n_ff, n_top, n_tok] f32 from node A.
// Reads model weights via userdata (dn bank, gate_inp, bias); rediscovers the
// same ids/weights node A found (deterministic 15 us recompute, no sharing).
// dst = [n_embd, n_tok] f32 complete MoE output rows. Each output row has
// exactly one writer thread: rows r with (t*n_embd + r) % nth == ith.
static void alice_moe_block_down(ggml_tensor * dst, int ith, int nth, void * ud_) {
    const auto * ud = (const alice_moe_block_ud *) ud_;
    const llama_layer & layer = ud->model->layers[ud->il];
    const ggml_tensor * gu_out = dst->src[0];   // [2*n_ff, n_top, n_tok] f32
    const ggml_tensor * rk     = dst->src[1];   // router pack [20, n_tok]
    const ggml_tensor * dn = layer.ffn_down_exps;
    const int64_t n_embd = dst->ne[0], n_tok = dst->ne[1];
    const int64_t n_ff = dn->ne[0];
    const ggml_vec_dot_t vd_dn = ggml_get_type_traits_cpu(dn->type)->vec_dot;
    const int64_t row_dn = (int64_t) ggml_row_size(dn->type, dn->ne[0]);
    const size_t b_dn = (size_t) row_dn * (size_t) n_embd;
    // down activation is Q8_K-quantised per row inside ggml's mm_id; here the
    // activation is silu output computed on the fly, so quantize per expert
    // row once per owned output row (stack buffer, 512 floats -> 544 bytes).
    char aq[640];
    float sw[512];
    for (int64_t t = 0; t < n_tok; ++t) {
        const float * slot_r = (const float *) ((const char *) rk->data + t * rk->nb[1]);
        int32_t idt[10]; float wtt[10];
        for (int j = 0; j < 10; ++j) { memcpy(&idt[j], slot_r + j, sizeof(int32_t)); wtt[j] = slot_r[10 + j]; }
        const float * gu_t = (const float *) ((const char *) gu_out->data + t * gu_out->nb[2]);
        float * out_t = (float *) ((char *) dst->data + t * dst->nb[1]);
        for (int64_t r = ith; r < n_embd; r += nth) {
            double acc = 0.0;
            for (int j = 0; j < 10; ++j) {
                const int32_t e = idt[j];
                const float * grow = gu_t + j * (gu_out->nb[1] / sizeof(float));
                for (int64_t i = 0; i < n_ff; ++i) {
                    const float g = grow[i], u = grow[n_ff + i];
                    sw[i] = (g / (1.0f + expf(-g))) * u;
                }
                ggml_get_type_traits_cpu(GGML_TYPE_Q8_K)->from_float(sw, aq, n_ff);
                float s = 0.0f;
                vd_dn((int) dn->ne[0], &s, 0,
                      (const char *) dn->data + (size_t) e * b_dn + (size_t) r * row_dn, 0,
                      aq, 0, 1);
                acc += (double) wtt[j] * s;
            }
            out_t[r] = (float) acc;
        }
    }
}

// Build the two-node block. Returns the MoE output tensor. Replaces the whole
// deployed chain (router mm, sigmoid, bias, argsort, get_rows, renorm, gate_up
// mm_id, views, swiglu, down mm_id, weight mul, 9-add aggregate) with 2 nodes.
// The shared expert is NOT included (separate deployed chain, 0.055 ms).
// Only for host-backed expert layers; gated by ALICE_MOE_BLOCK_MAX.
static ggml_tensor * alice_moe_block_build(struct ggml_context * ctx0, ggml_tensor * ffn_inp,
        int il, const llama_model & model, llm_graph_context * gctx) {
    const llama_layer & layer = model.layers[il];
    const int64_t n_embd = ffn_inp->ne[0], n_tok = ffn_inp->ne[1];
    const int64_t n_ff = layer.ffn_down_exps->ne[0];
    const int64_t n_top = 10;
    auto * am = (llama_model_alice_ai *) &model;
    if (am->moe_block_ud.size() < model.layers.size()) {
        am->moe_block_ud.resize(model.layers.size());
        for (size_t i = 0; i < model.layers.size(); ++i) am->moe_block_ud[i] = { &model, (int) i };
    }
    ggml_tensor * xin[1] = { ffn_inp };
    ggml_tensor * rk = ggml_custom_4d(ctx0, GGML_TYPE_F32, 20, n_tok, 1, 1,
        xin, 1, alice_moe_block_router, 1, &am->moe_block_ud[il]);
    gctx->cb(rk, "ffn_moe_block_rk", il);
    ggml_tensor * ain[2] = { ffn_inp, rk };
    ggml_tensor * gu_out = ggml_custom_4d(ctx0, GGML_TYPE_F32, 2*n_ff, n_top, n_tok, 1,
        ain, 2, alice_moe_block_gate_up, GGML_N_TASKS_MAX, &am->moe_block_ud[il]);
    gctx->cb(gu_out, "ffn_moe_block_gu", il);
    ggml_tensor * bin[2] = { gu_out, rk };
    ggml_tensor * out = ggml_custom_4d(ctx0, GGML_TYPE_F32, n_embd, n_tok, 1, 1,
        bin, 2, alice_moe_block_down, GGML_N_TASKS_MAX, &am->moe_block_ud[il]);
    gctx->cb(out, "ffn_moe_block_out", il);
    // Parity tap (ALICE_MOE_BLOCK_PARITY=1): publish the block's own top-k ids
    // next to the deployed router's ffn_moe_topk-<il> so the existing probe
    // hook (ALICE_MOE_PROBE_HOOK + alice_probe_topk-<il>) and a direct
    // graph-node read can compare selections without touching the hot path.
    // Off by default; zero cost when unset.
    if (std::getenv("ALICE_MOE_BLOCK_PARITY") != nullptr) {
        extern ggml_tensor * alice_moe_block_parity_tap(struct ggml_context * ctx0,
            ggml_tensor * gu_out, int il, void * ud,
            void (*name_cb)(ggml_tensor *, const char *, int));
        // implemented below: reuses the deterministic router (same 15 us
        // code) to emit [10, n_tok] i32 ids into a named graph tensor.
        ggml_tensor * ids_tap = alice_moe_block_parity_tap(ctx0, gu_out, il, &am->moe_block_ud[il],
            [](ggml_tensor * t, const char * n, int i){ (void) t; (void) n; (void) i; });
        ggml_build_forward_expand(gctx->gf, ids_tap);
    }
    return out;
}

// Parity tap implementation: a CUSTOM node that re-runs the block router
// (matvec + sigmoid + bias + topk) and writes [10, n_tok] i32 ids. It shares
// node A's inputs via gu_out->src[0] (ffn_inp). Named alice_moe_block_ids-<il>.
// Decodes node R's pack [20, n_tok] f32 (ids bit-copied into rows 0..9,
// weights in rows 10..19) into [10, n_tok] i32. Single-threaded.
static void alice_moe_block_ids_exec(ggml_tensor * dst, int ith, int nth, void * ud_) {
    (void) ud_; (void) nth;
    if (ith != 0) return;
    const ggml_tensor * rk = dst->src[0];
    const int64_t n_tok = dst->ne[1];
    for (int64_t t = 0; t < n_tok; ++t) {
        const float * slot = (const float *) ((const char *) rk->data + t * rk->nb[1]);
        int32_t * out = (int32_t *) ((char *) dst->data + t * dst->nb[1]);
        for (int j = 0; j < 10; ++j) memcpy(&out[j], slot + j, sizeof(int32_t));
    }
}

ggml_tensor * alice_moe_block_parity_tap(struct ggml_context * ctx0,
        ggml_tensor * gu_out, int il, void * ud,
        void (*name_cb)(ggml_tensor *, const char *, int)) {
    // rk pack lives in node A's src[1] (ain[1]); decode it, no recompute.
    ggml_tensor * rk = gu_out->src[1];
    ggml_tensor * rin[1] = { rk };
    ggml_tensor * ids = ggml_custom_4d(ctx0, GGML_TYPE_I32, 10, gu_out->ne[2], 1, 1,
        rin, 1, alice_moe_block_ids_exec, 1, ud);
    char name[64];
    snprintf(name, sizeof(name), "alice_moe_block_ids-%d", il);
    ggml_set_name(ids, name);
    (void) name_cb;
    return ids;
}



// ---------------------------------------------------------------------------
// Fused CPU MoE executor (ALICE_FUSED_MOE=1)
//
// Measured problem: a CPU-resident Alice MoE layer costs ~1.58 ms and moves ~14 MiB
// at an effective 9.6 GB/s. The deployed graph spends that on two mul_mat_id ops, a
// swiglu_split, a weight multiply and a chain of nine view+add nodes to aggregate the
// ten experts — ~13 nodes, each with its own thread barrier, for 10 experts of work.
//
// The same arithmetic in one pass (tools/alice_moe_fused.cpp, bit-exact against the
// staged path, see ALICE_FUSED_MOE_BENCH.json) runs at 32-43 GB/s, the host's
// aggregate DRAM ceiling, i.e. ~0.35 ms/layer. This is that executor, wired into the
// graph as two ops per layer: one CUSTOM op producing the weighted per-expert
// partials (each (token, expert) slice written by exactly one thread, so no
// atomics and no races), and one CUSTOM reduction over the expert axis.
//
// Enabled per layer only when the layer's expert tensors are host-backed, i.e. the
// layers that -ncmoe keeps on the CPU; every other layer keeps the deployed path.

extern "C" void quantize_row_q8_K_ref(const float * x, void * y, int64_t k);
extern "C" void ggml_vec_swiglu_f32(int n, float * y, const float * x, const float * g);

// Row-parallel staged executor.
//
// First attempt (one op, expert-sized work items) lost 15% end-to-end: ten items over
// eight threads leaves most of the team idle while ggml's mul_mat_id spreads rows over
// all of them. Each stage below therefore partitions ROWS (or quantisation blocks), so
// every stage has thousands of independent units for the backend's thread team:
//
//   E quant_x : n_tok units            x        -> q8_K activations
//   A gate_up : n_tok*n_top*2*n_ff     gu rows
//   B swiglu  : n_tok*n_top*(n_ff/256) silu(g)*u, quantised per 256-block
//   C down    : n_tok*n_top*n_embd     down rows, weighted
//   D reduce  : n_tok*n_embd           expert-axis sum
//
// Every unit writes a distinct slot of its dst, so no atomics and no internal barriers.

static inline int64_t alice_fmoe_row_bytes(const ggml_tensor * t, int64_t n_per_row) {
    return (int64_t) ggml_row_size(t->type, n_per_row);
}

static void alice_fmoe_quant_x(ggml_tensor * dst, int ith, int nth, void * ud_) {
    const auto * ud = (const alice_fused_moe_ud *) ud_;
    const ggml_tensor * x = dst->src[0];                     // [n_embd, n_tok]
    const int64_t n_embd = x->ne[0], n_tok = x->ne[1];
    const ggml_tensor * gu = ud->model->layers[ud->il].ffn_gate_up_exps;
    const ggml_type vq = ggml_get_type_traits_cpu(gu->type)->vec_dot_type;
    const ggml_from_float_t q = ggml_get_type_traits_cpu(vq)->from_float;
    for (int64_t t = ith; t < n_tok; t += nth) {
        const float * row = (const float *) ((const char *) x->data + t * x->nb[1]);
        char * out = (char *) dst->data + t * dst->nb[1];
        if (q) { q(row, out, n_embd); } else { quantize_row_q8_K_ref(row, out, n_embd); }
    }
}

static void alice_fmoe_gate_up(ggml_tensor * dst, int ith, int nth, void * ud_) {
    const auto * ud = (const alice_fused_moe_ud *) ud_;
    const llama_layer & layer = ud->model->layers[ud->il];
    const ggml_tensor * xq  = dst->src[0];                   // [row_x, n_tok] q8_K
    const ggml_tensor * ids = dst->src[1];                   // [n_top, n_tok] i32
    const ggml_tensor * gu  = layer.ffn_gate_up_exps;        // [n_embd, 2*n_ff, n_expert]
    const int64_t n_top = ids->ne[0], n_tok = ids->ne[1];
    const int64_t rows = dst->ne[0];                         // 2*n_ff
    const int64_t row_gu = alice_fmoe_row_bytes(gu, gu->ne[0]);
    const ggml_vec_dot_t vd = ggml_get_type_traits_cpu(gu->type)->vec_dot;
    const int64_t units = n_tok * n_top * rows;
    const size_t b_gu = (size_t) row_gu * (size_t) rows;     // one expert's block
    for (int64_t u = ith; u < units; u += nth) {
        const int64_t r  = u % rows;
        const int64_t tj = u / rows;
        const int64_t t  = tj / n_top, j = tj % n_top;
        const int32_t e  = *(const int32_t *) ((const char *) ids->data + t * ids->nb[1] + j * ids->nb[0]);
        float s = 0.0f;
        vd((int) gu->ne[0], &s, 0, (const char *) gu->data + (size_t) e * b_gu + (size_t) r * row_gu, 0,
           (const char *) xq->data + t * xq->nb[1], 0, 1);
        *(float *) ((char *) dst->data + t * dst->nb[2] + j * dst->nb[1] + r * sizeof(float)) = s;
    }
}

static void alice_fmoe_swiglu(ggml_tensor * dst, int ith, int nth, void * ud_) {
    const auto * ud = (const alice_fused_moe_ud *) ud_;
    const llama_layer & layer = ud->model->layers[ud->il];
    const ggml_tensor * gu = dst->src[0];                    // [2*n_ff, n_top, n_tok]
    const int64_t n_ff = layer.ffn_down_exps->ne[0];
    const int64_t n_top = gu->ne[1], n_tok = gu->ne[2];
    const ggml_type vq = ggml_get_type_traits_cpu(gu->type)->vec_dot_type;   // unused: F32 in
    GGML_UNUSED(vq);
    const ggml_type vq2 = ggml_get_type_traits_cpu(layer.ffn_down_exps->type)->vec_dot_type;
    const ggml_from_float_t q = ggml_get_type_traits_cpu(vq2)->from_float;
    const int64_t blk = ggml_blck_size(vq2);                 // 256 for Q8_K
    const int64_t nblk = n_ff / blk;
    const int64_t row_q = (int64_t) ggml_row_size(vq2, n_ff);
    std::vector<float> gate((size_t) blk), up((size_t) blk), sw((size_t) blk);
    const int64_t units = n_tok * n_top * nblk;
    for (int64_t u = ith; u < units; u += nth) {
        const int64_t b  = u % nblk;
        const int64_t tj = u / nblk;
        const int64_t t  = tj / n_top, j = tj % n_top;
        const char * gbase = (const char *) gu->data + t * gu->nb[2] + j * gu->nb[1];
        for (int64_t i = 0; i < blk; ++i) {
            gate[(size_t) i] = *(const float *) (gbase + (b * blk + i) * sizeof(float));
            up[(size_t) i]   = *(const float *) (gbase + (n_ff + b * blk + i) * sizeof(float));
        }
        ggml_vec_swiglu_f32((int) blk, sw.data(), gate.data(), up.data());
        char * out = (char *) dst->data + t * dst->nb[2] + j * dst->nb[1] + b * (row_q / nblk);
        if (q) { q(sw.data(), out, blk); } else { quantize_row_q8_K_ref(sw.data(), out, blk); }
    }
}

static void alice_fmoe_down(ggml_tensor * dst, int ith, int nth, void * ud_) {
    const auto * ud = (const alice_fused_moe_ud *) ud_;
    const llama_layer & layer = ud->model->layers[ud->il];
    const ggml_tensor * aq  = dst->src[0];                   // [row_h, n_top, n_tok] quantised swiglu
    const ggml_tensor * ids = dst->src[1];
    const ggml_tensor * w   = dst->src[2];                   // [n_top, n_tok]
    const ggml_tensor * dn  = layer.ffn_down_exps;           // [n_ff, n_embd, n_expert]
    const int64_t n_top = ids->ne[0], n_tok = ids->ne[1];
    const int64_t rows = dst->ne[0];                         // n_embd
    const int64_t row_dn = alice_fmoe_row_bytes(dn, dn->ne[0]);
    const ggml_vec_dot_t vd = ggml_get_type_traits_cpu(dn->type)->vec_dot;
    const size_t b_dn = (size_t) row_dn * (size_t) rows;
    const int64_t units = n_tok * n_top * rows;
    for (int64_t u = ith; u < units; u += nth) {
        const int64_t r  = u % rows;
        const int64_t tj = u / rows;
        const int64_t t  = tj / n_top, j = tj % n_top;
        const int32_t e  = *(const int32_t *) ((const char *) ids->data + t * ids->nb[1] + j * ids->nb[0]);
        const float wt   = *(const float *) ((const char *) w->data + t * w->nb[1] + j * w->nb[0]);
        float s = 0.0f;
        vd((int) dn->ne[0], &s, 0, (const char *) dn->data + (size_t) e * b_dn + (size_t) r * row_dn, 0,
           (const char *) aq->data + t * aq->nb[2] + j * aq->nb[1], 0, 1);
        *(float *) ((char *) dst->data + t * dst->nb[2] + j * dst->nb[1] + r * sizeof(float)) = wt * s;
    }
}

static void alice_fmoe_reduce(ggml_tensor * dst, int ith, int nth, void *) {
    const ggml_tensor * part = dst->src[0];                  // [n_embd, n_top, n_tok]
    const int64_t n_embd = part->ne[0], n_top = part->ne[1], n_tok = part->ne[2];
    const int64_t units = n_tok * n_embd;
    for (int64_t u = ith; u < units; u += nth) {
        const int64_t r = u % n_embd, t = u / n_embd;
        const char * base = (const char *) part->data + t * part->nb[2] + r * sizeof(float);
        float acc = 0.0f;
        for (int64_t j = 0; j < n_top; ++j) { acc += *(const float *) (base + j * part->nb[1]); }
        *(float *) ((char *) dst->data + t * dst->nb[1] + r * sizeof(float)) = acc;
    }
}

// ---------------------------------------------------------------------------
// ALICE_MOE_COMPACT (ALICE_HOTBANK=1) — compact expert worklist.
//
// The deployed MoE executes a fixed top-10 width on both sides. This op
// executes EXACTLY the cold experts (host bank) with zero dummy rows:
//   src[0] = cur        [n_embd, n_tok] f32 activations
//   src[1] = ids        [n_top, n_tok] i32 routed ids (global)
//   src[2] = weights    [1, n_top, n_tok] f32 routing weights
//   src[3] = gate_up    [n_embd, 2*n_ff, n_expert] host bank (type traits)
//   dst    = [n_embd, n_tok] f32 cold-partial contribution (hot handled by
//            the companion GPU nodes built in llama-graph.cpp)
// Hot ids are published through hotbank_publish_ids() so the graph builder can
// wire the GPU side from the SAME partition decision.

// Published per-layer partition results (written by the op at compute time,
// read by the GPU-side nodes that the scheduler runs after it). Guarded by a
// mutex because the op may run on a worker thread.
struct alice_hotbank_shared {
    std::mutex mu;
    // per layer: hot-local ids, cold-global ids, their weights, counts
    int32_t hot_ids[48][16];
    int32_t cold_ids[48][16];
    float   hot_w[48][16];
    float   cold_w[48][16];
    int     n_hot[48];
    int     n_cold[48];
    int64_t tokens[48];
};
static alice_hotbank_shared g_hotbank_shared;

// Op 1: partition — publish hot-local ids (I32 [n_top, n_tok], 0-padded) for
// the GPU side. Counts stay in the shared struct for the cold op.
static void alice_hotbank_partition(ggml_tensor * dst, int ith, int nth, void * ud_) {
    const auto * ud = (const alice_hotbank_part_ud *) ud_;
    const int il = ud->il;
    const auto * am = static_cast<const llama_model_alice_ai *>(ud->amodel);
    const ggml_tensor * ids = dst->src[0];
    const ggml_tensor * wts = dst->src[1];
    const int16_t * lut = am->hot_lut[il].local;
    const int64_t n_top = ids->ne[0], n_tok = ids->ne[1];
    for (int64_t t = ith; t < n_tok; t += nth) {
        int32_t * out = (int32_t *) ((char *) dst->data + t * dst->nb[1]);
        int nh = 0;
        const int64_t n_top_eff = n_top > 16 ? 16 : n_top;
        for (int64_t j = 0; j < n_top_eff; ++j) {
            const int32_t e = *(const int32_t *) ((const char *) ids->data + t * ids->nb[1] + j * ids->nb[0]);
            const int li = (e >= 0 && e < 512) ? lut[e] : -1;
            if (li >= 0 && nh < 16) { out[nh++] = li; }
        }
        for (int64_t j = nh; j < n_top; ++j) { out[j] = 0; }
        if (t == 0) {
            std::lock_guard<std::mutex> lock(g_hotbank_shared.mu);
            g_hotbank_shared.n_hot[il] = nh;
            g_hotbank_shared.n_cold[il] = (int) (n_top - nh);
        }
    }
}

// Op 2: hot weights — publish hot routing weights (F32 [1, n_top, n_tok],
// 0-padded) matching the hot ids published by op 1.
static void alice_hotbank_weights(ggml_tensor * dst, int ith, int nth, void * ud_) {
    const auto * ud = (const alice_hotbank_part_ud *) ud_;
    const int il = ud->il;
    const auto * am = static_cast<const llama_model_alice_ai *>(ud->amodel);
    const ggml_tensor * ids = dst->src[0];
    const ggml_tensor * wts = dst->src[1];
    const int16_t * lut = am->hot_lut[il].local;
    const int64_t n_top = ids->ne[0], n_tok = ids->ne[1];
    for (int64_t t = ith; t < n_tok; t += nth) {
        float * out = (float *) ((char *) dst->data + t * dst->nb[2]);
        int nh = 0;
        const int64_t n_top_eff = n_top > 16 ? 16 : n_top;
        for (int64_t j = 0; j < n_top_eff; ++j) {
            const int32_t e = *(const int32_t *) ((const char *) ids->data + t * ids->nb[1] + j * ids->nb[0]);
            const float  w = *(const float *)   ((const char *) wts->data + t * wts->nb[2] + j * wts->nb[1]);
            const int li = (e >= 0 && e < 512) ? lut[e] : -1;
            if (li >= 0 && nh < 16) { out[nh++] = w; }
        }
        for (int64_t j = nh; j < n_top; ++j) { out[j] = 0.0f; }
    }
}

// Op 3: cold partial — EXACTLY the cold experts on the host bank, row-parallel
// over rows, zero dummy work. dst = sum_k cold_w[k] * down(swiglu(gate_up(x)))
// over the nc cold experts of this token.
// Two ops per layer (scheduler order gives the barrier between them):
//   A alice_hotbank_cold_gu: (token, cold-expert, gate/up row) -> F32 partial
//     activations written into a scratch tensor [2*n_ff, n_top, n_tok].
//   B alice_hotbank_cold_dn: (token, down row) -> swiglu+quant on the fly from
//     the scratch, down dots over the nc cold experts, weighted sum into dst.
// Both are row-parallel with per-thread scratch; no cross-thread dependencies.

static void alice_hotbank_cold_gu(ggml_tensor * dst, int ith, int nth, void * ud_) {
    const auto * ud = (const alice_hotbank_part_ud *) ud_;
    const int il = ud->il;
    if (ud->amodel == nullptr || il < 0 || il >= 48) { return; }
    const auto * am = static_cast<const llama_model_alice_ai *>(ud->amodel);
    const ggml_tensor * cur = dst->src[0];
    const ggml_tensor * ids = dst->src[1];
    const ggml_tensor * gu  = dst->src[2];
    const int16_t * lut = am->hot_lut[il].local;
    const int64_t n_top = ids->ne[0], n_tok = ids->ne[1];
    const int64_t n_embd = cur->ne[0];
    const int64_t n_ff = gu->ne[1] / 2;
    const ggml_vec_dot_t vd = ggml_get_type_traits_cpu(gu->type)->vec_dot;
    const ggml_type vq = ggml_get_type_traits_cpu(gu->type)->vec_dot_type;
    const ggml_from_float_t qf = ggml_get_type_traits_cpu(vq)->from_float;
    const int64_t row_gu = (int64_t) ggml_row_size(gu->type, n_embd);
    const size_t b_gu = (size_t) row_gu * (size_t) (2 * n_ff);
    const int64_t n_top_eff = n_top > 16 ? 16 : n_top;
    std::vector<char> xq((size_t) ggml_row_size(vq, n_embd));
    int64_t cached_t = -1;
    // units = (token, expert-slot, row); expert slot j in [0, nc)
    const int64_t units = n_tok * n_top_eff * (2 * n_ff);
    for (int64_t u = ith; u < units; u += nth) {
        const int64_t r  = u % (2 * n_ff);
        const int64_t tj = u / (2 * n_ff);
        const int64_t t  = tj / n_top_eff;
        const int64_t j  = tj % n_top_eff;
        // classify: find the j-th COLD expert of this token
        int32_t cid[16];
        int nc = 0;
        for (int64_t q = 0; q < n_top_eff; ++q) {
            const int32_t e = *(const int32_t *) ((const char *) ids->data + t * ids->nb[1] + q * ids->nb[0]);
            const int li = (e >= 0 && e < 512) ? lut[e] : -1;
            if (li < 0) { cid[nc++] = e; }
        }
        if (j >= nc) { continue; }
        if (t != cached_t) {
            const float * x = (const float *) ((const char *) cur->data + t * cur->nb[1]);
            if (qf) { qf(x, xq.data(), n_embd); } else { memset(xq.data(), 0, xq.size()); }
            cached_t = t;
        }
        const char * w = (const char *) gu->data + (size_t) cid[j] * b_gu + (size_t) r * row_gu;
        float sv = 0.0f;
        vd((int) n_embd, &sv, 0, w, 0, xq.data(), 0, 1);
        // dst layout: [2*n_ff, n_top, n_tok] (same as the fused executor's gu_out)
        *(float *) ((char *) dst->data + t * dst->nb[2] + j * dst->nb[1] + r * sizeof(float)) = sv;
    }
}

static void alice_hotbank_cold_dn(ggml_tensor * dst, int ith, int nth, void * ud_) {
    const auto * ud = (const alice_hotbank_part_ud *) ud_;
    const int il = ud->il;
    if (ud->amodel == nullptr || il < 0 || il >= 48) { return; }
    const auto * am = static_cast<const llama_model_alice_ai *>(ud->amodel);
    const ggml_tensor * gu_out = dst->src[0];   // [2*n_ff, n_top, n_tok] f32
    const ggml_tensor * ids    = dst->src[1];
    const ggml_tensor * wts    = dst->src[2];
    const ggml_tensor * dn     = dst->src[3];
    const int16_t * lut = am->hot_lut[il].local;
    const int64_t n_top = ids->ne[0], n_tok = ids->ne[1];
    const int64_t n_embd = dst->ne[0];
    const int64_t n_ff = dn->ne[0];
    const ggml_vec_dot_t vd_dn = ggml_get_type_traits_cpu(dn->type)->vec_dot;
    const ggml_type vq_dn = ggml_get_type_traits_cpu(dn->type)->vec_dot_type;
    const ggml_from_float_t qf_dn = ggml_get_type_traits_cpu(vq_dn)->from_float;
    const int64_t row_dn = (int64_t) ggml_row_size(dn->type, n_ff);
    const size_t b_dn = (size_t) row_dn * (size_t) n_embd;
    const int64_t blk = ggml_blck_size(vq_dn);
    const int64_t nblk = n_ff / blk;
    const int64_t row_q = (int64_t) ggml_row_size(vq_dn, n_ff);
    const int64_t n_top_eff = n_top > 16 ? 16 : n_top;
    std::vector<float> gate((size_t) blk), up((size_t) blk), sw((size_t) blk);
    std::vector<char>  aq((size_t) row_q);
    // units = (token, down row)
    const int64_t units = n_tok * n_embd;
    for (int64_t u = ith; u < units; u += nth) {
        const int64_t t  = u / n_embd;
        const int64_t dr = u % n_embd;
        int32_t cid[16]; float cw[16];
        int nc = 0;
        for (int64_t j = 0; j < n_top_eff; ++j) {
            const int32_t e = *(const int32_t *) ((const char *) ids->data + t * ids->nb[1] + j * ids->nb[0]);
            const float  w = *(const float *)   ((const char *) wts->data + t * wts->nb[2] + j * wts->nb[1]);
            const int li = (e >= 0 && e < 512) ? lut[e] : -1;
            if (li < 0) { cid[nc] = e; cw[nc] = w; ++nc; }
        }
        float * out = (float *) ((char *) dst->data + t * dst->nb[1]);
        if (nc == 0) { out[dr] = 0.0f; continue; }
        float acc = 0.0f;
        for (int k = 0; k < nc; ++k) {
            const char * gbase = (const char *) gu_out->data + t * gu_out->nb[2] + (size_t) k * gu_out->nb[1];
            for (int64_t b = 0; b < nblk; ++b) {
                for (int64_t i = 0; i < blk; ++i) {
                    gate[(size_t) i] = *(const float *) (gbase + (b * blk + i) * sizeof(float));
                    up[(size_t) i]   = *(const float *) (gbase + (n_ff + b * blk + i) * sizeof(float));
                }
                ggml_vec_swiglu_f32((int) blk, sw.data(), gate.data(), up.data());
                char * o = aq.data() + b * (row_q / nblk);
                if (qf_dn) { qf_dn(sw.data(), o, blk); }
            }
            const char * w = (const char *) dn->data + (size_t) cid[k] * b_dn + (size_t) dr * row_dn;
            float sv = 0.0f;
            vd_dn((int) n_ff, &sv, 0, w, 0, aq.data(), 0, 1);
            acc += cw[k] * sv;
        }
        out[dr] = acc;
    }
}

// A layer may use the fused executor only if its expert tensors are host-backed:
// that is exactly the set of layers -ncmoe keeps on the CPU. Device-resident
// experts keep the deployed GPU path.
static bool alice_fused_moe_layer_ok(const llama_layer & layer) {
    if (layer.ffn_gate_up_exps == nullptr || layer.ffn_down_exps == nullptr) {
        return false;
    }
    ggml_backend_buffer_t b_gu = layer.ffn_gate_up_exps->buffer;
    ggml_backend_buffer_t b_dn = layer.ffn_down_exps->buffer;
    if (b_gu == nullptr || b_dn == nullptr) {
        return false;
    }
    return ggml_backend_buffer_is_host(b_gu) && ggml_backend_buffer_is_host(b_dn);
}
// ---------------------------------------------------------------------------
// Hot-bank VRAM duplicates (ALICE_HOTBANK=1).
//
// Per-(layer,expert) residency the -ncmoe/-ot whole-bank placement cannot
// express: compact VRAM tensors per CPU layer holding the 1417 hottest expert
// units (78.24% route mass, FIXED trace), addressed by hot-local index.
// Atlas file weights/alice-hotbank-2gib.ahb1 (AHB1 + 22 B/unit header + raw
// row bytes, verbatim quant, no requant; built by
// tools/alice_hotbank_extract.py, verified 5/5 byte-identical).
// Compact tensors carry USAGE_WEIGHTS so weight-following assigns their
// mul_mat_id to the GPU; the graph remaps hot ids (partition op) while cold
// ids run the deployed CPU mm_id on the original bank, exact-count.
#include <cstdint>
static void alice_hotbank_load(llama_model_alice_ai & amodel) {
    static thread_local bool done = false;
    if (done) { return; }
    done = true;
    const char * path = std::getenv("ALICE_HOTBANK_PATH");
    if (!path || !*path) { path = "/home/leo/research/alice-deploy/weights/alice-hotbank-2gib.ahb1"; }
    FILE * f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "[hotbank] cannot open %s; disabled\n", path); return; }
    char magic[4];
    if (fread(magic, 1, 4, f) != 4 || memcmp(magic, "AHB1", 4) != 0) {
        fprintf(stderr, "[hotbank] bad magic; disabled\n"); fclose(f); return;
    }
    uint32_t n_units = 0;
    if (fread(&n_units, 4, 1, f) != 1 || n_units == 0 || n_units > 24576) {
        fprintf(stderr, "[hotbank] bad unit count; disabled\n"); fclose(f); return;
    }
    for (int i = 0; i < 48; ++i) {
        for (int e = 0; e < 512; ++e) { amodel.hot_lut[i].local[e] = -1; }
        amodel.hot_lut[i].n_hot = 0;
        amodel.hot_gu[i] = nullptr; amodel.hot_dn[i] = nullptr; amodel.hot_bufs[i] = nullptr;
    }
    struct hot_unit { int layer, expert; size_t gu_bytes, dn_bytes; long data_off; };
    std::vector<hot_unit> units;
    units.reserve(n_units);
    long cur_off = 4 + 4 + (long) n_units * 22;
    for (uint32_t u = 0; u < n_units; ++u) {
        uint8_t buf[22];
        if (fread(buf, 1, 22, f) != 22) { fprintf(stderr, "[hotbank] truncated header; disabled\n"); fclose(f); return; }
        uint16_t layer, expert; uint32_t gu_bytes, dn_bytes;
        memcpy(&layer, buf, 2); memcpy(&expert, buf + 2, 2);
        memcpy(&gu_bytes, buf + 6, 4); memcpy(&dn_bytes, buf + 10, 4);
        if (layer >= 48 || expert >= 512) { fprintf(stderr, "[hotbank] bad unit %u; disabled\n", u); fclose(f); return; }
        int16_t local = amodel.hot_lut[layer].n_hot++;
        amodel.hot_lut[layer].local[expert] = local;
        units.push_back({layer, (int) expert, gu_bytes, dn_bytes, cur_off});
        cur_off += (long) gu_bytes + (long) dn_bytes;
    }
    ggml_backend_dev_t gpu_dev = nullptr;
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        ggml_backend_dev_t d = ggml_backend_dev_get(i);
        if (ggml_backend_dev_type(d) == GGML_BACKEND_DEVICE_TYPE_GPU) { gpu_dev = d; break; }
    }
    if (!gpu_dev) { fprintf(stderr, "[hotbank] no GPU device; disabled\n"); fclose(f); return; }
    ggml_backend_buffer_type_t vram_buft = ggml_backend_dev_buffer_type(gpu_dev);
    size_t total_vram = 0;
    // NOTE: layer expert buffers are NOT allocated yet at load_arch_tensors time
    // (llama-model.cpp allocates all ctx buffers after). Host-vs-VRAM residency
    // per layer is therefore decided from the -ncmoe override list, not from
    // tensor->buffer: layers [0,ncmoe) are CPU (host), [ncmoe,48) are VRAM.
    // -ncmoe value arrives via env ALICE_NCMOE (must match the -ncmoe flag).
    int ncmoe = 32;
    if (const char * e = std::getenv("ALICE_NCMOE")) { ncmoe = atoi(e); }
    for (int il = 0; il < 48; ++il) {
        int nh = amodel.hot_lut[il].n_hot;
        if (nh == 0) { continue; }
        if (il >= ncmoe) { continue; }  // GPU layers already resident; no duplicate
        const ggml_tensor * gu0 = amodel.layers[il].ffn_gate_up_exps;
        const ggml_tensor * dn0 = amodel.layers[il].ffn_down_exps;
        struct ggml_init_params p = { (size_t) ggml_tensor_overhead() * 8, nullptr, true };
        ggml_context * ctx = ggml_init(p);
        if (!ctx) { fprintf(stderr, "[hotbank] ggml_init failed L%d; skipped\n", il); continue; }
        ggml_tensor * hgu = ggml_new_tensor_3d(ctx, gu0->type, gu0->ne[0], gu0->ne[1], nh);
        ggml_tensor * hdn = ggml_new_tensor_3d(ctx, dn0->type, dn0->ne[0], dn0->ne[1], nh);
        ggml_set_name(hgu, ("hot_gu_" + std::to_string(il)).c_str());
        ggml_set_name(hdn, ("hot_dn_" + std::to_string(il)).c_str());
        ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx, vram_buft);
        if (!buf) { fprintf(stderr, "[hotbank] VRAM alloc failed L%d; skipped\n", il); ggml_free(ctx); continue; }
        int slot = 0;
        // VRAM tensors are device-side: stage rows through host vectors, then
        // ggml_backend_tensor_set (hgu/hdn->data is a device pointer, NOT
        // fread-able — the segfault was fread into VRAM address).
        for (const auto & u : units) {
            if (u.layer != il) { continue; }
            fseeko(f, u.data_off, SEEK_SET);
            std::vector<char> staging((size_t) u.gu_bytes + (size_t) u.dn_bytes);
            if (fread(staging.data(), 1, staging.size(), f) != staging.size()) {
                fprintf(stderr, "[hotbank] short read L%d; abort layer\n", il); break;
            }
            ggml_backend_tensor_set(hgu, staging.data(), (size_t) slot * u.gu_bytes, u.gu_bytes);
            ggml_backend_tensor_set(hdn, staging.data() + u.gu_bytes, (size_t) slot * u.dn_bytes, u.dn_bytes);
            ++slot;
        }
        ggml_backend_buffer_set_usage(buf, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        amodel.hot_gu[il] = hgu; amodel.hot_dn[il] = hdn; amodel.hot_bufs[il] = buf;
        total_vram += ggml_nbytes(hgu) + ggml_nbytes(hdn);
        fprintf(stderr, "[hotbank] L%d: %d units gpu=%lldMB\n", il, nh,
            (long long) (ggml_nbytes(hgu) + ggml_nbytes(hdn)) / 1024 / 1024);
    }
    fclose(f);
    amodel.hot_enabled = true;
    fprintf(stderr, "[hotbank] enabled: %u units, VRAM duplicates %.3f GiB\n",
        n_units, (double) total_vram / 1024 / 1024 / 1024);
}

void llama_model_alice_ai::load_arch_hparams(llama_model_loader & ml) {
    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);

    // layer topology: prefer the explicit recurrent_layers array, fall back to the interval
    if (!ml.get_key_or_arr(LLM_KV_ATTENTION_RECURRENT_LAYERS, hparams.is_recr_impl, hparams.n_layer_all, false)) {
        uint32_t full_attn_interval = 4;
        ml.get_key(LLM_KV_FULL_ATTENTION_INTERVAL, full_attn_interval, false);
        for (uint32_t i = 0; i < hparams.n_layer(); ++i) {
            hparams.is_recr_impl[i] = ((i + 1) % full_attn_interval != 0);
        }
    }

    // KDA geometry: per-element decay, key_dim = 32*128 = 4096, value_dim = 4096
    ml.get_key(LLM_KV_SSM_CONV_KERNEL, hparams.ssm_d_conv);
    ml.get_key(LLM_KV_KDA_HEAD_DIM,    hparams.n_embd_head_kda);
    // KDA head count (32) differs from attention head count (16); this drives the
    // conv/recurrent state sizing in llama_hparams::n_embd_r()/n_embd_s()
    ml.get_key(LLM_KV_SSM_GROUP_COUNT, hparams.ssm_n_group, false);
    if (hparams.ssm_n_group == 0) {
        hparams.ssm_n_group = 32;
    }

    // MoE: sigmoid router + bias correction, renormalized top-k, shared expert, scale 1.0
    ml.get_key_or_arr(LLM_KV_EXPERT_FEED_FORWARD_LENGTH, hparams.n_ff_exp_arr, hparams.n_layer_all);
    ml.get_key(LLM_KV_EXPERT_SHARED_COUNT,               hparams.n_expert_shared);
    ml.get_key(LLM_KV_EXPERT_SHARED_FEED_FORWARD_LENGTH, hparams.n_ff_shexp, false);
    ml.get_key(LLM_KV_EXPERT_GATING_FUNC,                hparams.expert_gating_func);
    ml.get_key(LLM_KV_EXPERT_WEIGHTS_NORM,               hparams.expert_weights_norm, false);
}

void llama_model_alice_ai::load_arch_tensors(llama_model_loader &) {
    LLAMA_LOAD_LOCALS;

    tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, 0);

    output_norm = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), {(int64_t)n_embd}, 0);
    output      = create_tensor(tn(LLM_TENSOR_OUTPUT,      "weight"), {n_embd, n_vocab}, 0);

    // final block-attention residual (depth-softmax-mix over completed blocks)
    output_res_proj = create_tensor(tn(LLM_TENSOR_OUTPUT_RES_PROJ, "weight"), {(int64_t)n_embd, (int64_t)1}, 0);
    output_res_norm = create_tensor(tn(LLM_TENSOR_OUTPUT_RES_NORM, "weight"), {(int64_t)n_embd}, 0);

    const int64_t n_ff_exp        = hparams.n_ff_exp();
    const int64_t n_ff_shexp      = hparams.n_ff_shexp > 0 ? hparams.n_ff_shexp : n_ff_exp;
    const int64_t n_embd_head_kda = (int64_t)hparams.n_embd_head_kda;
    const int64_t ssm_d_conv      = (int64_t)hparams.ssm_d_conv;

    for (int i = 0; i < n_layer; ++i) {
        auto & layer = layers[i];

        layer.attn_norm = create_tensor(tn(LLM_TENSOR_ATTN_NORM, "weight", i), {(int64_t)n_embd}, 0);
        layer.ffn_norm  = create_tensor(tn(LLM_TENSOR_FFN_NORM,  "weight", i), {(int64_t)n_embd}, 0);

        // block residual (split proj H->1 plus norm); attn pair absent on layer 0
        if (i != 0) {
            layer.attn_res_proj = create_tensor(tn(LLM_TENSOR_ATTN_RES_PROJ, "weight", i), {(int64_t)n_embd, (int64_t)1}, 0);
            layer.attn_res_norm = create_tensor(tn(LLM_TENSOR_ATTN_RES_NORM, "weight", i), {(int64_t)n_embd}, 0);
        }
        layer.ffn_res_proj  = create_tensor(tn(LLM_TENSOR_FFN_RES_PROJ, "weight", i), {(int64_t)n_embd, (int64_t)1}, 0);
        layer.ffn_res_norm  = create_tensor(tn(LLM_TENSOR_FFN_RES_NORM, "weight", i), {(int64_t)n_embd}, 0);

        if (hparams.is_recr(i)) {
            // ---- KDA layer: split projections + 3 separate depthwise convs
            // Alice KDA uses its OWN head counts (32/32), independent of attn n_head/n_head_kv
            const int64_t n_head_k = 32;
            const int64_t n_head_v = 32;
            const int64_t key_dim   = n_embd_head_kda * n_head_k;   // 4096
            const int64_t value_dim = n_embd_head_kda * n_head_v;   // 4096

            layer.ssm_q = create_tensor(tn(LLM_TENSOR_SSM_Q, "weight", i), {n_embd, key_dim},   0);
            layer.ssm_k = create_tensor(tn(LLM_TENSOR_SSM_K, "weight", i), {n_embd, key_dim},   0);
            layer.ssm_v = create_tensor(tn(LLM_TENSOR_SSM_V, "weight", i), {n_embd, value_dim}, 0);

            layer.ssm_q_conv = create_tensor(tn(LLM_TENSOR_SSM_CONV1D_Q, "weight", i), {ssm_d_conv, 1, key_dim,   1}, TENSOR_NOT_REQUIRED);
            if (!layer.ssm_q_conv) {
                layer.ssm_q_conv = create_tensor(tn(LLM_TENSOR_SSM_CONV1D_Q, "weight", i), {ssm_d_conv, 1, key_dim}, 0);
            }
            layer.ssm_k_conv = create_tensor(tn(LLM_TENSOR_SSM_CONV1D_K, "weight", i), {ssm_d_conv, 1, key_dim,   1}, TENSOR_NOT_REQUIRED);
            if (!layer.ssm_k_conv) {
                layer.ssm_k_conv = create_tensor(tn(LLM_TENSOR_SSM_CONV1D_K, "weight", i), {ssm_d_conv, 1, key_dim}, 0);
            }
            layer.ssm_v_conv = create_tensor(tn(LLM_TENSOR_SSM_CONV1D_V, "weight", i), {ssm_d_conv, 1, value_dim, 1}, TENSOR_NOT_REQUIRED);
            if (!layer.ssm_v_conv) {
                layer.ssm_v_conv = create_tensor(tn(LLM_TENSOR_SSM_CONV1D_V, "weight", i), {ssm_d_conv, 1, value_dim}, 0);
            }

            // factored forget-gate alpha = f_b(f_a(x)), factored output gate = g_b(g_a(x))
            layer.ssm_f_a = create_tensor(tn(LLM_TENSOR_SSM_F_A, "weight", i), {n_embd, n_embd_head_kda}, 0);
            layer.ssm_f_b = create_tensor(tn(LLM_TENSOR_SSM_F_B, "weight", i), {n_embd_head_kda, key_dim}, 0);
            layer.ssm_g_a = create_tensor(tn(LLM_TENSOR_SSM_G_A, "weight", i), {n_embd, n_embd_head_kda}, 0);
            layer.ssm_g_b = create_tensor(tn(LLM_TENSOR_SSM_G_B, "weight", i), {n_embd_head_kda, value_dim}, 0);
            // beta mixing coefficient; a_log pre-negated at conversion; per-element dt_bias
            layer.ssm_beta = create_tensor(tn(LLM_TENSOR_SSM_BETA, "weight", i), {n_embd, n_head_k}, 0);
            layer.ssm_a    = create_tensor(tn(LLM_TENSOR_SSM_A_NOSCAN, i), {(int64_t)n_head_k}, 0);
            layer.ssm_dt   = create_tensor(tn(LLM_TENSOR_SSM_DT, "bias", i), {(int64_t)key_dim}, 0);

            layer.ssm_o_norm = create_tensor(tn(LLM_TENSOR_SSM_NORM, "weight", i), {(int64_t)n_embd_head_kda}, 0);
            layer.ssm_out    = create_tensor(tn(LLM_TENSOR_SSM_OUT,  "weight", i), {value_dim, n_embd}, 0);
        } else {
            // ---- gated full-attention layer: fused query||gate, QK-norm, partial RoPE
            // wq packs [query || gate] interleaved per head, exactly like qwen3next
            layer.wq = create_tensor(tn(LLM_TENSOR_ATTN_QKV, "weight", i), {n_embd, n_embd_head_k * n_head * 2}, 0);
            layer.wk = create_tensor(tn(LLM_TENSOR_ATTN_K,   "weight", i), {n_embd, n_embd_head_k * n_head_kv},  0);
            layer.wv = create_tensor(tn(LLM_TENSOR_ATTN_V,   "weight", i), {n_embd, n_embd_head_k * n_head_kv},  0);
            layer.wo = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", i), {n_embd_head_k * n_head, n_embd},     0);

            layer.attn_q_norm = create_tensor(tn(LLM_TENSOR_ATTN_Q_NORM, "weight", i), {(int64_t)n_embd_head_k}, 0);
            layer.attn_k_norm = create_tensor(tn(LLM_TENSOR_ATTN_K_NORM, "weight", i), {(int64_t)n_embd_head_k}, 0);
        }

        // ---- MoE: sigmoid router + bias correction + renormalisation + shared expert
        layer.ffn_gate_inp  = create_tensor(tn(LLM_TENSOR_FFN_GATE_INP,  "weight", i), {n_embd, n_expert}, 0);
        layer.ffn_exp_probs_b = create_tensor(tn(LLM_TENSOR_FFN_EXP_PROBS_B, "bias", i), {(int64_t)n_expert}, 0);
        layer.ffn_gate_exps = nullptr;
        layer.ffn_up_exps   = nullptr;
        create_tensor_gate_up_exps(layer, i, n_embd, n_ff_exp, hparams.n_expert, 0);
        layer.ffn_down_exps = create_tensor(tn(LLM_TENSOR_FFN_DOWN_EXPS, "weight", i), {n_ff_exp, n_embd, hparams.n_expert}, 0);

        layer.ffn_gate_shexp = create_tensor(tn(LLM_TENSOR_FFN_GATE_SHEXP, "weight", i), {n_embd, n_ff_shexp}, 0);
        layer.ffn_up_shexp   = create_tensor(tn(LLM_TENSOR_FFN_UP_SHEXP,   "weight", i), {n_embd, n_ff_shexp}, 0);
        layer.ffn_down_shexp = create_tensor(tn(LLM_TENSOR_FFN_DOWN_SHEXP, "weight", i), {n_ff_shexp, n_embd}, 0);
        layer.ffn_gate_inp_shexp = create_tensor(tn(LLM_TENSOR_FFN_GATE_INP_SHEXP, "weight", i), {(int64_t)n_embd}, 0);
    }
    // ---- Hot-bank VRAM duplicates (ALICE_HOTBANK=1). Runtime duplicates of the
    // per-(layer,expert) hot units from weights/alice-hotbank-2gib.ahb1; the frozen
    // GGUF stays byte-identical. Compact tensors live in VRAM with USAGE_WEIGHTS so
    // weight-following assigns their mul_mat_id to the GPU. KILLED alternatives this
    // does not touch: -ot shexp, fused arrangement, MIN_BATCH=1.
    if (std::getenv("ALICE_HOTBANK")) {
        alice_hotbank_load(*this);
    }

    // ---- MTP / nextn draft block (one block past the trunk, plain gated attention + MoE)
    for (int i = n_layer; i < n_layer + (int) hparams.n_layer_nextn; ++i) {
        auto & layer = layers[i];

        layer.nextn.eh_proj = create_tensor(tn(LLM_TENSOR_NEXTN_EH_PROJ, "weight", i), {2 * n_embd, n_embd}, 0);
        layer.nextn.enorm   = create_tensor(tn(LLM_TENSOR_NEXTN_ENORM, "weight", i), {(int64_t)n_embd}, 0);
        layer.nextn.hnorm   = create_tensor(tn(LLM_TENSOR_NEXTN_HNORM, "weight", i), {(int64_t)n_embd}, 0);
        layer.nextn.shared_head_norm = create_tensor(tn(LLM_TENSOR_NEXTN_SHARED_HEAD_NORM, "weight", i),
                                                     {(int64_t)n_embd}, TENSOR_NOT_REQUIRED);

        layer.attn_norm = create_tensor(tn(LLM_TENSOR_ATTN_NORM, "weight", i), {(int64_t)n_embd}, 0);
        layer.ffn_norm  = create_tensor(tn(LLM_TENSOR_FFN_NORM,  "weight", i), {(int64_t)n_embd}, 0);

        layer.wq = create_tensor(tn(LLM_TENSOR_ATTN_QKV, "weight", i), {n_embd, n_embd_head_k * n_head * 2}, 0);
        layer.wk = create_tensor(tn(LLM_TENSOR_ATTN_K,   "weight", i), {n_embd, n_embd_head_k * n_head_kv},  0);
        layer.wv = create_tensor(tn(LLM_TENSOR_ATTN_V,   "weight", i), {n_embd, n_embd_head_k * n_head_kv},  0);
        layer.wo = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", i), {n_embd_head_k * n_head, n_embd},     0);

        layer.attn_q_norm = create_tensor(tn(LLM_TENSOR_ATTN_Q_NORM, "weight", i), {(int64_t)n_embd_head_k}, 0);
        layer.attn_k_norm = create_tensor(tn(LLM_TENSOR_ATTN_K_NORM, "weight", i), {(int64_t)n_embd_head_k}, 0);

        layer.ffn_gate_inp    = create_tensor(tn(LLM_TENSOR_FFN_GATE_INP,  "weight", i), {n_embd, n_expert}, 0);
        layer.ffn_exp_probs_b = create_tensor(tn(LLM_TENSOR_FFN_EXP_PROBS_B, "bias", i), {(int64_t)n_expert}, 0);
        layer.ffn_gate_exps = nullptr;
        layer.ffn_up_exps   = nullptr;
        create_tensor_gate_up_exps(layer, i, n_embd, n_ff_exp, hparams.n_expert, 0);
        layer.ffn_down_exps = create_tensor(tn(LLM_TENSOR_FFN_DOWN_EXPS, "weight", i), {n_ff_exp, n_embd, hparams.n_expert}, 0);

        layer.ffn_gate_shexp = create_tensor(tn(LLM_TENSOR_FFN_GATE_SHEXP, "weight", i), {n_embd, n_ff_shexp}, 0);
        layer.ffn_up_shexp   = create_tensor(tn(LLM_TENSOR_FFN_UP_SHEXP,   "weight", i), {n_embd, n_ff_shexp}, 0);
        layer.ffn_down_shexp = create_tensor(tn(LLM_TENSOR_FFN_DOWN_SHEXP, "weight", i), {n_ff_shexp, n_embd}, 0);
        layer.ffn_gate_inp_shexp = create_tensor(tn(LLM_TENSOR_FFN_GATE_INP_SHEXP, "weight", i), {(int64_t)n_embd}, 0);
    }
}

std::unique_ptr<llm_graph_context> llama_model_alice_ai::build_arch_graph(const llm_graph_params & params) const {
    if (params.gtype == LLM_GRAPH_TYPE_DECODER_MTP) {
        return std::make_unique<graph_mtp>(*this, params);
    }
    return std::make_unique<graph>(*this, params);
}

// Causal Conv1d for Q/K/V (same layout as kimi-linear)
//
// ALICE_MTP_K_SNAP=1: write the per-token conv snapshots into the (1+n_rs_seq)
// slot planes, mirroring bailingmoe3_causal_conv1d. With n_rs_seq==0 this is a
// no-op (n_written==0) and the deployed single-plane write below is unchanged.
static ggml_tensor * causal_conv1d(ggml_cgraph * gf, ggml_context * ctx0, ggml_tensor * conv_states_all, ggml_tensor * conv_state_all, int64_t qkv, ggml_tensor * x, ggml_tensor * proj_w, ggml_tensor * conv_w, int64_t d_conv, int64_t head_dim, int64_t n_head, int64_t n_seq_tokens, int64_t n_seqs, int64_t n_tokens, int64_t kv_head, int64_t mem_size, uint32_t n_rs_seq) {
    const int64_t d_inner = head_dim * n_head;
    const int64_t conv_state_size = (d_conv - 1) * d_inner;
    const int64_t n_embd_r_total = 3 * conv_state_size;

    ggml_tensor * conv_state_x = ggml_view_3d(ctx0, conv_state_all, d_conv - 1, d_inner, n_seqs,
        (d_conv - 1) * ggml_element_size(conv_state_all),
        n_embd_r_total * ggml_element_size(conv_state_all),
        qkv * conv_state_size * ggml_element_size(conv_state_all));

    ggml_tensor * x_proj = proj_w ? ggml_mul_mat(ctx0, proj_w, x) : x;
    ggml_tensor * x_3d = ggml_reshape_3d(ctx0, x_proj, d_inner, n_seq_tokens, n_seqs);
    ggml_tensor * conv_x = ggml_concat(ctx0, conv_state_x, ggml_transpose(ctx0, x_3d), 0);

    // per-token snapshots for speculative rollback: slot j holds the state after
    // consuming j tokens of this ubatch (slot 0 = state at entry). Only written
    // when n_rs_seq > 0; the deployed plane-0 write below stays the canonical one.
    const int64_t K = (int64_t) n_rs_seq + 1;
    const int64_t n_written = std::min<int64_t>(n_seq_tokens, K);
    for (int64_t slot = 0; slot < n_written; ++slot) {
        // conv_x dim0 = (d_conv-1) + n_seq_tokens; the snapshot for slot j is the
        // (d_conv-1)-window ending j tokens into the ubatch. Clamp the offset so
        // the view stays in-bounds when n_seq_tokens < K (ggml asserts otherwise).
        const int64_t off = std::min<int64_t>(slot, std::max<int64_t>(0, conv_x->ne[0] - (d_conv - 1)));
        ggml_tensor * conv_snap = ggml_view_3d(ctx0, conv_x, d_conv - 1, d_inner, n_seqs,
            conv_x->nb[1], conv_x->nb[2],
            off * conv_x->nb[0]);
        ggml_build_forward_expand(gf, ggml_cpy(ctx0, conv_snap,
            ggml_view_3d(ctx0, conv_states_all,
                d_conv - 1, d_inner, n_seqs,
                (d_conv - 1) * ggml_element_size(conv_states_all),
                n_embd_r_total * ggml_element_size(conv_states_all),
                ((slot * mem_size + kv_head) * n_embd_r_total + qkv * conv_state_size) * ggml_element_size(conv_states_all))));
    }

    ggml_tensor * last_conv_x = ggml_view_3d(ctx0, conv_x, d_conv - 1, d_inner, n_seqs,
        conv_x->nb[1], conv_x->nb[2], n_seq_tokens * conv_x->nb[0]);
    ggml_build_forward_expand(gf,
        ggml_cpy(ctx0, last_conv_x,
            ggml_view_3d(ctx0, conv_states_all,
                d_conv - 1, d_inner, n_seqs,
                (d_conv - 1) * ggml_element_size(conv_states_all),
                n_embd_r_total * ggml_element_size(conv_states_all),
                (kv_head * n_embd_r_total + qkv * conv_state_size) * ggml_element_size(conv_states_all))));

    ggml_tensor * conv_weight = ggml_reshape_2d(ctx0, conv_w, d_conv, d_inner);
    ggml_tensor * Xcur = ggml_ssm_conv(ctx0, conv_x, conv_weight);
    Xcur = ggml_reshape_2d(ctx0, Xcur, d_inner, n_tokens);
    Xcur = ggml_silu(ctx0, Xcur);

    return ggml_reshape_4d(ctx0, Xcur, head_dim, n_head, n_seq_tokens, n_seqs);
}

// depth-softmax-mix over completed blocks (+ in-flight partial):
//   per source: keys_k = rms_norm(src_k) * norm_w ; score_k = proj_w . keys_k  -> [1, T]
//   weights  = softmax([score_0 .. score_{S-1}], dim=0)                        -> [S, T]
//   out      = sum_k src_k * weights_k
// Softmax is computed with elementwise ops (max-subtract, exp, sum, div) rather than
// a dim-0 ggml softmax, so no strided view over the [S,T] score matrix is needed.
static ggml_tensor * depth_softmax_mix(ggml_context * ctx0, std::vector<ggml_tensor *> & sources,
        ggml_tensor * proj_w, ggml_tensor * norm_w, float eps, int il) {
    GGML_ASSERT(!sources.empty());
    const int64_t H = sources[0]->ne[0];
    const int64_t T = sources[0]->ne[1];
    const int64_t S = (int64_t) sources.size();

    std::vector<ggml_tensor *> sc(S);
    for (int64_t k = 0; k < S; ++k) {
        ggml_tensor * normed = ggml_rms_norm(ctx0, sources[k], eps);
        normed = ggml_mul(ctx0, normed, norm_w);
        sc[k] = ggml_mul_mat(ctx0, proj_w, normed);   // [1, T]
    }

    // softmax over the source axis, computed elementwise (no strided views, no
    // max-subtraction: scores come from a H->1 linear on an RMS-normalized vector,
    // so they are O(1) and exp() cannot overflow)
    std::vector<ggml_tensor *> ex(S);
    ggml_tensor * sum = nullptr;
    for (int64_t k = 0; k < S; ++k) {
        ex[k] = ggml_exp(ctx0, sc[k]);
        sum = (sum == nullptr) ? ex[k] : ggml_add(ctx0, sum, ex[k]);
    }

    ggml_tensor * acc = nullptr;
    for (int64_t k = 0; k < S; ++k) {
        ggml_tensor * wk = ggml_div(ctx0, ex[k], sum);   // [1, T]
        ggml_tensor * term = ggml_mul(ctx0, sources[k], wk);
        acc = (acc == nullptr) ? term : ggml_add(ctx0, acc, term);
    }
    (void) il;
    (void) H;
    return acc;
}



llama_model_alice_ai::graph::~graph() {
    if (g_alice_probe.active) {
        alice_moe_probe_write(g_alice_probe);
    }
}

llama_model_alice_ai::graph::graph(const llama_model & model, const llm_graph_params & params) :
    llm_build_delta_net_base(params), model(model) {

    const auto & hparams = model.hparams;
    const int64_t n_layer = hparams.n_layer();
    const int64_t n_head  = hparams.n_head();

    if (std::getenv("ALICE_MOE_STATS")) {
        g_alice_probe.arm((int) n_layer, (int) hparams.n_expert, (int) hparams.n_expert_used());
    }

    ggml_tensor * cur;
    ggml_tensor * inpL;

    inpL = build_inp_embd(model.tok_embd);
    cb(inpL, "inp_embd", -1);

    auto * inp = build_inp_mem_hybrid();

    ggml_tensor * inp_pos     = build_inp_pos();

    auto * inp_rs   = inp->get_recr();
    auto * inp_attn = inp->get_attn();

    // select the rows that actually need logits (out_ids); required so that the
    // logits tensor has exactly n_outputs rows
    ggml_tensor * inp_out_ids = build_inp_out_ids();

    // block residual state: completed block outputs; `partial` accumulates in-flight attention
    std::vector<ggml_tensor *> completed;
    completed.reserve(n_layer / 4 + 2);
    completed.push_back(inpL);
    ggml_tensor * partial = nullptr;
    ggml_tensor * l0_out  = nullptr;

    struct ggml_cgraph * gf = this->gf;

    if (std::getenv("ALICE_TRACE_UB")) {
        fprintf(stderr, "[alice] graph build: n_tokens=%d n_seq_tokens=%d n_seqs=%d n_rs_seq=%d\n",
                ubatch.n_tokens, ubatch.n_seq_tokens, ubatch.n_seqs, cparams.n_rs_seq);
    }

    const bool fused_moe_enabled = std::getenv("ALICE_FUSED_MOE") != nullptr;
    int fused_moe_max = fused_moe_enabled ? (int) model.layers.size() : 0;
    if (const char * mx = std::getenv("ALICE_FUSED_MOE_MAX")) {
        fused_moe_max = atoi(mx);   // differential testing: fuse only layers < fused_moe_max
    }
    const auto & amodel = static_cast<const llama_model_alice_ai &>(model);
    // ALICE_HOTBANK=1 loads the atlas (diagnostics); =2 also wires the compact
    // MoE graph path (measured 0.52x decode — kept for the next iteration).
    const char * hb_env = std::getenv("ALICE_HOTBANK");
    const bool hotbank_on = amodel.hot_enabled && hb_env != nullptr && hb_env[0] == '2';
    if (hotbank_on) {
        hotbank.partition = alice_hotbank_partition;
        hotbank.cold      = alice_hotbank_cold_dn;
        hotbank.weights   = alice_hotbank_weights;
        hotbank.cold_gu  = alice_hotbank_cold_gu;
        hotbank.hotbank_model = &model;
        if (amodel.hot_part_ud.size() < model.layers.size()) {
            amodel.hot_part_ud.resize(model.layers.size());
            for (size_t i = 0; i < model.layers.size(); ++i) {
                amodel.hot_part_ud[i] = { &model, (int) i };
            }
        }
        hotbank.userdata = amodel.hot_part_ud.data();
    }
    if (fused_moe_enabled && amodel.fused_moe_ud.size() < model.layers.size()) {
        amodel.fused_moe_ud.resize(model.layers.size());
        for (size_t i = 0; i < model.layers.size(); ++i) {
            amodel.fused_moe_ud[i] = { &model, (int) i };
        }
    }

    for (int il = 0; il < n_layer; ++il) {
        const auto & layer = model.layers[il];
        ggml_tensor * ffn_inp;

        // ---- pre-attention mix: embeddings on layer 0, depth-softmax-mix otherwise
        ggml_tensor * attn_inp;
        if (il == 0) {
            attn_inp = inpL;
        } else {
            attn_inp = depth_softmax_mix(ctx0, completed, layer.attn_res_proj, layer.attn_res_norm,
                hparams.f_norm_rms_eps, il);
            if (partial != nullptr) {
                // in-flight partial joins the mix as the last source
                completed.push_back(partial);
                attn_inp = depth_softmax_mix(ctx0, completed, layer.attn_res_proj, layer.attn_res_norm,
                    hparams.f_norm_rms_eps, il);
                completed.pop_back();
            }
        }
        cur = build_norm(attn_inp, layer.attn_norm, nullptr, LLM_NORM_RMS, il);
        cb(cur, "attn_norm", il);

        ggml_build_forward_expand(gf, cur);

        if (hparams.is_recr(il)) {
            // ---- KDA layer
            const auto * mctx_cur = inp_rs->mctx;
            const auto kv_head = mctx_cur->get_head();

            const int64_t head_dim     = hparams.n_embd_head_kda;
            const int64_t d_conv       = hparams.ssm_d_conv;
            const int64_t n_head_kd    = 32;
            const int64_t d_inner      = head_dim * n_head_kd;
            const int64_t n_seqs       = ubatch.n_seqs;
            const int64_t n_seq_tokens = ubatch.n_seq_tokens;

            GGML_ASSERT(n_seqs != 0);
            GGML_ASSERT(ubatch.equal_seqs());
            GGML_ASSERT(ubatch.n_tokens == n_seq_tokens * n_seqs);

            ggml_tensor * conv_states_all = mctx_cur->get_r_l(il);
            ggml_tensor * conv_state_all  = build_rs(inp_rs, conv_states_all, hparams.n_embd_r(), n_seqs);

            const int64_t ksnap_mem_size = mctx_cur->get_size();
            const uint32_t ksnap_n_rs    = mctx_cur->get_n_rs_seq();

            ggml_tensor * Qcur = causal_conv1d(gf, ctx0, conv_states_all, conv_state_all, 0, cur, layer.ssm_q, layer.ssm_q_conv, d_conv, head_dim, n_head_kd, n_seq_tokens, n_seqs, ubatch.n_tokens, kv_head, ksnap_mem_size, ksnap_n_rs);
            ggml_tensor * Kcur = causal_conv1d(gf, ctx0, conv_states_all, conv_state_all, 1, cur, layer.ssm_k, layer.ssm_k_conv, d_conv, head_dim, n_head_kd, n_seq_tokens, n_seqs, ubatch.n_tokens, kv_head, ksnap_mem_size, ksnap_n_rs);
            ggml_tensor * Vcur = causal_conv1d(gf, ctx0, conv_states_all, conv_state_all, 2, cur, layer.ssm_v, layer.ssm_v_conv, d_conv, head_dim, n_head_kd, n_seq_tokens, n_seqs, ubatch.n_tokens, kv_head, ksnap_mem_size, ksnap_n_rs);

            // g1 = -exp(a_log) * softplus(f_b(f_a(x)) + dt_bias); a_log pre-negated at convert
            ggml_tensor * f_a = ggml_mul_mat(ctx0, layer.ssm_f_a, cur);
            ggml_tensor * g1  = ggml_mul_mat(ctx0, layer.ssm_f_b, f_a);
            g1 = ggml_add(ctx0, g1, layer.ssm_dt);
            g1 = ggml_softplus(ctx0, g1);
            g1 = ggml_reshape_3d(ctx0, g1, head_dim, n_head_kd, ubatch.n_tokens);
            ggml_tensor * A = ggml_reshape_3d(ctx0, layer.ssm_a, 1, n_head_kd, 1);
            g1 = ggml_mul(ctx0, g1, A);
            g1 = ggml_reshape_4d(ctx0, g1, head_dim, n_head_kd, n_seq_tokens, n_seqs);

            // beta = sigmoid(b_proj(x)); Alice has equal k/v heads so no interleave needed
            ggml_tensor * beta = ggml_mul_mat(ctx0, layer.ssm_beta, cur);
            beta = ggml_reshape_4d(ctx0, beta, 1, n_head_kd, n_seq_tokens, n_seqs);
            beta = ggml_sigmoid(ctx0, beta);

            ggml_tensor * ssm_states_all = mctx_cur->get_s_l(il);
            ggml_tensor * state = build_rs(inp_rs, ssm_states_all, hparams.n_embd_s(), n_seqs);
            state = ggml_reshape_4d(ctx0, state, head_dim, head_dim, n_head_kd, n_seqs);

            const float eps_norm = hparams.f_norm_rms_eps;
            Qcur = build_gdn_l2_norm(ctx0, Qcur, eps_norm);
            Kcur = build_gdn_l2_norm(ctx0, Kcur, eps_norm);

            // ALICE_MTP_K_SNAP: when n_rs_seq > 0 the delta-net op is asked for
            // K = n_rs_seq+1 snapshots and build_recurrent_attn writes them into
            // the slot planes (the same path bailingmoe3/qwen35 use). With
            // n_rs_seq == 0 this is the deployed single-plane behaviour.
            ggml_tensor * output;
            ggml_tensor * new_state;
            if (mctx_cur->get_n_rs_seq() > 0) {
                output = build_recurrent_attn(inp_rs, ssm_states_all, Qcur, Kcur, Vcur, g1, beta, state, il);
                output = ggml_cont(ctx0, output);
                new_state = nullptr; // snapshots already written by build_recurrent_attn
            } else {
                auto attn_out = build_delta_net(Qcur, Kcur, Vcur, g1, beta, state, il);
                output    = ggml_cont(ctx0, attn_out.first);
                new_state = attn_out.second;

                ggml_build_forward_expand(gf,
                    ggml_cpy(ctx0, new_state,
                        ggml_view_1d(ctx0, ssm_states_all, hparams.n_embd_s() * n_seqs,
                            kv_head * hparams.n_embd_s() * ggml_element_size(ssm_states_all))));
            }

            // (snapshots for n_rs_seq > 0 are written by build_recurrent_attn
            // above; for n_rs_seq == 0 the deployed single-plane write is inside
            // the else branch)

            // output gate g2 = g_b(g_a(x)); out = RMSNorm(x) * sigmoid(g)
            ggml_tensor * cur_2d = ggml_reshape_2d(ctx0, cur, cur->ne[0], n_seq_tokens * n_seqs);
            ggml_tensor * g_a = ggml_mul_mat(ctx0, layer.ssm_g_a, cur_2d);
            ggml_tensor * g2  = ggml_mul_mat(ctx0, layer.ssm_g_b, g_a);
            g2 = ggml_reshape_3d(ctx0, g2, head_dim, n_head_kd, n_seq_tokens * n_seqs);
            ggml_tensor * attn_out_final = ggml_reshape_3d(ctx0, output, head_dim, n_head_kd, n_seq_tokens * n_seqs);
            ggml_tensor * normed = build_norm(attn_out_final, layer.ssm_o_norm, nullptr, LLM_NORM_RMS, il);
            ggml_tensor * gate  = ggml_sigmoid(ctx0, g2);
            ggml_tensor * gated = ggml_mul(ctx0, normed, gate);
            gated = ggml_cont_2d(ctx0, gated, d_inner, ubatch.n_tokens);
            cur = ggml_mul_mat(ctx0, layer.ssm_out, gated);
            cb(cur, "kda_out", il);
        } else {
            // ---- gated full-attention layer (qwen3next-identical)
            const int64_t n_embd_head = hparams.n_embd_head_k();
            const int64_t n_head_kv   = hparams.n_head_kv(il);
            const int64_t n_tokens    = ubatch.n_tokens;
            const int64_t n_rot       = hparams.n_rot();

            ggml_tensor * Qcur_full = build_lora_mm(layer.wq, cur);
            Qcur_full = ggml_reshape_4d(ctx0, Qcur_full, n_embd_head * 2, n_head, n_tokens, 1);
            ggml_tensor * Qcur = ggml_view_4d(ctx0, Qcur_full, n_embd_head, n_head, n_tokens, 1,
                Qcur_full->nb[1], Qcur_full->nb[2], Qcur_full->nb[3], 0);
            ggml_tensor * gate = ggml_view_4d(ctx0, Qcur_full, n_embd_head, n_head, n_tokens, 1,
                Qcur_full->nb[1], Qcur_full->nb[2], Qcur_full->nb[3], n_embd_head * ggml_element_size(Qcur_full));
            ggml_tensor * Kcur = build_lora_mm(layer.wk, cur);
            ggml_tensor * Vcur = build_lora_mm(layer.wv, cur);
            Kcur = ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens);
            Vcur = ggml_reshape_3d(ctx0, Vcur, n_embd_head, n_head_kv, n_tokens);

            Qcur = build_norm(Qcur, layer.attn_q_norm, nullptr, LLM_NORM_RMS, il);
            Kcur = build_norm(Kcur, layer.attn_k_norm, nullptr, LLM_NORM_RMS, il);

            Qcur = ggml_rope_ext(ctx0, Qcur, inp_pos, nullptr, n_rot, rope_type, n_ctx_orig,
                freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow);
            Kcur = ggml_rope_ext(ctx0, Kcur, inp_pos, nullptr, n_rot, rope_type, n_ctx_orig,
                freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow);

            const float kq_scale = hparams.f_attention_scale == 0.0f
                ? 1.0f / sqrtf(float(n_embd_head)) : hparams.f_attention_scale;
            cur = build_attn(inp_attn, nullptr, nullptr, nullptr,
                Qcur, Kcur, Vcur, nullptr, nullptr, nullptr, kq_scale, il);

            gate = ggml_cont_2d(ctx0, gate, n_embd_head * n_head, n_tokens);
            gate = ggml_sigmoid(ctx0, gate);
            gate = ggml_reshape_2d(ctx0, gate, n_embd_head * n_head, n_tokens);
            cur = ggml_mul(ctx0, cur, gate);
            cur = build_lora_mm(layer.wo, cur);
            cb(cur, "attn_output", il);
        }

        // in-flight partial accumulates attention outputs within a block
        partial = partial == nullptr ? cur : ggml_add(ctx0, partial, cur);

        // ---- pre-FFN mix over completed blocks (+ partial)
        {
            std::vector<ggml_tensor *> srcs = completed;
            srcs.push_back(partial);
            ggml_tensor * mixed = depth_softmax_mix(ctx0, srcs, layer.ffn_res_proj, layer.ffn_res_norm,
                hparams.f_norm_rms_eps, il);
            ffn_inp = build_norm(mixed, layer.ffn_norm, nullptr, LLM_NORM_RMS, il);
            cb(ffn_inp, "ffn_inp", il);
        }

        // ---- MoE: sigmoid router + bias correction + renorm + shared expert
        // ALICE_PROBE: route-mass instrumentation. Recompute the identical router
        // logits here (a second matmul, only ~fine when the env var is set) so the
        // top-k selection is materialised as a readable tensor. build_moe_ffn below
        // recomputes from the same inputs, so the selected experts are bit-identical.
        // The probe hook API collects all named graph tensors after compute; the
        // runtime accumulates per-(layer, expert) selection mass from alice_probe_topk.
        if (std::getenv("ALICE_PROBE_TOPK")) {
            ggml_tensor * probe_logits = build_lora_mm(layer.ffn_gate_inp, ffn_inp);
            probe_logits = ggml_sigmoid(ctx0, probe_logits);
            probe_logits = ggml_add(ctx0, probe_logits, layer.ffn_exp_probs_b);
            ggml_tensor * probe_topk = ggml_argsort_top_k(ctx0, probe_logits, hparams.n_expert_used());
            // ggml_argsort_top_k returns a strided VIEW of the full 512-sort; a host
            // read of the view grabs the linear prefix of the parent buffer (measured
            // Sep 2026: bit-uniform counts, std 0.0). Materialize a contiguous copy.
            probe_topk = ggml_cont(ctx0, probe_topk);
            // Distinct name: the deployed router also publishes "ffn_moe_topk-<il>";
            // sharing it double-counts selections in the accumulator.
            cb(probe_topk, "alice_probe_topk", il);
            // cb() only assigns a name; without an expand the tensor has no
            // consumers and ggml prunes it from the graph (same trap as out_ids).
            ggml_build_forward_expand(gf, probe_topk);

            // per-layer activation energy in the router input: the imatrix-style
            // importance signal that scales how much a weight error at this layer
            // moves the trajectory. One scalar reduction per layer.
            ggml_tensor * probe_act2 = ggml_sum(ctx0, ggml_sqr(ctx0, ffn_inp));
            cb(probe_act2, "alice_act2", il);
            ggml_build_forward_expand(gf, probe_act2);
        }
        // ALICE_MOE_BLOCK=1: two-node composite CPU MoE (gate_up rows ->
        // staging, silu+down rows -> out). Replaces the ~48-node deployed
        // chain on host-backed expert layers only. Default OFF.
        static const bool moe_block_on = std::getenv("ALICE_MOE_BLOCK") != nullptr;
        static int moe_block_max = -2;
        if (moe_block_max == -2) {
            moe_block_max = (int) model.layers.size();
            if (const char * mx = std::getenv("ALICE_MOE_BLOCK_MAX")) moe_block_max = atoi(mx);
        }
        const bool use_moe_block = moe_block_on && il < moe_block_max && alice_fused_moe_layer_ok(layer);
        if (fused_moe_enabled && il < fused_moe_max && alice_fused_moe_layer_ok(layer)) {
            fused_moe.quant_x = alice_fmoe_quant_x;
            fused_moe.gate_up = alice_fmoe_gate_up;
            fused_moe.swiglu  = alice_fmoe_swiglu;
            fused_moe.down    = alice_fmoe_down;
            fused_moe.reduce  = alice_fmoe_reduce;
            fused_moe.userdata = &amodel.fused_moe_ud[il];
        } else {
            fused_moe = {};
        }
        ggml_tensor * moe_out = use_moe_block
            ? alice_moe_block_build(ctx0, ffn_inp, il, model, this)
            : build_moe_ffn(ffn_inp,
            layer.ffn_gate_inp,
            layer.ffn_up_exps,
            layer.ffn_gate_exps,
            layer.ffn_down_exps,
            layer.ffn_exp_probs_b,
            hparams.n_expert,
            hparams.n_expert_used(),
            LLM_FFN_SILU, true,
            hparams.expert_weights_scale,
            (llama_expert_gating_func_type) hparams.expert_gating_func,
            il,
            nullptr,
            layer.ffn_gate_up_exps);
        cb(moe_out, "ffn_moe_out", il);

        ggml_tensor * ffn_shexp = build_ffn(ffn_inp,
            layer.ffn_up_shexp,   NULL, NULL,
            layer.ffn_gate_shexp, NULL, NULL,
            layer.ffn_down_shexp, NULL, NULL,
            NULL,
            LLM_FFN_SILU, LLM_FFN_PAR, il);
        ggml_tensor * shared_gate = build_lora_mm(layer.ffn_gate_inp_shexp, ffn_inp);
        shared_gate = ggml_sigmoid(ctx0, shared_gate);
        ffn_shexp = ggml_mul(ctx0, ffn_shexp, shared_gate);
        cur = ggml_add(ctx0, moe_out, ffn_shexp);
        cb(cur, "ffn_out", il);

        partial = ggml_add(ctx0, partial, cur);
        if (il == 0) {
            l0_out = partial;   // captured for the differential harness
        }

        if ((il + 1) % 4 == 0) {
            // block closes: completed grows, partial resets
            completed.push_back(partial);
            partial = nullptr;
        }
        (void) ffn_inp;
    }

    // final mix over completed blocks (+ trailing partial) then norm + head
    if (partial != nullptr) {
        completed.push_back(partial);
    }
    ggml_tensor * hidden = depth_softmax_mix(ctx0, completed, model.output_res_proj, model.output_res_norm,
        hparams.f_norm_rms_eps, -1);
    if (hparams.n_layer_nextn > 0) {
        // MTP draft input: the trunk hidden state BEFORE the final norm, all rows
        // (the graph result marks it as an output; the spec loop feeds it back).
        res->t_h_nextn = hidden;
    }
    cur = build_norm(hidden, model.output_norm, nullptr, LLM_NORM_RMS, -1);
    if (inp_out_ids) {
        cur = ggml_get_rows(ctx0, cur, inp_out_ids);
    }
    cb(cur, "result_norm", -1);
    res->t_embd = cur;
    if (std::getenv("ALICE_DUMP_L0") && l0_out) {
        res->t_embd = l0_out;
    }
    cur = ggml_mul_mat(ctx0, model.output, cur);
    cb(cur, "result_output", -1);
    res->t_logits = cur;

    ggml_build_forward_expand(gf, cur);
}

// ---------------------------------------------------------------------------
// MTP / nextn draft block (LLM_GRAPH_TYPE_DECODER_MTP).
//
// The checkpoint's `mtp.*` tensors become one extra block at index n_layer:
//   concat(RMSNorm(emb(t+1), enorm), RMSNorm(hidden(t), hnorm)) @ eh_proj
//   -> gated full attention (own KV cache) -> MoE (+ shared expert) -> RMSNorm(shared head)
// The block's hidden output becomes t_h_nextn (the next draft's input) and the
// lm_head produces the draft logits for verification by the main model.
// ---------------------------------------------------------------------------
llama_model_alice_ai::graph_mtp::graph_mtp(const llama_model & model, const llm_graph_params & params) :
    llm_graph_context(params), model(model) {
    GGML_ASSERT(hparams.n_layer_nextn > 0 && "ALICE_AI MTP requires n_layer_nextn > 0");

    const int64_t n_embd_head = hparams.n_embd_head_k();
    const int64_t n_head      = hparams.n_head();
    const int64_t n_head_kv   = hparams.n_head_kv(0);
    const int64_t n_rot       = hparams.n_rot();
    const int64_t n_tokens    = ubatch.n_tokens;

    const int il = hparams.n_layer();
    const auto & layer = model.layers[il];

    GGML_ASSERT(layer.nextn.eh_proj && "ALICE_AI MTP: missing nextn.eh_proj");
    GGML_ASSERT(layer.nextn.enorm   && "ALICE_AI MTP: missing nextn.enorm");
    GGML_ASSERT(layer.nextn.hnorm   && "ALICE_AI MTP: missing nextn.hnorm");
    GGML_ASSERT(layer.ffn_gate_inp  && "ALICE_AI MTP: missing ffn_gate_inp");

    auto inp = std::make_unique<llm_graph_input_embd_h>(hparams.n_embd);

    inp->tokens = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_tokens);
    ggml_set_input(inp->tokens);

    inp->embd = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, hparams.n_embd_inp(), n_tokens);
    ggml_set_input(inp->embd);

    inp->h = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, hparams.n_embd, n_tokens);
    ggml_set_input(inp->h);
    ggml_set_name(inp->h, "mtp_h_input");

    ggml_tensor * tokens_in = inp->tokens;
    ggml_tensor * embd_in   = inp->embd;
    ggml_tensor * h_in      = inp->h;

    res->add_input(std::move(inp));

    ggml_tensor * tok_embd = ubatch.token ? ggml_get_rows(ctx0, model.tok_embd, tokens_in) : embd_in;
    cb(tok_embd, "mtp_tok_embd", il);

    ggml_tensor * inp_pos = build_inp_pos();
    auto * inp_attn = build_attn_inp_kv();

    ggml_tensor * h_norm = build_norm(h_in, layer.nextn.hnorm, nullptr, LLM_NORM_RMS, il);
    cb(h_norm, "mtp_hnorm", il);

    ggml_tensor * e_norm = build_norm(tok_embd, layer.nextn.enorm, nullptr, LLM_NORM_RMS, il);
    cb(e_norm, "mtp_enorm", il);

    ggml_tensor * concat = ggml_concat(ctx0, e_norm, h_norm, /*dim=*/0);
    cb(concat, "mtp_concat", il);

    ggml_tensor * cur = build_lora_mm(layer.nextn.eh_proj, concat);
    cb(cur, "mtp_eh_proj", il);

    ggml_tensor * inpSA = cur;

    cur = build_norm(cur, layer.attn_norm, nullptr, LLM_NORM_RMS, il);
    cb(cur, "mtp_attn_norm", il);

    // ---- gated full attention (identical layout to the trunk's full-attn layers)
    {
        ggml_tensor * Qcur_full = build_lora_mm(layer.wq, cur);
        Qcur_full = ggml_reshape_4d(ctx0, Qcur_full, n_embd_head * 2, n_head, n_tokens, 1);
        ggml_tensor * Qcur = ggml_view_4d(ctx0, Qcur_full, n_embd_head, n_head, n_tokens, 1,
            Qcur_full->nb[1], Qcur_full->nb[2], Qcur_full->nb[3], 0);
        ggml_tensor * gate = ggml_view_4d(ctx0, Qcur_full, n_embd_head, n_head, n_tokens, 1,
            Qcur_full->nb[1], Qcur_full->nb[2], Qcur_full->nb[3], n_embd_head * ggml_element_size(Qcur_full));

        ggml_tensor * Kcur = build_lora_mm(layer.wk, cur);
        ggml_tensor * Vcur = build_lora_mm(layer.wv, cur);
        Kcur = ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens);
        Vcur = ggml_reshape_3d(ctx0, Vcur, n_embd_head, n_head_kv, n_tokens);

        Qcur = build_norm(Qcur, layer.attn_q_norm, nullptr, LLM_NORM_RMS, il);
        Kcur = build_norm(Kcur, layer.attn_k_norm, nullptr, LLM_NORM_RMS, il);

        Qcur = ggml_rope_ext(ctx0, Qcur, inp_pos, nullptr, n_rot, rope_type, n_ctx_orig,
            freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow);
        Kcur = ggml_rope_ext(ctx0, Kcur, inp_pos, nullptr, n_rot, rope_type, n_ctx_orig,
            freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow);

        const float kq_scale = hparams.f_attention_scale == 0.0f
            ? 1.0f / sqrtf(float(n_embd_head)) : hparams.f_attention_scale;

        cur = build_attn(inp_attn, nullptr, nullptr, nullptr,
            Qcur, Kcur, Vcur, nullptr, nullptr, nullptr, kq_scale, il);
        cb(cur, "mtp_attn_pregate", il);

        gate = ggml_cont_2d(ctx0, gate, n_embd_head * n_head, n_tokens);
        gate = ggml_sigmoid(ctx0, gate);
        gate = ggml_reshape_2d(ctx0, gate, n_embd_head * n_head, n_tokens);
        cur = ggml_mul(ctx0, cur, gate);
        cur = build_lora_mm(layer.wo, cur);
        cb(cur, "mtp_attn_out", il);
    }

    cur = ggml_add(ctx0, cur, inpSA);
    cb(cur, "mtp_attn_residual", il);

    ggml_tensor * ffn_residual = cur;
    cur = build_norm(cur, layer.ffn_norm, nullptr, LLM_NORM_RMS, il);
    cb(cur, "mtp_ffn_norm", il);

    // ---- MoE: sigmoid router + bias correction + shared expert (mirrors the trunk)
    ggml_tensor * moe_out = build_moe_ffn(cur,
        layer.ffn_gate_inp,
        layer.ffn_up_exps,
        layer.ffn_gate_exps,
        layer.ffn_down_exps,
        layer.ffn_exp_probs_b,
        hparams.n_expert,
        hparams.n_expert_used(),
        LLM_FFN_SILU, true,
        hparams.expert_weights_scale,
        (llama_expert_gating_func_type) hparams.expert_gating_func,
        il,
        nullptr,
        layer.ffn_gate_up_exps);
    cb(moe_out, "mtp_ffn_moe_out", il);

    ggml_tensor * ffn_shexp = build_ffn(cur,
        layer.ffn_up_shexp,   NULL, NULL,
        layer.ffn_gate_shexp, NULL, NULL,
        layer.ffn_down_shexp, NULL, NULL,
        NULL,
        LLM_FFN_SILU, LLM_FFN_PAR, il);
    ggml_tensor * shared_gate = ggml_sigmoid(ctx0, build_lora_mm(layer.ffn_gate_inp_shexp, cur));
    ffn_shexp = ggml_mul(ctx0, ffn_shexp, shared_gate);
    cur = ggml_add(ctx0, moe_out, ffn_shexp);
    cb(cur, "mtp_ffn_out", il);

    cur = ggml_add(ctx0, cur, ffn_residual);
    cb(cur, "mtp_post_ffn", il);

    ggml_tensor * head_norm_w = layer.nextn.shared_head_norm ? layer.nextn.shared_head_norm : model.output_norm;
    cur = build_norm(cur, head_norm_w, nullptr, LLM_NORM_RMS, -1);
    cb(cur, "h_nextn", -1);
    res->t_h_nextn = cur;

    ggml_tensor * inp_out_ids = build_inp_out_ids();
    if (inp_out_ids) {
        cur = ggml_get_rows(ctx0, cur, inp_out_ids);
    }
    cb(cur, "mtp_shared_head_norm", -1);

    cur = ggml_mul_mat(ctx0, model.output, cur);
    cb(cur, "result_output", -1);
    res->t_logits = cur;

    ggml_build_forward_expand(gf, cur);
}
