#include "models.h"

#include "ggml-backend.h"
#include "ggml.h"

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
// exist; reads back ffn_moe_topk-<il> via the scheduler's host sync and adds
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
    // name is "ffn_moe_topk-<il>"
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
    const char * want = "ffn_moe_topk-";
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
}

std::unique_ptr<llm_graph_context> llama_model_alice_ai::build_arch_graph(const llm_graph_params & params) const {
    return std::make_unique<graph>(*this, params);
}

// Causal Conv1d for Q/K/V (same layout as kimi-linear)
static ggml_tensor * causal_conv1d(ggml_cgraph * gf, ggml_context * ctx0, ggml_tensor * conv_states_all, ggml_tensor * conv_state_all, int64_t qkv, ggml_tensor * x, ggml_tensor * proj_w, ggml_tensor * conv_w, int64_t d_conv, int64_t head_dim, int64_t n_head, int64_t n_seq_tokens, int64_t n_seqs, int64_t n_tokens, int64_t kv_head) {
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

            ggml_tensor * Qcur = causal_conv1d(gf, ctx0, conv_states_all, conv_state_all, 0, cur, layer.ssm_q, layer.ssm_q_conv, d_conv, head_dim, n_head_kd, n_seq_tokens, n_seqs, ubatch.n_tokens, kv_head);
            ggml_tensor * Kcur = causal_conv1d(gf, ctx0, conv_states_all, conv_state_all, 1, cur, layer.ssm_k, layer.ssm_k_conv, d_conv, head_dim, n_head_kd, n_seq_tokens, n_seqs, ubatch.n_tokens, kv_head);
            ggml_tensor * Vcur = causal_conv1d(gf, ctx0, conv_states_all, conv_state_all, 2, cur, layer.ssm_v, layer.ssm_v_conv, d_conv, head_dim, n_head_kd, n_seq_tokens, n_seqs, ubatch.n_tokens, kv_head);

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

            auto attn_out = build_delta_net(Qcur, Kcur, Vcur, g1, beta, state, il);
            ggml_tensor * output    = ggml_cont(ctx0, attn_out.first);
            ggml_tensor * new_state = attn_out.second;
            ggml_build_forward_expand(gf,
                ggml_cpy(ctx0, new_state,
                    ggml_view_1d(ctx0, ssm_states_all, hparams.n_embd_s() * n_seqs,
                        kv_head * hparams.n_embd_s() * ggml_element_size(ssm_states_all))));

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
        // runtime accumulates per-(layer, expert) selection mass from ffn_moe_topk.
        if (std::getenv("ALICE_PROBE_TOPK")) {
            ggml_tensor * probe_logits = build_lora_mm(layer.ffn_gate_inp, ffn_inp);
            probe_logits = ggml_sigmoid(ctx0, probe_logits);
            probe_logits = ggml_add(ctx0, probe_logits, layer.ffn_exp_probs_b);
            ggml_tensor * probe_topk = ggml_argsort_top_k(ctx0, probe_logits, hparams.n_expert_used());
            cb(probe_topk, "ffn_moe_topk", il);
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
        ggml_tensor * moe_out = build_moe_ffn(ffn_inp,
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
