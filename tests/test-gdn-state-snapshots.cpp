// Known-answer test for the per-token recurrent-state snapshots emitted by
// ggml_gated_delta_net (contract documented in ggml/include/ggml.h right above
// the declaration).
//
// The contract under test:
//   * the output packs the attention scores [S_v,H_v,n_tokens,n_seqs] followed
//     by K state snapshots, most-recent first: slot 0 == final state,
//     slot s == the state s tokens back;
//   * K == 1 keeps only the final state;
//   * when n_tokens < K only slots 0..n_tokens-1 are written (the remaining
//     slots belong to the caller and MUST NOT be touched).
//
// The observable requirement is equivalence with a serial scan: invoking the op
// T times with n_tokens == 1 (each call seeded with the previous call's final
// state) must produce exactly the states and scores that one T-token invocation
// with K >= T produces. Speculative rollback restores rows straight out of those
// slots (llama_memory_recurrent::seq_rm -> rs_idx -> plane index), so an error
// here silently corrupts a rejected draft row instead of aborting.
//
// The test is model-free: it drives the op on synthetic tensors through the CPU
// backend only, so it needs no server, no GGUF and no GPU.
//
// Run: ./bin/test-gdn-state-snapshots

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include "models/recurrent-snapshot.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

// Snapshot slots and scores are produced by the same sequence of vector ops in
// the serial and the fused invocation, so bit equality is the expected result.
// The tolerance is kept explicit (and tiny) only so a legitimately different
// vectorisation of the state update does not turn into a false failure.
static constexpr float TOL = 0.0f;

static int g_failures = 0;
static int g_checks   = 0;

static void fail(const std::string & msg) {
    g_failures++;
    printf("  FAIL: %s\n", msg.c_str());
}

static std::string fmt(const char * what, int64_t a, int64_t b, float got, float want) {
    char buf[256];
    snprintf(buf, sizeof(buf), "%s[%lld][%lld] got %.9g want %.9g (diff %.3g)",
             what, (long long) a, (long long) b, (double) got, (double) want, (double) (got - want));
    return buf;
}

//------------------------------------------------------------------------------
// op driver
//------------------------------------------------------------------------------

struct gdn_case {
    int64_t S_v = 32;  // head size (KDA state rows/cols)
    int64_t H   = 2;   // v heads == k heads
    int64_t kda = 1;   // g is [S_v,H,T,B] (KDA) when 1, scalar [1,H,T,B] when 0
};

struct gdn_tensors {
    // all laid out exactly as the op documents them
    std::vector<float> q, k, v, g, beta, s0;
};

struct gdn_result {
    std::vector<float> scores;  // [S_v][H][T][B]
    std::vector<float> states;  // [K][B][H][S_v*S_v]
    int64_t K = 0;
};

// Runs the op once. `K` is the snapshot slot count handed to the op. When
// `sentinel` is non-null the whole output tensor is pre-filled with it before
// compute, so unwritten slots can be detected afterwards.
static bool run_gdn(const gdn_case & c, int64_t K, int64_t T, int64_t B,
                    const gdn_tensors & in, gdn_result & out,
                    const std::vector<float> * sentinel = nullptr) {
    const int64_t S_v = c.S_v;
    const int64_t H   = c.H;

    struct ggml_init_params ip = {
        /*.mem_size   =*/ (size_t) 16 * 1024 * 1024,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    struct ggml_context * ctx = ggml_init(ip);

    const int64_t g0 = c.kda ? S_v : 1;
    struct ggml_tensor * q  = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, S_v, H, T, B);
    struct ggml_tensor * k  = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, S_v, H, T, B);
    struct ggml_tensor * v  = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, S_v, H, T, B);
    struct ggml_tensor * g  = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, g0,  H, T, B);
    struct ggml_tensor * bb = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 1,   H, T, B);
    struct ggml_tensor * s  = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, S_v, S_v, H, B);

    struct ggml_tensor * res = ggml_gated_delta_net(ctx, q, k, v, g, bb, s, K);

    struct ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, res);

    ggml_backend_t backend = ggml_backend_cpu_init();
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (buf == nullptr) {
        fail("backend buffer allocation failed");
        ggml_free(ctx);
        ggml_backend_free(backend);
        return false;
    }

    ggml_backend_tensor_set(q,  in.q.data(),    0, in.q.size()    * sizeof(float));
    ggml_backend_tensor_set(k,  in.k.data(),    0, in.k.size()    * sizeof(float));
    ggml_backend_tensor_set(v,  in.v.data(),    0, in.v.size()    * sizeof(float));
    ggml_backend_tensor_set(g,  in.g.data(),    0, in.g.size()    * sizeof(float));
    ggml_backend_tensor_set(bb, in.beta.data(), 0, in.beta.size() * sizeof(float));
    ggml_backend_tensor_set(s,  in.s0.data(),   0, in.s0.size()   * sizeof(float));
    if (sentinel != nullptr) {
        ggml_backend_tensor_set(res, sentinel->data(), 0, ggml_nbytes(res));
    }

    const int rc = ggml_backend_graph_compute(backend, gf);
    if (rc != GGML_STATUS_SUCCESS) {
        fail("graph compute failed");
        ggml_backend_buffer_free(buf);
        ggml_backend_free(backend);
        ggml_free(ctx);
        return false;
    }

    std::vector<float> raw(ggml_nelements(res));
    ggml_backend_tensor_get(res, raw.data(), 0, ggml_nbytes(res));

    const size_t score_elems = (size_t) S_v * H * T * B;

    out.K = K;
    out.scores.assign(raw.begin(), raw.begin() + score_elems);
    out.states.assign(raw.begin() + score_elems, raw.end());

    ggml_backend_buffer_free(buf);
    ggml_backend_free(backend);
    ggml_free(ctx);
    return true;
}

//------------------------------------------------------------------------------
// reference serial scan
//------------------------------------------------------------------------------

struct serial_scan {
    std::vector<std::vector<float>> states;   // state after each token
    std::vector<std::vector<float>> scores;   // scores for each token
};

// Per-sequence, token-range slice with B == 1 (what a serial decode step sees).
static gdn_tensors slice_seq_tokens(const gdn_case & c, const gdn_tensors & in,
                                    int64_t T, int64_t B, int64_t b, int64_t t0, int64_t T_out) {
    const int64_t S_v = c.S_v;
    const int64_t H   = c.H;
    const int64_t g0  = c.kda ? S_v : 1;

    gdn_tensors o;
    o.q.resize(S_v * H * T_out);
    o.k.resize(S_v * H * T_out);
    o.v.resize(S_v * H * T_out);
    o.g.resize(g0 * H * T_out);
    o.beta.resize(H * T_out);
    o.s0.resize(S_v * S_v * H);

    for (int64_t t = 0; t < T_out; t++) {
        for (int64_t h = 0; h < H; h++) {
            const size_t src_tok = (size_t) ((b * T) + (t0 + t)) * H + h;
            const size_t dst_tok = (size_t) t * H + h;
            for (int64_t i = 0; i < S_v; i++) {
                o.q[dst_tok * S_v + i] = in.q[src_tok * S_v + i];
                o.k[dst_tok * S_v + i] = in.k[src_tok * S_v + i];
                o.v[dst_tok * S_v + i] = in.v[src_tok * S_v + i];
                if (c.kda) {
                    o.g[dst_tok * S_v + i] = in.g[src_tok * S_v + i];
                }
            }
            if (!c.kda) {
                o.g[dst_tok] = in.g[src_tok];
            }
            o.beta[dst_tok] = in.beta[src_tok];
        }
    }
    (void) B;
    return o;
}

// Runs the op once per token for tokens [t_begin, t_end) of sequence b,
// chaining the final state, exactly like the eager decode path does.
static bool serial_scan_run(const gdn_case & c, const gdn_tensors & in, int64_t T, int64_t B,
                            int64_t b, const std::vector<float> & s_start,
                            int64_t t_begin, int64_t t_end, serial_scan & sc) {
    std::vector<float> s_cur = s_start;
    sc.states.clear();
    sc.scores.clear();

    for (int64_t t = t_begin; t < t_end; t++) {
        gdn_tensors one = slice_seq_tokens(c, in, T, B, b, t, 1);
        one.s0 = s_cur;
        gdn_result r;
        if (!run_gdn(c, 1, 1, 1, one, r)) {
            return false;
        }
        sc.scores.push_back(r.scores);
        sc.states.push_back(r.states);
        s_cur = r.states;
    }
    return true;
}

//------------------------------------------------------------------------------
// data generation
//------------------------------------------------------------------------------

static void fill_uniform(std::vector<float> & dst, float lo, float hi, std::mt19937 & rng) {
    std::uniform_real_distribution<float> d(lo, hi);
    for (auto & x : dst) {
        x = d(rng);
    }
}

static gdn_tensors make_inputs(const gdn_case & c, int64_t T, int64_t B, uint32_t seed) {
    const int64_t S_v = c.S_v;
    const int64_t H   = c.H;
    const int64_t g0  = c.kda ? S_v : 1;

    std::mt19937 rng(seed);
    gdn_tensors in;
    in.q.resize(S_v * H * T * B);
    in.k.resize(S_v * H * T * B);
    in.v.resize(S_v * H * T * B);
    in.g.resize(g0 * H * T * B);
    in.beta.resize(H * T * B);
    in.s0.resize(S_v * S_v * H * B);

    fill_uniform(in.q,    -1.0f, 1.0f, rng);
    fill_uniform(in.k,    -1.0f, 1.0f, rng);
    fill_uniform(in.v,    -0.5f, 0.5f, rng);
    // gate: negative, like -exp(A_log)*softplus(..) in the real model
    fill_uniform(in.g,    -2.0f, -0.05f, rng);
    fill_uniform(in.beta,  0.0f, 1.0f, rng);
    fill_uniform(in.s0,   -0.1f, 0.1f, rng);
    return in;
}

//------------------------------------------------------------------------------
// checks
//------------------------------------------------------------------------------

static void cmp_vec(const char * what, int64_t idx, const std::vector<float> & got,
                    const std::vector<float> & want, float tol, const std::string & tag) {
    g_checks++;
    if (got.size() != want.size()) {
        fail(tag + ": size mismatch");
        return;
    }
    float  worst   = 0.0f;
    size_t worst_i = 0;
    for (size_t i = 0; i < got.size(); i++) {
        const float d = std::fabs(got[i] - want[i]);
        if (d > worst) { worst = d; worst_i = i; }
    }
    if (!(worst <= tol)) {
        fail(tag + ": " + fmt(what, idx, (int64_t) worst_i, got[worst_i], want[worst_i]) +
             " max|diff|=" + std::to_string((double) worst));
    }
}

// 1. fused T-token run vs the serial reference, for every snapshot slot and score
static void test_fused_vs_serial(const gdn_case & c, int64_t T, int64_t B, int64_t K, uint32_t seed) {
    const int64_t S_v = c.S_v;
    const int64_t H   = c.H;

    char tag[192];
    snprintf(tag, sizeof(tag), "fused-vs-serial S_v=%lld H=%lld T=%lld B=%lld K=%lld kda=%lld seed=%u",
             (long long) S_v, (long long) H, (long long) T, (long long) B, (long long) K,
             (long long) c.kda, seed);
    const std::string tg = tag;

    gdn_tensors in = make_inputs(c, T, B, seed);

    gdn_result fused;
    if (!run_gdn(c, K, T, B, in, fused)) {
        return;
    }

    const int64_t n_written = std::min<int64_t>(T, K);

    for (int64_t b = 0; b < B; b++) {
        serial_scan sc;
        std::vector<float> s0(in.s0.begin() + (size_t) b * S_v * S_v * H,
                              in.s0.begin() + (size_t) (b + 1) * S_v * S_v * H);
        if (!serial_scan_run(c, in, T, B, b, s0, 0, T, sc)) {
            return;
        }

        // attention scores
        for (int64_t t = 0; t < T; t++) {
            std::vector<float> got(S_v * H);
            for (int64_t h = 0; h < H; h++) {
                for (int64_t i = 0; i < S_v; i++) {
                    got[h * S_v + i] = fused.scores[((size_t) ((b * T) + t) * H + h) * S_v + i];
                }
            }
            cmp_vec("scores", t, got, sc.scores[t], TOL, tg + " seq " + std::to_string(b));
        }

        // snapshots: slot s == state s tokens back == state after token T-1-s
        for (int64_t s = 0; s < n_written; s++) {
            const int64_t t = T - 1 - s;
            std::vector<float> got(fused.states.begin() + ((size_t) s * B + b) * S_v * S_v * H,
                                   fused.states.begin() + ((size_t) s * B + b + 1) * S_v * S_v * H);
            cmp_vec("slot", s, got, sc.states[t], TOL,
                    tg + " seq " + std::to_string(b) + " slot " + std::to_string(s) +
                    " (must be state after token " + std::to_string(t) + ")");
        }
    }
    printf("  ok  %s\n", tag);
}

// 2. rollback: restore the snapshot for the accepted prefix and re-scan the suffix
static void test_rollback(const gdn_case & c, int64_t T, int64_t K, int64_t B, uint32_t seed) {
    const int64_t S_v = c.S_v;
    const int64_t H   = c.H;

    if (K < 2 || K > T) {
        return; // needs at least one snapshot slot beyond the final state
    }

    char tag[192];
    snprintf(tag, sizeof(tag), "rollback S_v=%lld T=%lld K=%lld B=%lld kda=%lld seed=%u",
             (long long) S_v, (long long) T, (long long) K, (long long) B, (long long) c.kda, seed);
    const std::string tg = tag;

    gdn_tensors in = make_inputs(c, T, B, seed);

    gdn_result fused;
    if (!run_gdn(c, K, T, B, in, fused)) {
        return;
    }

    for (int64_t b = 0; b < B; b++) {
        serial_scan ref;
        std::vector<float> s0(in.s0.begin() + (size_t) b * S_v * S_v * H,
                              in.s0.begin() + (size_t) (b + 1) * S_v * S_v * H);
        if (!serial_scan_run(c, in, T, B, b, s0, 0, T, ref)) {
            return;
        }

        // p = number of accepted tokens; the token at position p is rejected
        for (int64_t p = 1; p < T; p++) {
            const int64_t rollback = T - p;   // tokens to undo
            if (rollback < 1 || rollback > K - 1) {
                continue;
            }
            const int64_t slot = rollback;    // slot s == s tokens back

            std::vector<float> restored(fused.states.begin() + ((size_t) slot * B + b) * S_v * S_v * H,
                                        fused.states.begin() + ((size_t) (slot * B + b + 1)) * S_v * S_v * H);

            cmp_vec("restored", slot, restored, ref.states[p - 1], TOL,
                    tg + " seq " + std::to_string(b) + " accept=" + std::to_string(p));

            serial_scan cont;
            if (!serial_scan_run(c, in, T, B, b, restored, p, T, cont)) {
                return;
            }
            for (int64_t t = 0; t < T - p; t++) {
                cmp_vec("re-scan state", t, cont.states[t], ref.states[p + t], TOL,
                        tg + " seq " + std::to_string(b) + " accept=" + std::to_string(p));
                cmp_vec("re-scan scores", t, cont.scores[t], ref.scores[p + t], TOL,
                        tg + " seq " + std::to_string(b) + " accept=" + std::to_string(p));
            }
        }
    }
    printf("  ok  %s\n", tag);
}

// 3. contract guard: slots >= min(n_tokens, K) must be left untouched
static void test_unwritten_slots(const gdn_case & c, int64_t T, int64_t K, uint32_t seed) {
    const int64_t S_v = c.S_v;
    const int64_t H   = c.H;
    const int64_t B   = 1;

    const int64_t n_written = std::min<int64_t>(T, K);
    if (n_written >= K) {
        return; // every slot is written; nothing to guard
    }

    const float SENTINEL = 1234.5f;

    gdn_tensors in = make_inputs(c, T, B, seed);

    const size_t total = (size_t) S_v * H * T * B + (size_t) K * S_v * S_v * H * B;
    const std::vector<float> sentinel(total, SENTINEL);

    gdn_result fused;
    if (!run_gdn(c, K, T, B, in, fused, &sentinel)) {
        return;
    }

    char tag[192];
    snprintf(tag, sizeof(tag), "unwritten-slots S_v=%lld T=%lld K=%lld seed=%u",
             (long long) S_v, (long long) T, (long long) K, seed);

    for (int64_t s = n_written; s < K; s++) {
        const std::vector<float> got(fused.states.begin() + (size_t) s * S_v * S_v * H * B,
                                     fused.states.begin() + (size_t) (s + 1) * S_v * S_v * H * B);
        g_checks++;
        for (size_t i = 0; i < got.size(); i++) {
            if (got[i] != SENTINEL) {
                fail(std::string(tag) + ": slot " + std::to_string(s) + " was written at elem " +
                     std::to_string(i) + " (contract: only slots 0.." + std::to_string(n_written - 1) +
                     " may be written)");
                break;
            }
        }
    }
    printf("  ok  %s\n", tag);
}

//------------------------------------------------------------------------------
// 4. conv-state snapshot planes
//
// alice_ai.cpp packs the Q|K|V conv states into one recurrent-cache cell and
// writes (1 + n_rs_seq) planes per cell (delta-net-base.cpp build_conv_state()
// does the same for the other KDA archs). The planes are read back by
// llama_memory_recurrent::s_copy(), which indexes a plane by the ROLLBACK DEPTH:
// plane r is the state r tokens back. So conv plane r has to be the conv window
// ending r tokens before the last token of the ubatch -- the same "s tokens back"
// convention section 1/2 prove for the S-state planes.
//
// The offset arithmetic lives in models/recurrent-snapshot.h and is shared with
// the model code, so this section fails whenever the model writes a conv plane
// from the wrong step.

static void test_conv_snapshot_planes(int64_t d_conv, int64_t n_tokens, int64_t K, int64_t n_seqs, int64_t qkv) {
    const int64_t win     = d_conv - 1;
    const int64_t C       = 4;              // channel stand-in for d_inner
    const int64_t row_w   = 3 * win * C;    // Q|K|V conv states packed in one cell
    const int64_t mem_size = n_seqs;        // one cell per sequence
    const int64_t n_written = std::min<int64_t>(n_tokens, K);
    const float   SENTINEL  = 1234.5f;

    char tag[192];
    snprintf(tag, sizeof(tag), "conv-planes d_conv=%lld T=%lld K=%lld B=%lld qkv=%lld",
             (long long) d_conv, (long long) n_tokens, (long long) K, (long long) n_seqs, (long long) qkv);

    std::mt19937 rng((uint32_t) (0xC0FFEE ^ (d_conv * 131 + n_tokens * 17 + K * 7 + n_seqs * 3 + qkv)));
    // ggml layout: element (a, ch, s) of a [A, C, B] tensor sits at
    // a + ch*A + s*A*C  (that is what the view/cpy below actually moves)
    std::vector<float> hist(win * C * n_seqs);
    std::vector<float> tok(n_tokens * C * n_seqs);
    {
        std::uniform_real_distribution<float> d(-1.0f, 1.0f);
        for (int64_t s = 0; s < n_seqs; s++) {
            for (int64_t ch = 0; ch < C; ch++) {
                for (int64_t a = 0; a < win; a++) {
                    hist[(size_t) (a + ch * win + s * win * C)] = d(rng);
                }
                for (int64_t a = 0; a < n_tokens; a++) {
                    tok[(size_t) (a + ch * n_tokens + s * n_tokens * C)] = d(rng);
                }
            }
        }
    }

    struct ggml_init_params ip = {
        /*.mem_size   =*/ (size_t) 16 * 1024 * 1024,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    struct ggml_context * ctx = ggml_init(ip);

    struct ggml_tensor * hist_t = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, win, C, n_seqs);
    struct ggml_tensor * tok_t  = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, n_tokens, C, n_seqs);
    struct ggml_tensor * conv_x = ggml_concat(ctx, hist_t, tok_t, 0);   // [win + n_tokens, C, n_seqs]

    // the recurrent cache: mem_size cells per plane, (1 + K) planes
    struct ggml_tensor * cache = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, row_w, mem_size * (1 + K));

    struct ggml_cgraph * gf = ggml_new_graph(ctx);
    for (int64_t slot = 0; slot < n_written; ++slot) {
        const int64_t off = llm_conv_snapshot_offset(n_tokens, slot);
        struct ggml_tensor * snap = ggml_view_3d(ctx, conv_x, win, C, n_seqs,
                conv_x->nb[1], conv_x->nb[2], (size_t) off * conv_x->nb[0]);
        struct ggml_tensor * plane = ggml_view_3d(ctx, cache, win, C, n_seqs,
                (size_t) win * sizeof(float),
                (size_t) row_w * sizeof(float),
                (size_t) (((slot * mem_size) * row_w) + qkv * win * C) * sizeof(float));
        // cell `head` of the plane sits at row (slot*mem_size + head); the seq dim
        // of the view strides by exactly one row, like the real cache view
        ggml_build_forward_expand(gf, ggml_cpy(ctx, snap, plane));
    }

    ggml_backend_t backend = ggml_backend_cpu_init();
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (buf == nullptr) {
        fail(std::string(tag) + ": backend buffer allocation failed");
        ggml_free(ctx);
        ggml_backend_free(backend);
        return;
    }

    ggml_backend_tensor_set(hist_t, hist.data(), 0, hist.size() * sizeof(float));
    ggml_backend_tensor_set(tok_t,  tok.data(),  0, tok.size()  * sizeof(float));
    {
        std::vector<float> sent(mem_size * (1 + K) * row_w, SENTINEL);
        ggml_backend_tensor_set(cache, sent.data(), 0, ggml_nbytes(cache));
    }

    if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) {
        fail(std::string(tag) + ": graph compute failed");
        ggml_backend_buffer_free(buf);
        ggml_backend_free(backend);
        ggml_free(ctx);
        return;
    }

    std::vector<float> got((size_t) mem_size * (1 + K) * row_w);
    ggml_backend_tensor_get(cache, got.data(), 0, ggml_nbytes(cache));

    // expected window: after j = n_tokens - slot tokens the conv state is the
    // window of the last `win` inputs, i.e. absolute conv_x indices [j, j+win-1]
    for (int64_t slot = 0; slot < n_written; ++slot) {
        const int64_t j = n_tokens - slot;
        for (int64_t head = 0; head < n_seqs; head++) {
            const float * row = got.data() + ((size_t) (slot * mem_size + head) * row_w) + qkv * win * C;
            g_checks++;
            for (int64_t ch = 0; ch < C; ch++) {
                for (int64_t r = 0; r < win; r++) {
                    const int64_t a = j + r;   // absolute input index in conv_x
                    const float want = a < win
                        ? hist[(size_t) (a + ch * win + head * win * C)]
                        : tok[(size_t) ((a - win) + ch * n_tokens + head * n_tokens * C)];
                    // cache slab layout is column-major within the (win, C) block
                    const float have = row[r + ch * win];
                    if (have != want) {
                        fail(std::string(tag) + ": plane " + std::to_string(slot) + " (must be " +
                             std::to_string(slot) + " tokens back = state after token " + std::to_string(j) +
                             ") cell " + std::to_string(head) + " ch " + std::to_string(ch) +
                             " r " + std::to_string(r) + ": got " + std::to_string((double) have) +
                             " want " + std::to_string((double) want));
                        break;
                    }
                }
            }
        }
    }

    // contract: planes >= min(n_tokens, K) are caller-owned
    for (int64_t slot = n_written; slot < K; slot++) {
        const float * row = got.data() + (size_t) (slot * mem_size) * row_w;
        g_checks++;
        for (int64_t i = 0; i < mem_size * row_w; i++) {
            if (row[i] != SENTINEL) {
                fail(std::string(tag) + ": plane " + std::to_string(slot) + " was written at elem " +
                     std::to_string(i) + " (only planes 0.." + std::to_string(n_written - 1) +
                     " may be written)");
                break;
            }
        }
    }

    ggml_backend_buffer_free(buf);
    ggml_backend_free(backend);
    ggml_free(ctx);

    if (g_failures == 0) {
        printf("  ok  %s\n", tag);
    }
}

//------------------------------------------------------------------------------

int main() {
    printf("test-gdn-state-snapshots: ggml_gated_delta_net per-token state snapshots\n");
    printf("tolerance: %g (exact; fused and serial run the same vector ops in the same order)\n\n",
           (double) TOL);

    struct shape {
        int64_t S_v, H, kda;
    };
    const shape shapes[] = {
        { 32, 2, 1 },   // KDA gate (g is [S_v,H,T,B])
        { 32, 2, 0 },   // scalar gate (g is [1,H,T,B])
        { 16, 1, 1 },   // single head
        { 64, 4, 1 },
    };

    printf("== 1. fused N-token run vs serial decode (scores + per-token states)\n");
    for (const auto & sh : shapes) {
        gdn_case c = { sh.S_v, sh.H, sh.kda };
        // T == K (all slots live), K == 1 (final only), n_tokens > K, n_tokens < K
        test_fused_vs_serial(c, 4, 1, 4, 1);
        test_fused_vs_serial(c, 4, 1, 1, 2);
        test_fused_vs_serial(c, 4, 1, 8, 3);   // n_tokens < K
        test_fused_vs_serial(c, 8, 1, 4, 4);   // n_tokens > K
        test_fused_vs_serial(c, 3, 2, 3, 5);   // two sequences
        test_fused_vs_serial(c, 5, 3, 5, 6);
    }

    printf("\n== 2. rollback: restore snapshot for the accepted prefix, re-scan the suffix\n");
    for (const auto & sh : shapes) {
        gdn_case c = { sh.S_v, sh.H, sh.kda };
        test_rollback(c, 4, 4, 1, 7);
        test_rollback(c, 4, 2, 1, 8);
        test_rollback(c, 6, 3, 2, 9);
        test_rollback(c, 8, 4, 2, 10);
    }

    printf("\n== 3. contract guard: slots >= min(n_tokens,K) stay caller-owned\n");
    for (const auto & sh : shapes) {
        gdn_case c = { sh.S_v, sh.H, sh.kda };
        test_unwritten_slots(c, 1, 4, 11);
        test_unwritten_slots(c, 2, 5, 12);
        test_unwritten_slots(c, 3, 3, 13);
    }

    printf("\n== 4. conv-state planes: plane r must be the conv window r tokens back\n");
    for (int64_t qkv = 0; qkv < 3; qkv++) {
        test_conv_snapshot_planes(4, 4, 4, 1, qkv);
        test_conv_snapshot_planes(4, 4, 2, 1, qkv);
        test_conv_snapshot_planes(4, 1, 4, 1, qkv);
        test_conv_snapshot_planes(4, 4, 4, 2, qkv);
        test_conv_snapshot_planes(4, 3, 3, 3, qkv);
        test_conv_snapshot_planes(4, 2, 5, 2, qkv);
    }

    printf("\n%d checks, %d failures\n", g_checks, g_failures);
    return g_failures == 0 ? 0 : 1;
}
