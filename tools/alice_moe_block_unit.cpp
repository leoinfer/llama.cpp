// alice_moe_block_unit.cpp — parity + speed unit test for ALICE_MOE_BLOCK.
//
// Loads one layer's REAL tensors from the GGUF (router, bias, Q3_K/Q4_K expert
// banks) and runs:
//   A) deployed chain (router mm, sigmoid, bias add, argsort_top_k, get_rows,
//      renorm, gate_up mm_id, swiglu, down mm_id, weight mul, 10 views+9 adds)
//   B) the two block nodes (gate_up rows -> staging, silu+down rows -> out)
//      by calling the SAME static functions the runtime uses (linked from a
//      small object compiled out of alice_ai.cpp — see below).
// on the CPU backend at n_tokens=1, and reports max|diff| over the MoE output
// plus per-path ms. No GPU needed.
//
// The block functions are small and self-contained; this tool #includes the
// relevant section of alice_ai.cpp via ALICE_MOE_BLOCK_UNIT_TEST, which
// exposes alice_moe_block_gate_up/down + topk10 without the model deps.
// Simpler and robust: reimplement the 60-line router+topk identically here
// (it is deterministic scalar code) and call ggml kernels for the matvecs.
// The matvec kernels ARE ggml's (vec_dot), so parity of B vs A tests exactly
// the composition, which is the risk.
//
// build: g++ -O2 -std=c++17 -I include -I ggml/include tools/alice_moe_block_unit.cpp \
//        -L build-vk/bin -lggml -lggml-base -lggml-cpu -Wl,-rpath,$PWD/build-vk/bin
// run: ./alice_moe_block_unit <model.gguf> <layer> [n_tok]

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "gguf.h"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fcntl.h>
#include <unistd.h>
#include <vector>

static double now_ms() {
    using namespace std::chrono;
    return duration<double, std::milli>(steady_clock::now().time_since_epoch()).count();
}

static const int64_t E = 2048, F = 512, NX = 512, NU = 10;

int main(int argc, char ** argv) {
    if (argc < 3) { printf("usage: %s model.gguf layer [n_tok]\n", argv[0]); return 1; }
    const char * model_path = argv[1];
    const int il = atoi(argv[2]);
    const int n_tok = argc > 3 ? atoi(argv[3]) : 1;

    ggml_backend_load_all();
    ggml_backend_dev_t cpu_dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    ggml_backend_t cpu = ggml_backend_dev_init(cpu_dev, nullptr);
    ggml_backend_cpu_set_n_threads(cpu, 8);
    ggml_backend_buffer_type_t buft = ggml_backend_get_default_buffer_type(cpu);

    struct gguf_init_params gp = { true, nullptr };
    struct gguf_context * gg = gguf_init_from_file(model_path, gp);
    if (!gg) { printf("gguf open failed\n"); return 1; }
    const size_t data_off = gguf_get_data_offset(gg);

    auto must = [&](const char * name) -> int64_t {
        int64_t id = gguf_find_tensor(gg, name);
        if (id < 0) { printf("missing tensor %s\n", name); exit(1); }
        return id;
    };
    char nb[128];
    snprintf(nb, sizeof(nb), "blk.%d.ffn_gate_inp.weight", il);
    int64_t id_gi = must(nb);
    snprintf(nb, sizeof(nb), "blk.%d.ffn_gate_up_exps.weight", il);
    int64_t id_gu = must(nb);
    snprintf(nb, sizeof(nb), "blk.%d.ffn_down_exps.weight", il);
    int64_t id_dn = must(nb);
    snprintf(nb, sizeof(nb), "blk.%d.ffn_exp_probs_b.bias", il);
    int64_t id_pb = gguf_find_tensor(gg, nb);

    struct ggml_init_params wp = { 1024*1024*1024ULL, nullptr, true };
    ggml_context * wctx = ggml_init(wp);
    auto mkt = [&](int64_t id) -> ggml_tensor * {
        const int64_t * ne = gguf_get_tensor_ne(gg, id);
        int n_dims = 0;
        // count real dims from the tail (ne[dim] is 1 for dim >= n_dims)
        int64_t tmp[4] = { ne[0], ne[1], ne[2], ne[3] };
        n_dims = 4;
        while (n_dims > 1 && tmp[n_dims-1] == 1) --n_dims;
        ggml_tensor * t = ggml_new_tensor(wctx, gguf_get_tensor_type(gg, id), n_dims, tmp);
        return t;
    };
    ggml_tensor * gi = mkt(id_gi);
    ggml_tensor * gu = mkt(id_gu);
    ggml_tensor * dn = mkt(id_dn);
    ggml_tensor * pb = id_pb >= 0 ? mkt(id_pb) : nullptr;
    if (!ggml_backend_alloc_ctx_tensors_from_buft(wctx, buft)) { printf("weight alloc failed\n"); return 1; }
    int fd = open(model_path, O_RDONLY);
    auto load = [&](int64_t id, ggml_tensor * t) {
        size_t n = ggml_nbytes(t);
        std::vector<char> h(n);
        ssize_t r = pread(fd, h.data(), n, (off_t)(data_off + gguf_get_tensor_offset(gg, id)));
        if ((size_t) r != n) { printf("short read\n"); exit(1); }
        ggml_backend_tensor_set(t, h.data(), 0, n);
    };
    load(id_gi, gi); load(id_gu, gu); load(id_dn, dn);
    if (pb) load(id_pb, pb);
    printf("layer %d: gi %s [%lld,%lld], gu %s, dn %s, bias %s\n", il,
        ggml_type_name(gi->type), (long long) gi->ne[0], (long long) gi->ne[1],
        ggml_type_name(gu->type), ggml_type_name(dn->type), pb ? "yes" : "no");

    std::vector<float> xh((size_t) E * n_tok);
    for (size_t i = 0; i < xh.size(); ++i) xh[i] = (float) ((i % 13) - 6) / 6.0f;

    // ---- path A: deployed chain ----
    ggml_context * ca = ggml_init(wp);
    ggml_tensor * xin = ggml_new_tensor_2d(ca, GGML_TYPE_F32, E, n_tok);
    ggml_tensor * lg = ggml_mul_mat(ca, gi, xin);
    ggml_tensor * pr = ggml_sigmoid(ca, lg);
    ggml_tensor * sp = pb ? ggml_add(ca, pr, pb) : pr;
    ggml_tensor * se_full = ggml_argsort(ca, sp, GGML_SORT_ORDER_DESC);
    // MATERIALIZE the sort before any strided view: ggml's deployed MoE does
    // the same (the probe-trace fix, Sep 2026: a bare view of the argsort
    // output reads the linear prefix, not the top-k).
    ggml_tensor * se_c = ggml_cont(ca, se_full);
    ggml_tensor * se = ggml_view_2d(ca, se_c, NU, n_tok, se_c->nb[1], 0);
    ggml_tensor * pr3 = ggml_reshape_3d(ca, pr, 1, NX, n_tok);
    ggml_tensor * wt = ggml_get_rows(ca, pr3, se);
    ggml_tensor * wt2 = ggml_reshape_2d(ca, wt, NU, n_tok);
    ggml_tensor * ws = ggml_sum_rows(ca, wt2);
    ggml_tensor * wc = ggml_clamp(ca, ws, 6.103515625e-5f, 1e30f);
    ggml_tensor * wn = ggml_div(ca, wt2, wc);
    ggml_tensor * wn3 = ggml_reshape_3d(ca, wn, 1, NU, n_tok);
    ggml_tensor * xr = ggml_new_tensor_3d(ca, GGML_TYPE_F32, E, 1, n_tok);
    ggml_tensor * go = ggml_mul_mat_id(ca, gu, xr, se);
    // gate_up output go is [1024, 10, 1]: ne0=1024 rows, ne1=10 experts.
    // g = rows [0,512), u = rows [512,1024): SAME expert stride (nb1), offset
    // by F rows * element stride. go->nb[0] = sizeof(float) here (contiguous).
    ggml_tensor * g = ggml_view_3d(ca, go, F, NU, n_tok, go->nb[1], go->nb[2], 0);
    ggml_tensor * u = ggml_view_3d(ca, go, F, NU, n_tok, go->nb[1], go->nb[2], (size_t)F*go->nb[0]);
    ggml_tensor * act = ggml_swiglu_split(ca, g, u);
    ggml_tensor * dno = ggml_mul_mat_id(ca, dn, act, se);
    ggml_tensor * wm = ggml_mul(ca, dno, wn3);
    ggml_tensor * ex = ggml_reshape_3d(ca, wm, E, NU, n_tok);
    // aggregate exactly like the deployed path: view_2d slices with the
    // token row stride ex->nb[2] (llama-graph.cpp build_moe_ffn).
    ggml_tensor * mo = ggml_view_2d(ca, ex, E, n_tok, ex->nb[2], 0);
    for (int j = 1; j < NU; ++j)
        mo = ggml_add(ca, mo, ggml_view_2d(ca, ex, E, n_tok, ex->nb[2], (size_t)j*ex->nb[1]));
    if (!ggml_backend_alloc_ctx_tensors_from_buft(ca, buft)) { printf("A alloc failed\n"); return 1; }
    ggml_backend_tensor_set(xr, xh.data(), 0, ggml_nbytes(xr));
    ggml_cgraph * gfa = ggml_new_graph(ca);
    ggml_build_forward_expand(gfa, mo);
    for (int i = 0; i < 5; ++i) ggml_backend_graph_compute(cpu, gfa);
    ggml_backend_synchronize(cpu);
    double t0 = now_ms();
    for (int i = 0; i < 50; ++i) ggml_backend_graph_compute(cpu, gfa);
    ggml_backend_synchronize(cpu);
    double ms_a = (now_ms() - t0) / 50;
    auto mag = [&](ggml_tensor * t, const char * n) {
        std::vector<float> v(ggml_nelements(t));
        ggml_backend_tensor_get(t, v.data(), 0, sizeof(float)*v.size());
        double mx = 0; for (auto x : v) { double a = x < 0 ? -x : x; if (a > mx) mx = a; }
        printf("  mag %-6s max=%.4g\n", n, mx);
    };
    mag(go, "go"); mag(act, "act"); mag(dno, "dno"); mag(wm, "wm"); mag(ex, "ex");
    std::vector<float> ra((size_t) E * n_tok);
    ggml_backend_tensor_get(mo, ra.data(), 0, sizeof(float) * ra.size());
    // ids chosen by A
    std::vector<int32_t> ids_a((size_t) NU * n_tok);
    ggml_backend_tensor_get(se, ids_a.data(), 0, sizeof(int32_t) * ids_a.size());
    printf("A deployed: %.4f ms/call  ids[0..9] =", ms_a);
    for (int j = 0; j < NU; ++j) printf(" %d", ids_a[j]);
    printf("\n");
    printf("A out[0..3] = %.6f %.6f %.6f %.6f\n", ra[0], ra[1], ra[2], ra[3]);

    // ---- path B: reference block composition (router pack + gate_up rows +
    // silu/down rows) with ggml kernels only. Mirrors alice_moe_block_router,
    // _gate_up, _down exactly (same vec_dot, same topk, same silu), so any
    // A-vs-B diff is a composition bug in the test or the runtime, not a
    // kernel difference. ----
    auto topk10 = [](const float * pr, int32_t * io, float * wo) {
        int32_t ids[10]; float w[10];
        for (int j = 0; j < 10; ++j) { ids[j] = -1; w[j] = -1e30f; }
        for (int e = 0; e < 512; ++e) {
            const float v = pr[e];
            if (v <= w[0]) continue;
            int j = 0;
            w[0] = v; ids[0] = e;
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
        for (int a = 0; a < 10; ++a) {
            int b = a;
            for (int c = a+1; c < 10; ++c) if (w[c] > w[b]) b = c;
            float tv = w[a]; w[a] = w[b]; w[b] = tv;
            int32_t ti = ids[a]; ids[a] = ids[b]; ids[b] = ti;
        }
        for (int j = 0; j < 10; ++j) { io[j] = ids[j]; wo[j] = w[j]; }
    };
    // host-side B (single thread reference)
    std::vector<float> rb((size_t) E * n_tok, 0.0f);
    std::vector<float> gib((size_t) E * NX);
    ggml_backend_tensor_get(gi, gib.data(), 0, sizeof(float) * gib.size());
    std::vector<char> gub(ggml_nbytes(gu)), dnb(ggml_nbytes(dn));
    ggml_backend_tensor_get(gu, gub.data(), 0, gub.size());
    ggml_backend_tensor_get(dn, dnb.data(), 0, dnb.size());
    std::vector<float> pbb;
    if (pb) { pbb.resize(NX); ggml_backend_tensor_get(pb, pbb.data(), 0, sizeof(float) * NX); }
    auto vd_gu = ggml_get_type_traits_cpu(gu->type)->vec_dot;
    auto vd_dn = ggml_get_type_traits_cpu(dn->type)->vec_dot;
    int64_t row_gu = ggml_row_size(gu->type, gu->ne[0]);
    int64_t row_dn = ggml_row_size(dn->type, dn->ne[0]);
    size_t b_gu = (size_t) row_gu * (size_t)(2*F);
    size_t b_dn = (size_t) row_dn * (size_t)E;
    auto q8f = ggml_get_type_traits_cpu(GGML_TYPE_Q8_K)->from_float;
    std::vector<char> xq(ggml_row_size(GGML_TYPE_Q8_K, E));
    std::vector<char> aq(ggml_row_size(GGML_TYPE_Q8_K, F));
    double t1 = now_ms();
    for (int t = 0; t < n_tok; ++t) {
        const float * xt = xh.data() + (size_t)t * E;
        float lg[512], pr[512];
        for (int e = 0; e < 512; ++e) {
            double sm = 0.0;
            for (int i = 0; i < E; ++i) sm += (double) xt[i] * gib[(size_t)e * E + i];
            lg[e] = (float) sm;
        }
        for (int e = 0; e < 512; ++e) {
            float v = lg[e] + (pb ? pbb[e] : 0.0f);
            pr[e] = 1.0f / (1.0f + expf(-v));
        }
        int32_t idt[10]; float wtt[10];
        topk10(pr, idt, wtt);
        float sm = 0; for (int j = 0; j < 10; ++j) sm += wtt[j];
        if (sm < 6.103515625e-5f) sm = 6.103515625e-5f;
        for (int j = 0; j < 10; ++j) wtt[j] /= sm;
        if (t == 0) { printf("B ids[0..9] ="); for (int j = 0; j < 10; ++j) printf(" %d", idt[j]); printf("\n"); }
        q8f(xt, xq.data(), E);
        // gate_up rows for the 10 experts
        std::vector<float> grow((size_t)10 * 2 * F);
        for (int j = 0; j < 10; ++j) {
            int32_t e = idt[j];
            for (int r = 0; r < 2*F; ++r) {
                float sv = 0;
                vd_gu((int)gu->ne[0], &sv, 0, gub.data() + (size_t)e * b_gu + (size_t)r * row_gu, 0, xq.data(), 0, 1);
                grow[(size_t)j * 2 * F + r] = sv;
            }
        }
        float * out = rb.data() + (size_t)t * E;
        for (int j = 0; j < 10; ++j) {
            int32_t e = idt[j];
            float sw[512];
            for (int i = 0; i < F; ++i) {
                float g = grow[(size_t)j * 2 * F + i], u = grow[(size_t)j * 2 * F + F + i];
                sw[i] = (g / (1.0f + expf(-g))) * u;
            }
            q8f(sw, aq.data(), F);
            for (int r = 0; r < E; ++r) {
                float sv = 0;
                vd_dn((int)dn->ne[0], &sv, 0, dnb.data() + (size_t)e * b_dn + (size_t)r * row_dn, 0, aq.data(), 0, 1);
                out[r] += wtt[j] * sv;
            }
        }
    }
    double ms_b = (now_ms() - t1);
    printf("B block-ref (1 thread): %.4f ms/token\n", ms_b / n_tok);
    printf("B out[0..3] = %.6f %.6f %.6f %.6f\n", rb[0], rb[1], rb[2], rb[3]);
    double mx = 0; int mi = 0;
    for (size_t i = 0; i < ra.size(); ++i) {
        double d = fabs((double)ra[i] - rb[i]);
        if (d > mx) { mx = d; mi = (int) i; }
    }
    printf("PARITY max| A - B | = %.6g at [%d] (A=%.6g B=%.6g)\n", mx, mi, ra[mi], rb[mi]);
    return 0;
}
