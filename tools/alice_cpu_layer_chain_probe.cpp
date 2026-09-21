// alice_cpu_layer_chain_probe.cpp — attribute the deployed CPU MoE layer.
//
// The deployed path costs ~1.58 ms/layer while isolated expert mm_id arithmetic
// is only ~0.20 ms (ALICE_VK_MMID_SHAPE_PROBE.json / SCATTER probe). This probe
// rebuilds the EXACT deployed per-layer op chain on the CPU backend at
// n_tokens=1 and times every op class separately, so the ~1.35 ms gap is
// attributed rather than guessed:
//
//   router mm          build_lora_mm(ffn_gate_inp, ffn_inp)   [2048,512] f32
//   sigmoid            probs over 512
//   bias add           exp_probs_b (512)
//   argsort_top_k      over 512 experts -> 10 ids
//   weights get_rows   [1,10,1]
//   renorm chain       sum_rows + clamp + div
//   gate_up mm_id      [2048,1024,512] Q3_K or Q4_K, 10 ids
//   swiglu_split
//   weights mul
//   aggregate          10 views + 9 adds
//   shared expert      3 dense mm (up/gate [2048,512], down [512,2048]) + gate
//
// Each class gets a fresh ctx; ALL tensors (inputs + op results) are allocated
// by ggml_backend_alloc_ctx_tensors_from_buft BEFORE any tensor_set, because
// ggml_backend_tensor_set asserts on an unallocated buffer. 5 warmup + 200
// reps, backend-synchronized.
//
// build: g++ -O2 -std=c++17 -I include -I ggml/include tools/alice_cpu_layer_chain_probe.cpp \
//        -L build-vk/bin -lggml -lggml-base -lggml-cpu \
//        -Wl,-rpath,$PWD/build-vk/bin
// run:   ./alice_cpu_layer_chain_probe [Q3_K|Q4_K]

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <chrono>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

static double now_ms() {
    using namespace std::chrono;
    return duration<double, std::milli>(steady_clock::now().time_since_epoch()).count();
}

static const int64_t N_EMBD = 2048;
static const int64_t N_FF   = 512;
static const int64_t N_EXP  = 512;
static const int64_t N_USED = 10;

// build the graph in a fresh ctx, allocate everything, set inputs, time it
template <typename F, typename S>
static double run_class(ggml_backend_t bk, ggml_backend_buffer_type_t buft, F build, S set_inputs) {
    struct ggml_init_params ip = { 64*1024*1024, nullptr, true };
    ggml_context * c = ggml_init(ip);
    ggml_tensor * out = build(c);
    ggml_backend_buffer_t b = ggml_backend_alloc_ctx_tensors_from_buft(c, buft);
    if (!b) { fprintf(stderr, "class alloc failed\n"); ggml_free(c); return -1.0; }
    set_inputs(out);
    ggml_cgraph * gf = ggml_new_graph(c);
    ggml_build_forward_expand(gf, out);
    for (int i = 0; i < 5; ++i) {
        if (ggml_backend_graph_compute(bk, gf) != GGML_STATUS_SUCCESS) { ggml_free(c); return -2.0; }
    }
    ggml_backend_synchronize(bk);
    const int reps = 200;
    double t0 = now_ms();
    for (int i = 0; i < reps; ++i) {
        if (ggml_backend_graph_compute(bk, gf) != GGML_STATUS_SUCCESS) { ggml_free(c); return -2.0; }
    }
    ggml_backend_synchronize(bk);
    double ms = (now_ms() - t0) / reps;
    ggml_free(c);
    return ms;
}

// find the unique leaf f32 [N_EMBD,1] input in a graph
static ggml_tensor * find_embd_leaf(ggml_tensor * root) {
    std::vector<ggml_tensor *> stk = { root };
    ggml_tensor * found = nullptr;
    while (!stk.empty()) {
        ggml_tensor * t = stk.back(); stk.pop_back();
        bool is_leaf = true;
        for (int i = 0; i < GGML_MAX_SRC; ++i) if (t->src[i]) { is_leaf = false; stk.push_back(t->src[i]); }
        if (is_leaf && t->type == GGML_TYPE_F32 && t->ne[0] == N_EMBD && t->ne[1] == 1) found = t;
    }
    return found;
}

int main(int argc, char ** argv) {
    ggml_type qt = (argc > 1 && strcmp(argv[1], "Q4_K") == 0) ? GGML_TYPE_Q4_K : GGML_TYPE_Q3_K;
    ggml_backend_load_all();
    ggml_backend_dev_t cpu_dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    ggml_backend_t cpu = ggml_backend_dev_init(cpu_dev, nullptr);
    ggml_backend_buffer_type_t buft = ggml_backend_get_default_buffer_type(cpu);
    printf("quant: %s   backend: %s  n_tokens=1  n_used=10  (5 warmup + 200 reps)\n",
           ggml_type_name(qt), ggml_backend_name(cpu));

    // ---- host weights, allocated once, shared by every class ----
    struct ggml_init_params wp = { 256*1024*1024, nullptr, true };
    ggml_context * wctx = ggml_init(wp);
    ggml_tensor * gate_inp    = ggml_new_tensor_2d(wctx, GGML_TYPE_F32, N_EMBD, N_EXP);
    ggml_tensor * gu          = ggml_new_tensor_3d(wctx, qt, N_EMBD, 2*N_FF, N_EXP);
    ggml_tensor * dn          = ggml_new_tensor_3d(wctx, qt, N_FF,   N_EMBD,   N_EXP);
    ggml_tensor * up_sh       = ggml_new_tensor_2d(wctx, GGML_TYPE_F32, N_EMBD, N_FF);
    ggml_tensor * gate_sh     = ggml_new_tensor_2d(wctx, GGML_TYPE_F32, N_EMBD, N_FF);
    ggml_tensor * down_sh     = ggml_new_tensor_2d(wctx, GGML_TYPE_F32, N_FF, N_EMBD);
    ggml_tensor * gate_inp_sh = ggml_new_tensor_2d(wctx, GGML_TYPE_F32, N_EMBD, 1);
    ggml_tensor * probs_b     = ggml_new_tensor_1d(wctx, GGML_TYPE_F32, N_EXP);
    ggml_backend_buffer_t wbuf = ggml_backend_alloc_ctx_tensors_from_buft(wctx, buft);
    if (!wbuf) { fprintf(stderr, "weight alloc failed\n"); return 1; }
    {
        std::mt19937 rng(1234);
        auto fill = [&](ggml_tensor * t) {
            std::vector<float> h((size_t) ggml_nelements(t));
            for (auto & v : h) v = (float) ((rng() % 2000) - 1000) / 30000.0f;
            ggml_backend_tensor_set(t, h.data(), 0, ggml_nbytes(t));
        };
        fill(gate_inp); fill(gu); fill(dn); fill(up_sh); fill(gate_sh);
        fill(down_sh); fill(gate_inp_sh); fill(probs_b);
    }

    // ---- host input data ----
    std::vector<float> xh((size_t) N_EMBD);
    for (int64_t i = 0; i < N_EMBD; ++i) xh[i] = (float) ((i % 13) - 6) / 6.0f;
    std::vector<float> s512((size_t) N_EXP);
    for (int i = 0; i < N_EXP; ++i) s512[i] = 0.25f;
    std::vector<float> b512((size_t) N_EXP);
    for (int i = 0; i < N_EXP; ++i) b512[i] = 0.10f;
    std::vector<float> as512((size_t) N_EXP);
    for (int i = 0; i < N_EXP; ++i) as512[i] = (float) (i % 97);
    std::vector<float> p512((size_t) N_EXP);
    for (int i = 0; i < N_EXP; ++i) p512[i] = (float) (i % 31) / 31.0f;
    std::vector<float> w10((size_t) N_USED);
    for (int i = 0; i < N_USED; ++i) w10[i] = 1.0f / N_USED;
    std::vector<float> n10((size_t) N_USED);
    for (int i = 0; i < N_USED; ++i) n10[i] = (float) (i + 1);
    std::vector<int32_t> ih10;
    for (int i = 0; i < N_USED; ++i) ih10.push_back(i * 51 % (int) N_EXP);
    std::vector<float> e2048x10((size_t) N_EMBD * N_USED);
    for (size_t i = 0; i < e2048x10.size(); ++i) e2048x10[i] = (float) ((i % 17) - 8) / 8.0f;

    struct row { const char * name; double ms; };
    std::vector<row> rows;

    // 1) router mm [512,1] = [2048,512] x [2048,1]
    rows.push_back({"router_mm", run_class(cpu, buft,
        [&](ggml_context * c) { return ggml_mul_mat(c, gate_inp, ggml_new_tensor_2d(c, GGML_TYPE_F32, N_EMBD, 1)); },
        [&](ggml_tensor * o) { ggml_backend_tensor_set(o->src[1], xh.data(), 0, ggml_nbytes(o->src[1])); })});

    // 2) sigmoid over 512
    rows.push_back({"sigmoid_512", run_class(cpu, buft,
        [&](ggml_context * c) { return ggml_sigmoid(c, ggml_new_tensor_2d(c, GGML_TYPE_F32, N_EXP, 1)); },
        [&](ggml_tensor * o) { ggml_backend_tensor_set(o->src[0], s512.data(), 0, ggml_nbytes(o->src[0])); })});

    // 3) bias add 512
    rows.push_back({"bias_add_512", run_class(cpu, buft,
        [&](ggml_context * c) { return ggml_add(c, ggml_new_tensor_2d(c, GGML_TYPE_F32, N_EXP, 1), probs_b); },
        [&](ggml_tensor * o) { ggml_backend_tensor_set(o->src[0], b512.data(), 0, ggml_nbytes(o->src[0])); })});

    // 4) argsort_top_k over 512 -> 10
    rows.push_back({"argsort_topk_512", run_class(cpu, buft,
        [&](ggml_context * c) { return ggml_argsort_top_k(c, ggml_new_tensor_2d(c, GGML_TYPE_F32, N_EXP, 1), N_USED); },
        [&](ggml_tensor * o) { ggml_backend_tensor_set(o->src[0], as512.data(), 0, ggml_nbytes(o->src[0])); })});

    // 5) weights get_rows [1,10,1] from probs [1,512,1]
    rows.push_back({"weights_get_rows", run_class(cpu, buft,
        [&](ggml_context * c) {
            ggml_tensor * ids = ggml_new_tensor_1d(c, GGML_TYPE_I32, N_USED);
            ggml_tensor * pr  = ggml_new_tensor_3d(c, GGML_TYPE_F32, 1, N_EXP, 1);
            return ggml_get_rows(c, pr, ids);
        },
        [&](ggml_tensor * o) {
            ggml_backend_tensor_set(o->src[0], p512.data(), 0, ggml_nbytes(o->src[0]));
            ggml_backend_tensor_set(o->src[1], ih10.data(), 0, ggml_nbytes(o->src[1]));
        })});

    // 6) renorm chain: sum_rows + clamp + div
    rows.push_back({"renorm_chain", run_class(cpu, buft,
        [&](ggml_context * c) {
            ggml_tensor * w = ggml_new_tensor_2d(c, GGML_TYPE_F32, N_USED, 1);
            ggml_tensor * s = ggml_sum_rows(c, w);
            s = ggml_clamp(c, s, 6.103515625e-5, INFINITY);
            return ggml_div(c, w, s);
        },
        [&](ggml_tensor * o) { ggml_backend_tensor_set(o->src[0], n10.data(), 0, ggml_nbytes(o->src[0])); })});

    // 7) expert pair + swiglu + weights mul
    rows.push_back({"expert_pair+swiglu+wmul", run_class(cpu, buft,
        [&](ggml_context * c) {
            ggml_tensor * ids = ggml_new_tensor_1d(c, GGML_TYPE_I32, N_USED);
            ggml_tensor * w   = ggml_new_tensor_3d(c, GGML_TYPE_F32, 1, N_USED, 1);
            ggml_tensor * xin = ggml_new_tensor_2d(c, GGML_TYPE_F32, N_EMBD, 1);
            ggml_tensor * xr  = ggml_reshape_3d(c, xin, N_EMBD, 1, 1);
            ggml_tensor * guo = ggml_mul_mat_id(c, gu, xr, ids);
            ggml_tensor * gg  = ggml_view_3d(c, guo, N_FF, N_USED, 1, guo->nb[1], guo->nb[2], 0);
            ggml_tensor * uu  = ggml_view_3d(c, guo, N_FF, N_USED, 1, guo->nb[1], guo->nb[2], N_FF*guo->nb[0]);
            ggml_tensor * act = ggml_swiglu_split(c, gg, uu);
            ggml_tensor * dno = ggml_mul_mat_id(c, dn, act, ids);
            return ggml_mul(c, dno, w);
        },
        [&](ggml_tensor * o) {
            ggml_tensor * dno = o->src[0];
            ggml_tensor * ids = dno->src[2];
            ggml_tensor * act = dno->src[1];
            ggml_tensor * gg  = act->src[0];
            ggml_tensor * guo = gg->src[0];
            ggml_tensor * xin = guo->src[1]->src[0];
            ggml_backend_tensor_set(ids, ih10.data(), 0, ggml_nbytes(ids));
            ggml_backend_tensor_set(o->src[1], w10.data(), 0, ggml_nbytes(o->src[1]));
            ggml_backend_tensor_set(xin, xh.data(), 0, ggml_nbytes(xin));
        })});

    // 7b) mm_id pair + swiglu only (arithmetic floor)
    rows.push_back({"  of which mm_id+swiglu only", run_class(cpu, buft,
        [&](ggml_context * c) {
            ggml_tensor * ids = ggml_new_tensor_1d(c, GGML_TYPE_I32, N_USED);
            ggml_tensor * xin = ggml_new_tensor_2d(c, GGML_TYPE_F32, N_EMBD, 1);
            ggml_tensor * xr  = ggml_reshape_3d(c, xin, N_EMBD, 1, 1);
            ggml_tensor * guo = ggml_mul_mat_id(c, gu, xr, ids);
            ggml_tensor * gg  = ggml_view_3d(c, guo, N_FF, N_USED, 1, guo->nb[1], guo->nb[2], 0);
            ggml_tensor * uu  = ggml_view_3d(c, guo, N_FF, N_USED, 1, guo->nb[1], guo->nb[2], N_FF*guo->nb[0]);
            ggml_tensor * act = ggml_swiglu_split(c, gg, uu);
            return ggml_mul_mat_id(c, dn, act, ids);
        },
        [&](ggml_tensor * o) {
            ggml_tensor * ids = o->src[2];
            ggml_tensor * act = o->src[1];
            ggml_tensor * gg  = act->src[0];
            ggml_tensor * guo = gg->src[0];
            ggml_tensor * xin = guo->src[1]->src[0];
            ggml_backend_tensor_set(ids, ih10.data(), 0, ggml_nbytes(ids));
            ggml_backend_tensor_set(xin, xh.data(), 0, ggml_nbytes(xin));
        })});

    // 8) aggregate: 10 views + 9 adds
    rows.push_back({"aggregate_9adds", run_class(cpu, buft,
        [&](ggml_context * c) {
            ggml_tensor * e = ggml_new_tensor_3d(c, GGML_TYPE_F32, N_EMBD, N_USED, 1);
            ggml_tensor * views[16];
            for (int i = 0; i < N_USED; ++i)
                views[i] = ggml_view_2d(c, e, N_EMBD, 1, e->nb[2], i*e->nb[1]);
            ggml_tensor * o = views[0];
            for (int i = 1; i < N_USED; ++i) o = ggml_add(c, o, views[i]);
            return o;
        },
        [&](ggml_tensor * o) {
            // the leaf is the parent of the first VIEW in the chain
            ggml_tensor * e = nullptr;
            std::vector<ggml_tensor *> stk = { o };
            while (!stk.empty() && e == nullptr) {
                ggml_tensor * t = stk.back(); stk.pop_back();
                for (int i = 0; i < GGML_MAX_SRC; ++i) if (t->src[i]) stk.push_back(t->src[i]);
                if (t->op == GGML_OP_VIEW && t->ne[2] == N_USED && t->ne[0] == N_EMBD) e = t->src[0];
            }
            if (e) ggml_backend_tensor_set(e, e2048x10.data(), 0, ggml_nbytes(e));
        })});

    // 9) shared expert: 3 dense mm + silu + mul + sigmoid gate + mul + add
    rows.push_back({"shared_expert_chain", run_class(cpu, buft,
        [&](ggml_context * c) {
            ggml_tensor * xin = ggml_new_tensor_2d(c, GGML_TYPE_F32, N_EMBD, 1);
            ggml_tensor * up  = ggml_mul_mat(c, up_sh, xin);
            ggml_tensor * gt  = ggml_mul_mat(c, gate_sh, xin);
            ggml_tensor * a   = ggml_silu(c, gt);
            ggml_tensor * m   = ggml_mul(c, up, a);
            ggml_tensor * d   = ggml_mul_mat(c, down_sh, m);
            ggml_tensor * gi  = ggml_mul_mat(c, gate_inp_sh, xin);
            gi = ggml_sigmoid(c, gi);
            return ggml_add(c, d, ggml_mul(c, d, gi));
        },
        [&](ggml_tensor * o) {
            ggml_tensor * xin = find_embd_leaf(o);
            if (xin) ggml_backend_tensor_set(xin, xh.data(), 0, ggml_nbytes(xin));
        })});

    printf("%-28s %10s\n", "class", "ms/call");
    double sum = 0.0;
    for (auto & r : rows) {
        printf("%-28s %10.4f\n", r.name, r.ms);
        if (r.name[0] != ' ') sum += r.ms;
    }
    printf("%-28s %10.4f\n", "SUM(classes)", sum);
    printf("%-28s %10.4f\n", "deployed per-layer (measured)", 1.58);
    printf("%-28s %10.4f\n", "unattributed", 1.58 - sum);
    return 0;
}
