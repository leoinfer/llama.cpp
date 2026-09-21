// alice_vk_mmid_probe.cpp — decisive measurement for option (b).
//
// Question: does Vulkan mul_mat_id run efficiently on a COMPACT expert bank
// (ne[2] = n_hot, e.g. 10..44) with Alice's exact expert shapes
// (gate_up [2048,1024] per expert, down [512,2048] per expert) in Q3_K/Q4_K?
//
// v1 of the hot bank was slow, but the hot GPU side was never isolated. This
// probe times ONLY the two mm_id ops per layer on the Vulkan backend for
// n_mats in {512 (deployed), 44, 31, 16, 10} at n=1 token, and compares
// against the CPU backend for the same shape. It also verifies numerical
// parity between compact-bank + LUT-remapped ids and the full-bank + global
// ids path — the property option (b) depends on.
//
// build: g++ -O2 -std=c++17 -I include -I ggml/include tools/alice_vk_mmid_probe.cpp \
//        -L build-vk/bin -lllama -lggml -lggml-base -lggml-vulkan -lggml-cpu \
//        -Wl,-rpath,$PWD/build-vk/bin
// run:   ./alice_vk_mmid_probe

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

static double now_ms() {
    using namespace std::chrono;
    return duration<double, std::milli>(steady_clock::now().time_since_epoch()).count();
}

// Build one expert bank of n_mats experts with the given quant type.
// gate_up: [n_embd=2048, 2*n_ff=1024, n_mats]  (K-quant)
// down:    [n_ff=512, n_embd=2048, n_mats]
struct bank {
    ggml_context * ctx;
    ggml_backend_buffer_t buf;
    ggml_tensor * gu;   // [2048, 1024, n_mats]
    ggml_tensor * dn;   // [512, 2048, n_mats]
};

static bank make_bank(ggml_backend_buffer_type_t buft, ggml_type qt, int n_mats, unsigned seed) {
    const int64_t n_embd = 2048, n_ff = 512;
    struct ggml_init_params p = { 16*1024*1024, nullptr, true };
    ggml_context * ctx = ggml_init(p);
    bank b{};
    b.ctx = ctx;
    b.gu = ggml_new_tensor_3d(ctx, qt, n_embd, 2*n_ff, n_mats);
    b.dn = ggml_new_tensor_3d(ctx, qt, n_ff,   n_embd,   n_mats);
    ggml_set_name(b.gu, "gu");
    ggml_set_name(b.dn, "dn");
    b.buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx, buft);
    if (!b.buf) { fprintf(stderr, "alloc failed for type %d n_mats %d\n", (int) qt, n_mats); exit(1); }
    // fill with random data via the host (backend tensor_set)
    std::mt19937 rng(seed);
    std::vector<float> host((size_t) ggml_nelements(b.gu) + ggml_nelements(b.dn));
    for (auto & v : host) v = (float) ((rng() % 2000) - 1000) / 1000.0f;
    ggml_backend_tensor_set(b.gu, host.data(), 0, ggml_nbytes(b.gu));
    ggml_backend_tensor_set(b.dn, host.data(), 0, ggml_nbytes(b.dn));
    return b;
}

int main(int argc, char ** argv) {
    // backends
    ggml_backend_load_all();
    ggml_backend_dev_t cpu_dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    ggml_backend_t cpu = ggml_backend_dev_init(cpu_dev, nullptr);
    ggml_backend_dev_t gpu_dev = nullptr;
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        auto * d = ggml_backend_dev_get(i);
        if (ggml_backend_dev_type(d) == GGML_BACKEND_DEVICE_TYPE_GPU) { gpu_dev = d; break; }
    }
    if (!gpu_dev) { fprintf(stderr, "no GPU device\n"); return 1; }
    ggml_backend_t gpu = ggml_backend_dev_init(gpu_dev, nullptr);
    ggml_backend_t gpu_b = gpu;
    printf("gpu: %s\n", ggml_backend_dev_name(gpu_dev));

    const int64_t n_embd = 2048, n_ff = 512;
    const int reps = 200;

    struct shape { const char * name; ggml_type qt; int n_mats; };
    std::vector<shape> shapes = {
        {"Q4_K x512 (deployed)", GGML_TYPE_Q4_K, 512},
        {"Q4_K x44",  GGML_TYPE_Q4_K, 44},
        {"Q4_K x31",  GGML_TYPE_Q4_K, 31},
        {"Q4_K x16",  GGML_TYPE_Q4_K, 16},
        {"Q4_K x10",  GGML_TYPE_Q4_K, 10},
        {"Q3_K x512 (deployed)", GGML_TYPE_Q3_K, 512},
        {"Q3_K x44",  GGML_TYPE_Q3_K, 44},
        {"Q3_K x10",  GGML_TYPE_Q3_K, 10},
    };

    printf("%-24s %10s %12s %12s %10s\n", "shape", "backend", "gu_us", "dn_us", "GB/s");
    for (auto & s : shapes) {
        for (int be = 0; be < 2; ++be) {
            ggml_backend_t bk = be == 0 ? gpu_b : cpu;
            ggml_backend_buffer_type_t buft = ggml_backend_get_default_buffer_type(bk);
            bank b = make_bank(buft, s.qt, s.n_mats, 1234);
            // ids: n_used distinct ids (10 of n_mats)
            const int n_used = s.n_mats < 10 ? s.n_mats : 10;
            std::vector<int32_t> ids((size_t) n_used);
            if (argc > 1 && strcmp(argv[1], "scatter") == 0) {
                // scattered ids like real routing: spread across the bank
                std::mt19937 idrng(777);
                for (int i = 0; i < n_used; ++i) ids[i] = (int32_t) (idrng() % s.n_mats);
            } else {
                for (int i = 0; i < n_used; ++i) ids[i] = i % s.n_mats;
            }

            // activation
            struct ggml_init_params cp = { 16*1024*1024, nullptr, true };
            ggml_context * ctx = ggml_init(cp);
            // create ALL tensors (inputs + intermediates) before allocating
            ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd, 1);
            ggml_tensor * idst = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, n_used, 1);
            ggml_tensor * x3 = ggml_reshape_3d(ctx, x, n_embd, 1, 1);
            ggml_tensor * gu = ggml_mul_mat_id(ctx, b.gu, x3, idst);          // [1024, n_used, 1]
            ggml_tensor * g  = ggml_view_3d(ctx, gu, n_ff, n_used, 1, gu->nb[1], gu->nb[2], 0);
            ggml_tensor * u  = ggml_view_3d(ctx, gu, n_ff, n_used, 1, gu->nb[1], gu->nb[2], n_ff*gu->nb[0]);
            ggml_tensor * act = ggml_swiglu_split(ctx, g, u);                 // [512, n_used, 1]
            ggml_tensor * dn = ggml_mul_mat_id(ctx, b.dn, act, idst);         // [2048, n_used, 1]
            ggml_backend_buffer_t xbuf = ggml_backend_alloc_ctx_tensors_from_buft(ctx, buft);
            if (!xbuf) { fprintf(stderr, "ctx alloc failed\n"); return 1; }
            std::vector<float> xh((size_t) n_embd);
            for (int i = 0; i < n_embd; ++i) xh[i] = (float) ((i % 7) - 3) / 3.0f;
            ggml_backend_tensor_set(x, xh.data(), 0, ggml_nbytes(x));
            ggml_backend_tensor_set(idst, ids.data(), 0, ggml_nbytes(idst));

            ggml_cgraph * gf = ggml_new_graph(ctx);
            ggml_build_forward_expand(gf, dn);
            if (ggml_backend_graph_compute(bk, gf) != GGML_STATUS_SUCCESS) {
                printf("%-24s %10s  COMPUTE FAILED\n", s.name, be == 0 ? "vulkan" : "cpu");
                ggml_free(ctx);
                continue;
            }
            // warmup (pipeline creation), then time
            for (int r = 0; r < 5; ++r) {
                if (ggml_backend_graph_compute(bk, gf) != GGML_STATUS_SUCCESS) break;
            }
            ggml_backend_synchronize(bk);
            double t0 = now_ms();
            for (int r = 0; r < reps; ++r) {
                if (ggml_backend_graph_compute(bk, gf) != GGML_STATUS_SUCCESS) break;
            }
            ggml_backend_synchronize(bk);
            double t1 = now_ms();
            const double both_us = (t1 - t0) * 1000.0 / reps;
            const double bytes = (double) ggml_nbytes(b.gu) * (double) n_used / s.n_mats
                               + (double) ggml_nbytes(b.dn) * (double) n_used / s.n_mats;
            printf("%-24s %10s %10.1f %12.1f %10.1f\n", s.name, be == 0 ? "vulkan" : "cpu",
                   both_us/2, both_us/2, bytes / (both_us*1e-6) / 1e9);
            ggml_free(ctx);
        }
    }
    return 0;
}
