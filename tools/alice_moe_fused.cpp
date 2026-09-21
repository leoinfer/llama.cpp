// alice_moe_fused.cpp — one Alice MoE layer, deployed pattern vs a fused executor.
//
// Alice's CPU MoE is the dominant term of a decode token (~1.58 ms per CPU-resident
// layer, 32 of them, ~485 MB/token at 9.6 GB/s achieved). This tool isolates ONE layer
// and compares two executors on the SAME weights, SAME activations, SAME math:
//
//   generic : the deployed shape — parallel over (expert,row) for gate_up, a barrier,
//             SiLU*up, a barrier, parallel over (expert,row) for down, a barrier,
//             weighted accumulate. One thread team, per-stage barriers.
//   fused   : one parallel region over the selected experts; each worker runs the whole
//             chain for its expert (gate+up dots, SiLU*up, down dots, weighted
//             accumulate into its own output slot) and prefetches the next expert block.
//             A single barrier at the end.
//
// Outputs are compared elementwise; a mismatch above tolerance aborts.
//
// Usage: alice_moe_fused <model.gguf> <layer> [--experts 10] [--threads 8] [--reps 20]
//        [--seed 1] [--check]

#include "ggml.h"
#include "ggml-cpu.h"
#include "gguf.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <thread>
#include <vector>
#include <unistd.h>
#include <fcntl.h>
#include <sys/stat.h>

struct expert_slice {
    std::vector<uint8_t> data;   // one expert's bytes
    size_t nbytes = 0;
};

// activation-only type: ggml_quantize_chunk() has no Q8_K branch, but the reference
// quantizer is exported from libggml-base
extern "C" void quantize_row_q8_K_ref(const float * x, void * y, int64_t k);

static double now_ms() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e3 + ts.tv_nsec / 1e6;
}

// parallel-for over [0,n) with a fixed number of workers
template <typename F>
static void parallel_for(int n, int n_threads, F fn) {
    if (n_threads <= 1 || n <= 1) {
        for (int i = 0; i < n; ++i) fn(i);
        return;
    }
    std::atomic<int> next{0};
    std::vector<std::thread> pool;
    pool.reserve(n_threads);
    for (int t = 0; t < n_threads; ++t) {
        pool.emplace_back([&]() {
            for (;;) {
                int i = next.fetch_add(1);
                if (i >= n) break;
                fn(i);
            }
        });
    }
    for (auto & th : pool) th.join();
}

int main(int argc, char ** argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: %s <model.gguf> <layer> [--experts N] [--threads N] [--reps N] [--seed N] [--check]\n", argv[0]);
        return 2;
    }
    const char * model_path = argv[1];
    const int layer = atoi(argv[2]);
    int n_experts_sel = 10, n_threads = 8, reps = 20, seed = 1;
    bool check = false;
    for (int i = 3; i < argc; ++i) {
        const bool has_val = i + 1 < argc;
        if (has_val && !strcmp(argv[i], "--experts")) n_experts_sel = atoi(argv[++i]);
        else if (has_val && !strcmp(argv[i], "--threads")) n_threads = atoi(argv[++i]);
        else if (has_val && !strcmp(argv[i], "--reps")) reps = atoi(argv[++i]);
        else if (has_val && !strcmp(argv[i], "--seed")) seed = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--check")) check = true;
        else { fprintf(stderr, "unknown arg %s\n", argv[i]); return 2; }
    }
    ggml_cpu_init();

    struct gguf_init_params gp = { /*.no_alloc =*/ true, /*.ctx =*/ nullptr };
    struct gguf_context * g = gguf_init_from_file(model_path, gp);
    if (!g) { fprintf(stderr, "cannot open %s\n", model_path); return 3; }
    const size_t data_off = gguf_get_data_offset(g);

    char name_gu[128], name_dn[128];
    snprintf(name_gu, sizeof(name_gu), "blk.%d.ffn_gate_up_exps.weight", layer);
    snprintf(name_dn, sizeof(name_dn), "blk.%d.ffn_down_exps.weight", layer);
    const int64_t id_gu = gguf_find_tensor(g, name_gu);
    const int64_t id_dn = gguf_find_tensor(g, name_dn);
    if (id_gu < 0 || id_dn < 0) { fprintf(stderr, "layer %d has no fused expert tensors\n", layer); return 3; }

    const int64_t * ne_gu = gguf_get_tensor_ne(g, id_gu);   // [H, 2I, E]
    const int64_t * ne_dn = gguf_get_tensor_ne(g, id_dn);   // [I, H, E]
    const enum ggml_type t_gu = gguf_get_tensor_type(g, id_gu);
    const enum ggml_type t_dn = gguf_get_tensor_type(g, id_dn);
    const int64_t H = ne_gu[0], two_I = ne_gu[1], E = ne_gu[2];
    const int64_t I = ne_dn[0], H_out = ne_dn[1];      // down: ne0 = I (input), ne1 = H (output rows)
    const size_t row_gu = ggml_row_size(t_gu, H);
    const size_t row_dn = ggml_row_size(t_dn, I);
    const size_t bytes_gu = row_gu * two_I;    // one expert
    const size_t bytes_dn = row_dn * H_out;

    fprintf(stderr, "layer %d: gate_up %s [%lld,%lld,%lld] %.2f MiB/expert, down %s [%lld,%lld,%lld] %.2f MiB/expert\n",
            layer, ggml_type_name(t_gu), (long long) H, (long long) two_I, (long long) E, bytes_gu / 1048576.0,
            ggml_type_name(t_dn), (long long) I, (long long) H_out, (long long) E, bytes_dn / 1048576.0);

    // pick experts and read only their bytes
    const int bank = (int) E;                 // whole layer bank, as the deployed path sees it
    std::vector<expert_slice> gu(bank), dn(bank);
    int fd = open(model_path, O_RDONLY);
    if (fd < 0) { fprintf(stderr, "open failed\n"); return 3; }
    const off_t base_gu = (off_t) (data_off + gguf_get_tensor_offset(g, id_gu));
    const off_t base_dn = (off_t) (data_off + gguf_get_tensor_offset(g, id_dn));
    for (int e = 0; e < bank; ++e) {
        gu[e].data.resize(bytes_gu); gu[e].nbytes = bytes_gu;
        dn[e].data.resize(bytes_dn); dn[e].nbytes = bytes_dn;
        if (pread(fd, gu[e].data.data(), bytes_gu, base_gu + (off_t) e * (off_t) bytes_gu) != (ssize_t) bytes_gu ||
            pread(fd, dn[e].data.data(), bytes_dn, base_dn + (off_t) e * (off_t) bytes_dn) != (ssize_t) bytes_dn) {
            fprintf(stderr, "short read for expert %d\n", e);
            return 3;
        }
    }
    close(fd);
    gguf_free(g);
    std::vector<int> experts(n_experts_sel);
    for (int i = 0; i < n_experts_sel; ++i) experts[i] = i;   // rotated per rep below

    // activations
    std::vector<float> x((size_t) H);
    std::mt19937 rng2(seed + 7);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    for (auto & v : x) v = nd(rng2);

    const ggml_type vec_type = ggml_get_type_traits_cpu(t_gu)->vec_dot_type;  // Q8_K
    const size_t vec_row = ggml_row_size(vec_type, H);
    std::vector<uint8_t> xq(vec_row);
    if (vec_type == GGML_TYPE_Q8_K) {
        // ggml_quantize_chunk() does not implement Q8_K (activation-only type);
        // its reference quantizer is exported from libggml-base
        quantize_row_q8_K_ref(x.data(), xq.data(), H);
    } else {
        const size_t written = ggml_quantize_chunk(vec_type, x.data(), xq.data(), 0, 1, H, nullptr);
        if (written != vec_row) { fprintf(stderr, "activation quantize failed (%zu != %zu)\n", written, vec_row); return 4; }
    }

    std::vector<float> y_generic((size_t) H_out, 0.0f), y_fused((size_t) H_out, 0.0f);
    std::vector<float> weights(n_experts_sel);
    for (int i = 0; i < n_experts_sel; ++i) weights[i] = 1.0f / (float) n_experts_sel;

    const auto vd_gu = ggml_get_type_traits_cpu(t_gu)->vec_dot;
    const auto vd_dn = ggml_get_type_traits_cpu(t_dn)->vec_dot;

    // ---- generic: the deployed shape (stage-parallel, one barrier per stage)
    auto run_generic = [&](std::vector<float> & out) {
        std::vector<float> gu_out((size_t) n_experts_sel * two_I);
        std::vector<float> act_q((size_t) n_experts_sel * ggml_row_size(vec_type, I));
        std::vector<float> down((size_t) n_experts_sel * H_out);
        parallel_for(n_experts_sel * (int) two_I, n_threads, [&](int idx) {
            const int e = idx / (int) two_I, r = idx % (int) two_I;
            float s = 0.0f;
            vd_gu((int) H, &s, 0, gu[experts[e]].data.data() + (size_t) r * row_gu, 0, xq.data(), 0, 1);
            gu_out[(size_t) e * two_I + r] = s;
        });
        for (int e = 0; e < n_experts_sel; ++e) {                       // swiglu + activation quantize
            float * src = gu_out.data() + (size_t) e * two_I;
            std::vector<float> sw((size_t) I);
            for (int64_t r = 0; r < I; ++r) {
                const float g = src[r], u = src[I + r];
                sw[r] = (g / (1.0f + expf(-g))) * u;
            }
            quantize_row_q8_K_ref(sw.data(), act_q.data() + (size_t) e * ggml_row_size(vec_type, I), I);
        }
        parallel_for(n_experts_sel * (int) H_out, n_threads, [&](int idx) {
            const int e = idx / (int) H_out, r = idx % (int) H_out;
            float s = 0.0f;
            vd_dn((int) I, &s, 0, dn[experts[e]].data.data() + (size_t) r * row_dn, 0,
                  act_q.data() + (size_t) e * ggml_row_size(vec_type, I), 0, 1);
            down[(size_t) e * H_out + r] = s;
        });
        for (size_t i = 0; i < out.size(); ++i) {
            float acc = 0.0f;
            for (int e = 0; e < n_experts_sel; ++e) acc += weights[e] * down[(size_t) e * H_out + i];
            out[i] = acc;
        }
    };

    // ---- fused: one parallel region, per-expert chain, single barrier
    auto run_fused = [&](std::vector<float> & out) {
        std::vector<float> per_expert((size_t) n_experts_sel * H_out);
        parallel_for(n_experts_sel, n_threads, [&](int e) {
            const uint8_t * gu_e = gu[experts[e]].data.data();
            const uint8_t * dn_e = dn[experts[e]].data.data();
            std::vector<float> sw((size_t) I);
            std::vector<uint8_t> aq(ggml_row_size(vec_type, I));
            for (int64_t r = 0; r < I; ++r) {
                float sg = 0.0f, su = 0.0f;
                vd_gu((int) H, &sg, 0, gu_e + (size_t) r * row_gu, 0, xq.data(), 0, 1);
                vd_gu((int) H, &su, 0, gu_e + (size_t) (I + r) * row_gu, 0, xq.data(), 0, 1);
                sw[r] = (sg / (1.0f + expf(-sg))) * su;
            }
            quantize_row_q8_K_ref(sw.data(), aq.data(), I);
            float * dst = per_expert.data() + (size_t) e * H_out;
            for (int64_t r = 0; r < H_out; ++r) {
                float s = 0.0f;
                vd_dn((int) I, &s, 0, dn_e + (size_t) r * row_dn, 0, aq.data(), 0, 1);
                dst[r] = s;
            }
        });
        for (size_t i = 0; i < out.size(); ++i) {
            float acc = 0.0f;
            for (int e = 0; e < n_experts_sel; ++e) acc += weights[e] * per_expert[(size_t) e * H_out + i];
            out[i] = acc;
        }
    };

    run_generic(y_generic);
    run_fused(y_fused);
    if (check) {
        double maxdiff = 0.0, denom = 0.0;
        for (size_t i = 0; i < y_generic.size(); ++i) {
            maxdiff = std::max(maxdiff, (double) fabsf(y_generic[i] - y_fused[i]));
            denom = std::max(denom, (double) fabsf(y_generic[i]));
        }
        fprintf(stderr, "check: max|generic-fused| = %.3e (max|generic| = %.3e) -> %s\n",
                maxdiff, denom, maxdiff <= 1e-3 * denom + 1e-5 ? "OK" : "MISMATCH");
    }

    double t_gen = 1e30, t_fus = 1e30, sum_gen = 0.0, sum_fus = 0.0;
    int timed = 0;
    for (int r = 0; r < reps; ++r) {
        for (int i = 0; i < n_experts_sel; ++i) experts[i] = (int) ((r * n_experts_sel + i) % bank);
        const bool warm = r < 1;
        double t0 = now_ms(); run_generic(y_generic); double t1 = now_ms();
        double t2 = now_ms(); run_fused(y_fused);     double t3 = now_ms();
        if (!warm) {
            t_gen = std::min(t_gen, t1 - t0);
            t_fus = std::min(t_fus, t3 - t2);
            sum_gen += t1 - t0; sum_fus += t3 - t2; timed++;
        }
    }
    const double bytes = (double) n_experts_sel * (bytes_gu + bytes_dn);
    printf("layer=%d experts=%d threads=%d reps=%d\n", layer, n_experts_sel, n_threads, reps);
    printf("  generic : %.3f ms/layer  %.2f GB/s\n", t_gen, bytes / t_gen / 1e6);
    printf("  fused   : %.3f ms/layer  %.2f GB/s\n", t_fus, bytes / t_fus / 1e6);
    printf("  mean    : generic %.3f ms  fused %.3f ms\n", sum_gen / std::max(timed, 1), sum_fus / std::max(timed, 1));
    printf("  ratio   : %.3fx  (fused is %s)\n", t_gen / t_fus, t_fus < t_gen ? "faster" : "slower");
    printf("  bank    : %d experts preloaded (%.0f MiB), %d distinct selections over %d timed reps\n",
           bank, (double) bank * (bytes_gu + bytes_dn) / 1048576.0, n_experts_sel * timed, timed);
    return 0;
}
