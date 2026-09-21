// alice_qgate.cpp — quality-gate harness for the Q3.9-MIX decision.
//
// Runs one prompt through one model with the SAME offload split that the
// deployment uses, and dumps the comparison material the gate needs:
//
//   <prefix>.logits.f32   full float32 logits of the last position (n_vocab)
//   <prefix>.topk.bin     per-position top-K (int32 ids + float32 logits)
//   <prefix>.meta.json    geometry, timing, top-10 of the last position
//
// Two runs (baseline vs candidate) are compared by tools/alice_qgate_cmp.py:
// KL divergence both directions, top-1/5/10 agreement, and rank correlation.
// Route-level comparison is available for free: export ALICE_MOE_STATS +
// ALICE_PROBE_TOPK and the shared graph probe records the router mass of the
// same forward pass.
//
// build:
//   g++ -O3 -std=c++17 -I include -I ggml/include -I common -I . \
//       tools/alice_qgate.cpp -L build-vk/bin -lllama-common -lllama -lggml \
//       -lggml-base -Wl,-rpath,$PWD/build-vk/bin -o build-vk/bin/alice_qgate
//
// run:
//   ./build-vk/bin/alice_qgate model.gguf prompt.txt out --ngl 99 --ncmoe 32

#include "llama.h"
#include "common.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

static std::string read_file(const char * path) {
    FILE * f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "cannot open %s\n", path); exit(1); }
    std::string s;
    char buf[65536];
    size_t n;
    while ((n = fread(buf, 1, sizeof(buf), f)) > 0) s.append(buf, n);
    fclose(f);
    return s;
}

int main(int argc, char ** argv) {
    if (argc < 4) {
        fprintf(stderr, "usage: %s <model.gguf> <out_prefix_dir> --prompt FILE [--prompt FILE ...] "
                        "[--ngl N] [--ncmoe N] [--ctx N] [--threads N] [--topk K] [--max-pos N]\n"
                        "       one model load, one fresh context per prompt; outputs per prompt:\n"
                        "       <dir>/<prompt-stem>.logits.f32 / .topk.bin / .meta.json\n", argv[0]);
        return 2;
    }
    const char * model_path = argv[1];
    const char * out_dir = argv[2];

    int ngl = 99, ncmoe = 0, n_ctx = 4096, n_threads = 8, topk = 10, max_pos = 0;
    std::vector<std::string> prompts;
    for (int i = 3; i < argc; ++i) {
        if      (!strcmp(argv[i], "--prompt") && i + 1 < argc) prompts.push_back(argv[++i]);
        else if (!strcmp(argv[i], "--ngl") && i + 1 < argc)     ngl = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--ncmoe") && i + 1 < argc)   ncmoe = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--ctx") && i + 1 < argc)     n_ctx = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--threads") && i + 1 < argc) n_threads = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--topk") && i + 1 < argc)    topk = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--max-pos") && i + 1 < argc) max_pos = atoi(argv[++i]);
        else { fprintf(stderr, "unknown arg %s\n", argv[i]); return 2; }
    }
    if (prompts.empty()) { fprintf(stderr, "no --prompt given\n"); return 2; }

    ggml_backend_load_all();

    static std::vector<llama_model_tensor_buft_override> overrides;
    if (ncmoe > 0) {
        llm_add_n_cpu_ffn_overrides(ncmoe, LLM_FFN_EXPS_REGEX, overrides);
    }
    overrides.push_back({ nullptr, nullptr });

    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = ngl;
    mparams.tensor_buft_overrides = overrides.data();

    llama_model * model = llama_model_load_from_file(model_path, mparams);
    if (!model) { fprintf(stderr, "load failed\n"); return 1; }

    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int n_vocab = llama_vocab_n_tokens(vocab);
    const int n_batch = 2048;

    int rc_total = 0;
    for (const std::string & pf : prompts) {
        std::string stem = pf;
        {
            size_t slash = stem.find_last_of('/');
            if (slash != std::string::npos) stem = stem.substr(slash + 1);
            size_t dot = stem.find_last_of('.');
            if (dot != std::string::npos) stem = stem.substr(0, dot);
        }
        char prefix[4096];
        snprintf(prefix, sizeof(prefix), "%s/%s", out_dir, stem.c_str());

        llama_context_params cparams = llama_context_default_params();
        cparams.n_ctx     = (uint32_t) n_ctx;
        cparams.n_batch   = (uint32_t) n_batch;
        cparams.n_ubatch  = 512;
        cparams.n_seq_max = 1;
        cparams.n_threads = n_threads;
        cparams.n_threads_batch = n_threads;
        llama_context * ctx = llama_init_from_model(model, cparams);
        if (!ctx) { fprintf(stderr, "[%s] ctx failed\n", stem.c_str()); rc_total = 1; continue; }

        const std::string text = read_file(pf.c_str());
        std::vector<llama_token> toks(text.size() + 8);
        int n_tok = llama_tokenize(vocab, text.data(), (int) text.size(), toks.data(), (int) toks.size(), true, true);
        if (n_tok < 0) { toks.resize(-n_tok); n_tok = llama_tokenize(vocab, text.data(), (int) text.size(), toks.data(), (int) toks.size(), true, true); }
        if (n_tok <= 0) { fprintf(stderr, "[%s] tokenize failed\n", stem.c_str()); llama_free(ctx); rc_total = 1; continue; }
        toks.resize(n_tok);
        if (n_tok > n_ctx - 8) {
            fprintf(stderr, "[%s] prompt %d tokens does not fit ctx %d\n", stem.c_str(), n_tok, n_ctx);
            llama_free(ctx);
            rc_total = 1;
            continue;
        }

        std::vector<int32_t> topk_ids;
        std::vector<float>   topk_vals;
        std::vector<int32_t> argmax_ids;
        std::vector<float>   last_logits;
        double nll_sum = 0.0;          // teacher-forced negative log-likelihood of the prompt
        int64_t nll_count = 0;
        int64_t top1_hits = 0;

        const auto t0 = std::chrono::steady_clock::now();
        int pos0 = 0;
        bool failed = false;
        while (pos0 < n_tok) {
            const int n = std::min(n_batch, n_tok - pos0);
            llama_batch batch = llama_batch_init(n, 0, 1);
            for (int i = 0; i < n; ++i) {
                batch.token[i] = toks[pos0 + i];
                batch.pos[i]   = pos0 + i;
                batch.n_seq_id[i] = 1;
                batch.seq_id[i][0] = 0;
                batch.logits[i] = 1;
            }
            batch.n_tokens = n;
            if (llama_decode(ctx, batch) != 0) { fprintf(stderr, "[%s] decode failed at %d\n", stem.c_str(), pos0); llama_batch_free(batch); failed = true; break; }

            std::vector<int> idx(n_vocab);
            for (int p = 0; p < n; ++p) {
                const float * logits = llama_get_logits_ith(ctx, p);
                if (!logits) { continue; }
                const int gpos = pos0 + p;

                // teacher-forced likelihood of the ACTUAL next token (quality metric that
                // does not depend on any divergence reference)
                if (gpos + 1 < n_tok) {
                    const int next = (int) toks[gpos + 1];
                    float m = -INFINITY;
                    for (int i = 0; i < n_vocab; ++i) {
                        if (logits[i] > m) { m = logits[i]; }
                    }
                    double z = 0.0;
                    for (int i = 0; i < n_vocab; ++i) {
                        z += std::exp((double) logits[i] - (double) m);
                    }
                    const double logp = (double) logits[next] - (double) m - std::log(z);
                    nll_sum += -logp;
                    nll_count += 1;
                    // top-1 accuracy at the same position
                    int am = 0;
                    float amv = logits[0];
                    for (int i = 1; i < n_vocab; ++i) {
                        if (logits[i] > amv) { amv = logits[i]; am = i; }
                    }
                    if (am == next) { top1_hits += 1; }
                }

                if (max_pos > 0 && gpos < n_tok - max_pos) {
                    if (gpos == n_tok - 1) last_logits.assign(logits, logits + n_vocab);
                    continue;
                }
                for (int i = 0; i < n_vocab; ++i) idx[i] = i;
                std::partial_sort(idx.begin(), idx.begin() + topk, idx.end(),
                                  [&](int a, int b) { return logits[a] > logits[b]; });
                for (int k = 0; k < topk; ++k) {
                    topk_ids.push_back(idx[k]);
                    topk_vals.push_back(logits[idx[k]]);
                }
                argmax_ids.push_back(idx[0]);
                if (gpos == n_tok - 1) last_logits.assign(logits, logits + n_vocab);
            }
            llama_batch_free(batch);
            pos0 += n;
        }
        const auto t1 = std::chrono::steady_clock::now();
        const double pp_s = std::chrono::duration<double>(t1 - t0).count();

        if (failed || last_logits.empty()) { fprintf(stderr, "[%s] no logits captured\n", stem.c_str()); llama_free(ctx); rc_total = 1; continue; }

        char path[4096];
        snprintf(path, sizeof(path), "%s.logits.f32", prefix);
        FILE * f = fopen(path, "wb");
        fwrite(last_logits.data(), sizeof(float), last_logits.size(), f);
        fclose(f);

        snprintf(path, sizeof(path), "%s.topk.bin", prefix);
        f = fopen(path, "wb");
        const int32_t n_pos = (int32_t) argmax_ids.size();
        fwrite(&n_pos, sizeof(n_pos), 1, f);
        fwrite(&topk, sizeof(topk), 1, f);
        fwrite(topk_ids.data(), sizeof(int32_t), topk_ids.size(), f);
        fwrite(topk_vals.data(), sizeof(float), topk_vals.size(), f);
        fclose(f);

        snprintf(path, sizeof(path), "%s.meta.json", prefix);
        f = fopen(path, "w");
        fprintf(f, "{\n \"model\": \"%s\",\n \"prompt\": \"%s\",\n \"n_vocab\": %d,\n \"n_tokens\": %d,\n",
                model_path, pf.c_str(), n_vocab, n_tok);
        fprintf(f, " \"ngl\": %d, \"ncmoe\": %d, \"ctx\": %d, \"threads\": %d, \"topk\": %d,\n",
                ngl, ncmoe, n_ctx, n_threads, topk);
        fprintf(f, " \"prefill_s\": %.3f,\n \"prefill_tps\": %.3f,\n", pp_s, n_tok / pp_s);
        fprintf(f, " \"positions_dumped\": %d,\n", n_pos);
        fprintf(f, " \"nll_mean\": %s,\n", nll_count ? std::to_string(nll_sum / nll_count).c_str() : "null");
        fprintf(f, " \"ppl\": %s,\n", nll_count ? std::to_string(std::exp(nll_sum / nll_count)).c_str() : "null");
        fprintf(f, " \"top1_accuracy\": %s,\n", nll_count ? std::to_string((double) top1_hits / nll_count).c_str() : "null");
        fprintf(f, " \"nll_positions\": %lld,\n", (long long) nll_count);
        fprintf(f, " \"last_top10\": [");
        for (int k = 0; k < std::min(topk, 10); ++k) {
            fprintf(f, "%s[%d, %.6f]", k ? ", " : "", topk_ids[(n_pos - 1) * topk + k], topk_vals[(n_pos - 1) * topk + k]);
        }
        fprintf(f, "],\n \"last_argmax\": %d\n}\n", argmax_ids[n_pos - 1]);
        fclose(f);

        printf("[%s] %d tokens, %.2f t/s prefill, positions %d, argmax(last) %d\n",
               stem.c_str(), n_tok, n_tok / pp_s, n_pos, argmax_ids[n_pos - 1]);
        fflush(stdout);
        llama_free(ctx);
    }

    llama_model_free(model);
    return rc_total;
}
