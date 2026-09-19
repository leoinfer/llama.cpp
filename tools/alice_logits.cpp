// alice_logits.cpp — dump top-k logits from a GGUF via the llama.cpp API.
// The differential harness compares these against the PyTorch reference.
//
// build:  g++ -O2 -std=c++17 -I include -I ggml/include alice_logits.cpp \
//              -L build-cpu/bin -lllama -lggml -lggml-base -Wl,-rpath,$PWD/build-cpu/bin
// run:    ./alice_logits <model.gguf> <tok1> <tok2> ...

#include "llama.h"

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>
#include <algorithm>
#include <cmath>

int main(int argc, char ** argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: %s <model.gguf> <tok...>\n", argv[0]);
        return 2;
    }

    ggml_backend_load_all();

    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = 0;
    llama_model * model = llama_model_load_from_file(argv[1], mparams);
    if (!model) { fprintf(stderr, "load failed\n"); return 1; }

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx     = 512;
    cparams.n_batch   = 512;
    cparams.n_seq_max = 1;
    cparams.embeddings = getenv("ALICE_EMBD") != nullptr;
    llama_context * ctx = llama_init_from_model(model, cparams);
    if (!ctx) { fprintf(stderr, "ctx failed\n"); return 1; }

    std::vector<llama_token> toks;
    for (int i = 2; i < argc; ++i) {
        toks.push_back((llama_token) atoi(argv[i]));
    }
    printf("prompt tokens:");
    for (auto t : toks) printf(" %d", t);
    printf("\n");

    llama_batch batch = llama_batch_get_one(toks.data(), (int32_t) toks.size());
    if (llama_decode(ctx, batch) != 0) { fprintf(stderr, "decode failed\n"); return 1; }

    const float * logits = llama_get_logits_ith(ctx, -1);
    const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));

    std::vector<int> idx(n_vocab);
    for (int i = 0; i < n_vocab; ++i) idx[i] = i;
    std::partial_sort(idx.begin(), idx.begin() + 10, idx.end(),
        [&](int a, int b) { return logits[a] > logits[b]; });

    if (const float * emb = llama_get_embeddings_ith(ctx, -1)) {
        const int64_t ne = llama_model_n_embd(model);
        printf("embd[0..7]:");
        for (int i = 0; i < 8 && i < ne; ++i) printf(" %.6f", emb[i]);
        printf("\n");
        double ss = 0; int nn = 0;
        for (int64_t i = 0; i < ne; ++i) { ss += (double) emb[i] * emb[i]; nn++; }
        printf("embd norm: %.6f  (n=%d)\n", sqrt(ss), nn);
    }

    printf("vocab: %d\n", n_vocab);
    printf("top-10:\n");
    for (int k = 0; k < 10; ++k) {
        printf("   %7d  %+.4f\n", idx[k], logits[idx[k]]);
    }
    for (int i = 2; i < argc; ++i) {
        int t = atoi(argv[i]);
        printf("probe tok %d logit %+.4f\n", t, logits[t]);
    }

    llama_free(ctx);
    llama_model_free(model);
    return 0;
}
