// mtp_graph_dump.cpp -- full-width fp32 dump of the Qwen3.8 MTP draft graph.
//
// Gate 2 of the MTP tranche: produce one native dump of the draft block's stage
// tensors for an exactly specified, deterministic input so it can be compared
// stage by stage against the pure-numpy reference (mtp_min.mtp_forward).
//
// Why this is a standalone tool and not LLAMA_DUMP_VALUES:
//   LLAMA_DUMP_VALUES prints at most 64 floats per tensor (src/llama-context.cpp),
//   which cannot support a conformance comparison.  This tool installs a
//   llama_context_params.cb_eval callback instead, which is public API and is
//   honoured in src/llama-context.cpp for the graph it builds, and writes every
//   element of the named tensors as little-endian float32.
//
// Nothing in src/ or common/ is modified; the tool only uses public headers.
//
// Build (documented in MTP_GRAPH_CONFORMANCE_REPORT.md):
//   g++ -std=c++17 -O2 -I include -I ggml/include tests/mtp_graph_dump.cpp \
//       -L build-i3/bin -lllama -lggml -lggml-base \
//       -Wl,-rpath,$PWD/build-i3/bin -o /tmp/mtp_graph_dump
//
// Run:
//   /tmp/mtp_graph_dump --model /tmp/qwen4exp-mtp-L4.gguf \
//       --inputs-dir <dir> --out-dir <dir> [--threads 4] [--steps 5]

#include "llama.h"

#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <set>
#include <algorithm>
#include <fstream>
#include <sstream>

// Stage tensors we need values for.  Names are matched with the "-<il>" suffix
// that llama_context::graph_get_cb() appends when il >= 0, and exactly otherwise.
static const char * kTargets[] = {
    "mtp_tok_embd",              // token embedding row
    "mtp_e",                     // embedding residual  (fc_embedding(rms_norm(embed)))
    "mtp_fc_hidden",             // fc_hidden(rms_norm(h)) before the residual add
    "mtp_fused",                 // fused h + e                                  <-- stage 1
    "mtp_attn_out",              // attention block output                       <-- stage 2
    "mtp_ffn_out",               // MoE block output                             <-- stage 3
    "mtp_multi",                 // pre-mixer 4-stream residual                  <-- stage 4
    "h_nextn",                   // the same tensor, published under its chain name
    "result_norm",               // collapsed sample hidden                      <-- stage 5
    "result_output",             // draft logits                                 <-- stage 6

    // extra observation points, used to localise a divergence
    "Qcur_normed", "Kcur_normed", "gate_reshaped", "attn_pregate", "attn_gated",
    "attn_output", "gate_sigmoid",
    "indexer_k_raw", "indexer_k_pooled", "indexer_k", "indexer_q",
    "indexer_score", "indexer_score_tokens", "indexer_top_k",
    "ffn_moe_probs", "ffn_moe_topk", "ffn_moe_weights", "ffn_moe_weights_sum",
    "ffn_moe_weights_norm", "ffn_moe_weighted", "ffn_moe_gate_up", "ffn_moe_gate",
    "ffn_moe_up", "ffn_moe_silu", "ffn_moe_swiglu", "ffn_moe_down", "ffn_moe_out",
    "shared_expert_gate_sigmoid", "ffn_shexp", "ffn_shexp_gated", "ffn_out",
    "hc_norm", "hc_gate", "hc_mixed", "hc_inject", "hc_combine",
};

struct dumper {
    FILE * bin  = nullptr;
    FILE * jsonl = nullptr;
    std::set<std::string> seen;         // every node name the graph executed
    std::set<std::string> dumped;       // target names actually dumped
    long long offset = 0;
    int step = 0;
    int dumped_records = 0;
};

static bool name_matches(const char * name, const char * base) {
    const size_t n = strlen(base);
    if (strncmp(name, base, n) != 0) {
        return false;
    }
    return name[n] == '\0' || name[n] == '-';
}

static bool is_target(const char * name) {
    for (const char * t : kTargets) {
        if (name_matches(name, t)) {
            return true;
        }
    }
    return false;
}

static const char * type_name(enum ggml_type t) {
    switch (t) {
        case GGML_TYPE_F32: return "f32";
        case GGML_TYPE_F16: return "f16";
        case GGML_TYPE_I32: return "i32";
        default:            return ggml_type_name(t);
    }
}

static bool dump_cb(struct ggml_tensor * t, bool ask, void * user_data) {
    dumper * d = (dumper *) user_data;

    if (ask) {
        d->seen.insert(std::string(t->name));
        return is_target(t->name);
    }

    if (!ggml_is_contiguous(t)) {
        fprintf(stderr, "WARN: %s is not contiguous; dumping in ggml memory order\n", t->name);
    }

    const int64_t n = ggml_nelements(t);
    std::vector<float> buf;
    const char * dtype = type_name(t->type);
    if (t->type == GGML_TYPE_F32) {
        buf.resize((size_t) n);
        ggml_backend_tensor_get(t, buf.data(), 0, (size_t) n * sizeof(float));
    } else if (t->type == GGML_TYPE_F16) {
        std::vector<ggml_fp16_t> half((size_t) n);
        ggml_backend_tensor_get(t, half.data(), 0, (size_t) n * sizeof(ggml_fp16_t));
        buf.resize((size_t) n);
        for (int64_t i = 0; i < n; ++i) {
            buf[(size_t) i] = ggml_fp16_to_fp32(half[(size_t) i]);
        }
    } else if (t->type == GGML_TYPE_I32) {
        // integer stages (router selection, indexer top-k) are written as float32
        // so the dump stays one homogeneous stream; the JSONL keeps dtype "i32"
        std::vector<int32_t> ival((size_t) n);
        ggml_backend_tensor_get(t, ival.data(), 0, (size_t) n * sizeof(int32_t));
        buf.resize((size_t) n);
        for (int64_t i = 0; i < n; ++i) {
            buf[(size_t) i] = (float) ival[(size_t) i];
        }
    } else {
        // any other type: record its presence but no values
        fprintf(d->jsonl,
                "{\"step\":%d,\"name\":\"%s\",\"dtype\":\"%s\",\"ne\":[%lld,%lld,%lld,%lld],"
                "\"offset\":null,\"nbytes\":0,\"nelements\":%lld}\n",
                d->step, t->name, dtype,
                (long long) t->ne[0], (long long) t->ne[1], (long long) t->ne[2], (long long) t->ne[3],
                (long long) n);
        d->dumped.insert(std::string(t->name));
        d->dumped_records++;
        return true;
    }

    fprintf(d->jsonl,
            "{\"step\":%d,\"name\":\"%s\",\"dtype\":\"%s\",\"ne\":[%lld,%lld,%lld,%lld],"
            "\"offset\":%lld,\"nbytes\":%lld,\"nelements\":%lld}\n",
            d->step, t->name, dtype,
            (long long) t->ne[0], (long long) t->ne[1], (long long) t->ne[2], (long long) t->ne[3],
            d->offset, (long long) n * 4, (long long) n);
    fwrite(buf.data(), sizeof(float), (size_t) n, d->bin);
    d->offset += (long long) n * 4;
    d->dumped.insert(std::string(t->name));
    d->dumped_records++;
    return true;
}

// --- tiny input format -------------------------------------------------------
// <inputs-dir>/mtp_inputs.txt:
//     n_steps <int>
//     hc_width <int>
// <inputs-dir>/mtp_inputs_h.f32     : n_steps * hc_width little-endian float32, row major
// <inputs-dir>/mtp_inputs_tokens.i32: n_steps little-endian int32

static bool read_file(const std::string & path, std::vector<char> & out) {
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        return false;
    }
    f.seekg(0, std::ios::end);
    const std::streamoff n = f.tellg();
    f.seekg(0, std::ios::beg);
    out.resize((size_t) n);
    if (n > 0) {
        f.read(out.data(), n);
    }
    return true;
}

// FNV-1a over the exact input bytes this process read: the compare step recomputes
// it from the same files, so "both sides consumed identical inputs" is checked,
// not assumed. (No SHA-256 here to keep the tool dependency-free.)
static uint64_t fnv1a1(const void * data, size_t n) {
    const uint8_t * b = (const uint8_t *) data;
    uint64_t h = 14695981039346656037ULL;
    for (size_t i = 0; i < n; ++i) {
        h ^= (uint64_t) b[i];
        h *= 1099511628211ULL;
    }
    return h;
}

int main(int argc, char ** argv) {
    std::string model_path;
    std::string inputs_dir;
    std::string out_dir;
    int  threads    = 4;
    bool disable_fa = true;

    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&](const char * what) -> std::string {
            if (i + 1 >= argc) { fprintf(stderr, "missing value for %s\n", what); exit(2); }
            return argv[++i];
        };
        if      (a == "--model")      { model_path = next("--model"); }
        else if (a == "--inputs-dir") { inputs_dir = next("--inputs-dir"); }
        else if (a == "--out-dir")    { out_dir    = next("--out-dir"); }
        else if (a == "--threads")    { threads    = atoi(next("--threads").c_str()); }
        else if (a == "--flash")      { disable_fa = false; }
        else { fprintf(stderr, "unknown argument: %s\n", a.c_str()); return 2; }
    }
    if (model_path.empty() || inputs_dir.empty() || out_dir.empty()) {
        fprintf(stderr, "usage: %s --model M.gguf --inputs-dir D --out-dir D [--threads N] [--flash]\n", argv[0]);
        return 2;
    }

    // ---- inputs ----
    int n_steps = 0, hc_width = 0;
    {
        std::ifstream f(inputs_dir + "/mtp_inputs.txt");
        if (!f) { fprintf(stderr, "cannot open %s/mtp_inputs.txt\n", inputs_dir.c_str()); return 1; }
        std::string key; long long v;
        while (f >> key >> v) {
            if      (key == "n_steps")  { n_steps  = (int) v; }
            else if (key == "hc_width") { hc_width = (int) v; }
        }
    }
    if (n_steps <= 0 || hc_width <= 0) { fprintf(stderr, "bad inputs header\n"); return 1; }

    std::vector<char> h_raw, tok_raw;
    if (!read_file(inputs_dir + "/mtp_inputs_h.f32", h_raw) ||
        (long long) h_raw.size() != (long long) n_steps * hc_width * 4) {
        fprintf(stderr, "mtp_inputs_h.f32 missing or wrong size\n");
        return 1;
    }
    if (!read_file(inputs_dir + "/mtp_inputs_tokens.i32", tok_raw) ||
        (long long) tok_raw.size() != (long long) n_steps * 4) {
        fprintf(stderr, "mtp_inputs_tokens.i32 missing or wrong size\n");
        return 1;
    }
    const float *   h      = (const float *)   h_raw.data();
    const int32_t * tokens = (const int32_t *) tok_raw.data();

    // proof of input identity: hashes of the bytes this process actually read
    {
        FILE * f = fopen((out_dir + "/native_inputs_hash.json").c_str(), "w");
        if (f) {
            fprintf(f,
                    "{\"n_steps\": %d, \"hc_width\": %d,\n"
                    " \"hidden_bytes\": %lld, \"hidden_fnv1a1\": \"%016llx\",\n"
                    " \"tokens_bytes\": %lld, \"tokens_fnv1a1\": \"%016llx\"}\n",
                    n_steps, hc_width,
                    (long long) h_raw.size(), (unsigned long long) fnv1a1(h_raw.data(), h_raw.size()),
                    (long long) tok_raw.size(), (unsigned long long) fnv1a1(tok_raw.data(), tok_raw.size()));
            fclose(f);
        }
    }

    // ---- model + MTP context ----
    llama_backend_init();

    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = 0;
    mparams.load_mtp     = true;    // otherwise has_mtp == false and the MTP graph asserts

    llama_model * model = llama_model_load_from_file(model_path.c_str(), mparams);
    if (model == nullptr) { fprintf(stderr, "failed to load model\n"); return 1; }

    printf("model: n_layer=%u n_layer_nextn=%u n_embd=%u n_embd_out=%u\n",
           (unsigned) llama_model_n_layer(model),
           (unsigned) llama_model_n_layer_nextn(model),
           (unsigned) llama_model_n_embd(model),
           (unsigned) llama_model_n_embd_out(model));

    dumper d;
    const std::string bin_path   = out_dir + "/native_dump.bin";
    const std::string jsonl_path = out_dir + "/native_dump.jsonl";
    d.bin   = fopen(bin_path.c_str(), "wb");
    d.jsonl = fopen(jsonl_path.c_str(), "w");
    if (!d.bin || !d.jsonl) { fprintf(stderr, "cannot open dump outputs in %s\n", out_dir.c_str()); return 1; }

    llama_context_params cparams = llama_context_default_params();
    cparams.ctx_type        = LLAMA_CONTEXT_TYPE_MTP;
    cparams.n_ctx           = 512;
    cparams.n_batch         = 8;
    cparams.n_ubatch        = 8;
    cparams.n_seq_max       = 1;
    cparams.n_rs_seq        = 4;             // unused (no recurrent layer in the draft filters)
    cparams.n_threads       = threads;
    cparams.n_threads_batch = threads;
    cparams.type_k          = GGML_TYPE_F32; // no cache rounding on either side
    cparams.type_v          = GGML_TYPE_F32;
    cparams.embeddings      = false;
    if (disable_fa) {
        cparams.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;   // match the reference's explicit softmax
    }
    cparams.cb_eval           = dump_cb;
    cparams.cb_eval_user_data = &d;

    llama_context * ctx = llama_init_from_model(model, cparams);
    if (ctx == nullptr) { fprintf(stderr, "failed to create MTP context\n"); return 1; }
    printf("flash_attn=%s type_k=%s type_v=%s\n",
           disable_fa ? "disabled" : "enabled(per model)",
           ggml_type_name(cparams.type_k), ggml_type_name(cparams.type_v));

    llama_batch batch = llama_batch_init(1, hc_width, 1);
    batch.token = (llama_token *) malloc(sizeof(llama_token));   // MTP needs token AND embd

    FILE * api_logits = fopen((out_dir + "/native_logits_api.f32").c_str(), "wb");

    int rc = 0;
    for (int s = 0; s < n_steps; ++s) {
        batch.n_tokens    = 1;
        batch.token[0]    = (llama_token) tokens[s];
        batch.pos[0]      = (llama_pos) s;
        batch.n_seq_id[0] = 1;
        batch.seq_id[0][0] = 0;
        batch.logits[0]   = 1;
        memcpy(batch.embd, h + (size_t) s * hc_width, (size_t) hc_width * sizeof(float));

        d.step = s;
        const int ret = llama_decode(ctx, batch);
        if (ret != 0) {
            fprintf(stderr, "llama_decode failed at step %d: %d\n", s, ret);
            rc = 1;
            break;
        }

        const float * lg = llama_get_logits_ith(ctx, 0);
        if (lg != nullptr && api_logits != nullptr) {
            fwrite(lg, sizeof(float), (size_t) llama_vocab_n_tokens(llama_model_get_vocab(model)), api_logits);
        }
        fflush(d.jsonl);
        fflush(d.bin);
        printf("step %d decoded (token=%d pos=%d)\n", s, (int) tokens[s], s);
    }

    fclose(d.bin);
    fclose(d.jsonl);
    if (api_logits) fclose(api_logits);

    // graph inventory: the evidence for which attention path the draft block took
    {
        FILE * f = fopen((out_dir + "/native_graph_nodes.json").c_str(), "w");
        fprintf(f, "{\n  \"nodes\": [\n");
        bool first = true;
        for (const auto & n : d.seen) {
            fprintf(f, "%s    \"%s\"", first ? "" : ",\n", n.c_str());
            first = false;
        }
        fprintf(f, "\n  ],\n  \"n_nodes\": %zu,\n  \"dumped\": [\n", d.seen.size());
        first = true;
        for (const auto & n : d.dumped) {
            fprintf(f, "%s    \"%s\"", first ? "" : ",\n", n.c_str());
            first = false;
        }
        fprintf(f, "\n  ],\n  \"n_dumped\": %zu,\n  \"records\": %d,\n  \"n_steps\": %d\n}\n",
                d.dumped.size(), d.dumped_records, n_steps);
        fclose(f);
    }

    printf("dumped %d records across %d steps; %zu distinct graph node names\n",
           d.dumped_records, n_steps, d.seen.size());

    llama_batch_free(batch);   // also frees the malloc'd batch.token above
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return rc;
}
