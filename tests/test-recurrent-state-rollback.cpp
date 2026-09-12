#include "arg.h"
#include "common.h"
#include "ggml-backend.h"
#include "llama.h"

#include "../src/llama-io.h"
#include "../src/llama-memory.h"
#include "../src/llama-memory-hybrid.h"
#include "../src/llama-memory-recurrent.h"

#include <algorithm>
#include <clocale>
#include <cmath>
#include <cstdio>
#include <limits>
#include <set>
#include <vector>

static bool decode_tokens(llama_context * ctx, const std::vector<llama_token> & tokens, uint32_t count) {
    llama_batch batch = llama_batch_init(count, 0, 1);
    for (uint32_t pos = 0; pos < count; ++pos) {
        common_batch_add(batch, tokens[pos], pos, { 0 }, pos + 1 == count);
    }
    const bool ok = llama_decode(ctx, batch) == 0;
    llama_batch_free(batch);
    return ok;
}

static bool decode_one(llama_context * ctx, llama_token tok, llama_pos pos) {
    llama_batch batch = llama_batch_init(1, 0, 1);
    common_batch_add(batch, tok, pos, { 0 }, true);
    const bool ok = llama_decode(ctx, batch) == 0;
    llama_batch_free(batch);
    return ok;
}

struct cache_buffer_collector : llama_io_write_i {
    std::set<ggml_backend_buffer_t> buffers;
    size_t size = 0;

    void write(const void *, size_t n) override {
        size += n;
    }

    void write_tensor(ggml_tensor * tensor, size_t, size_t n) override {
        buffers.insert(tensor->buffer);
        size += n;
    }

    size_t n_bytes() override {
        return size;
    }
};

static llama_context * init_ctx(llama_model * model, llama_context_params cparams, uint8_t fill) {
    llama_context * ctx = llama_init_from_model(model, cparams);
    if (ctx == nullptr || fill == 0) {
        return ctx;
    }

    // Use a full ubatch so buffer discovery preserves prefill allocation sizes.
    const uint32_t n_tokens = llama_n_ubatch(ctx);
    if (!decode_tokens(ctx, std::vector<llama_token>(n_tokens, 0), n_tokens)) {
        llama_free(ctx);
        return nullptr;
    }
    llama_synchronize(ctx);
    cache_buffer_collector collector;
    llama_get_memory(ctx)->state_write(collector);
    llama_memory_clear(llama_get_memory(ctx), true);
    if (collector.buffers.empty()) {
        fprintf(stderr, "%s : no cache buffers found\n", __func__);
        llama_free(ctx);
        return nullptr;
    }
    for (auto * buffer : collector.buffers) {
        ggml_backend_buffer_clear(buffer, fill);
    }
    return ctx;
}

static llama_context * make_ctx(const common_params & params, llama_model * model, uint8_t fill) {
    auto cparams = common_context_params_to_llama(params);
    cparams.n_seq_max = 1;
    cparams.n_rs_seq  = 8;
    cparams.n_batch   = std::max(cparams.n_batch,  (uint32_t) (cparams.n_rs_seq + 1));
    cparams.n_ubatch  = std::max(cparams.n_ubatch, (uint32_t) (cparams.n_rs_seq + 1));
    return init_ctx(model, cparams, fill);
}

static float logit_diff(float a, float b) {
    return std::isfinite(a) && std::isfinite(b) ? std::fabs(a - b) : std::numeric_limits<float>::infinity();
}

// Roll back multiple sequences, then replay them in a single batch whose
// per-seq token count exceeds n_ubatch: each seq's replay spans several
// ubatches while its rollback restore is still pending. Compared against a
// reference context that never advanced past the rollback point and decodes
// the identical replay batch.
static bool test_multi_seq_split_replay(const common_params & params, llama_model * model, const int n_vocab, uint8_t fill) {
    constexpr uint32_t  n_seqs     = 2;
    constexpr uint32_t  n_ubatch   = 16;
    constexpr uint32_t  n_prompt   = 19;
    constexpr uint32_t  n_rollback = 3;
    constexpr uint32_t  n_replay   = 40; // > n_ubatch so each seq spans multiple ubatches
    constexpr llama_pos p0         = n_prompt - n_rollback;

    const auto make_ctx_multi = [&]() {
        auto cparams = common_context_params_to_llama(params);
        cparams.n_seq_max  = n_seqs;
        cparams.n_rs_seq   = 8;
        cparams.n_ctx      = 256;
        cparams.n_batch    = 256;
        cparams.n_ubatch   = n_ubatch;
        cparams.kv_unified = false;
        return init_ctx(model, cparams, fill);
    };

    llama_context * ctx_roll = make_ctx_multi();
    llama_context * ctx_ref  = make_ctx_multi();
    if (ctx_roll == nullptr || ctx_ref == nullptr) {
        fprintf(stderr, "%s : failed to init multi-seq contexts\n", __func__);
        return false;
    }

    const auto cleanup = [&]() {
        llama_free(ctx_roll);
        llama_free(ctx_ref);
    };

    if (llama_n_rs_seq(ctx_roll) < n_rollback) {
        fprintf(stderr, "%s : skipping because n_rs_seq is too small\n", __func__);
        cleanup();
        return true;
    }

    const auto tok = [&](uint32_t seq, llama_pos pos) {
        return (llama_token) ((7*(uint32_t) pos + 31*seq + 1) % (uint32_t) n_vocab);
    };

    bool ok = true;

    // both contexts decode the identical [0, p0) prefill; only ctx_roll decodes
    // the tail, which is then rolled back so its restore is pending at replay
    for (uint32_t s = 0; s < n_seqs && ok; ++s) {
        llama_batch batch = llama_batch_init(n_prompt, 0, 1);
        for (llama_pos pos = 0; pos < (llama_pos) p0; ++pos) {
            common_batch_add(batch, tok(s, pos), pos, { (llama_seq_id) s }, false);
        }
        ok = ok && llama_decode(ctx_roll, batch) == 0;
        ok = ok && llama_decode(ctx_ref,  batch) == 0;

        common_batch_clear(batch);
        for (llama_pos pos = p0; pos < (llama_pos) n_prompt; ++pos) {
            common_batch_add(batch, tok(s, pos), pos, { (llama_seq_id) s }, false);
        }
        ok = ok && llama_decode(ctx_roll, batch) == 0;
        llama_batch_free(batch);

        ok = ok && llama_memory_seq_rm(llama_get_memory(ctx_roll), (llama_seq_id) s, p0, -1);

        // a second partial removal while one is pending must be refused
        ok = ok && !llama_memory_seq_rm(llama_get_memory(ctx_roll), (llama_seq_id) s, p0 - 1, -1);
    }
    if (!ok) {
        fprintf(stderr, "%s : multi-seq prefill/rollback failed\n", __func__);
        cleanup();
        return false;
    }

    llama_batch batch = llama_batch_init(n_seqs*n_replay, 0, 1);
    for (uint32_t s = 0; s < n_seqs; ++s) {
        for (uint32_t i = 0; i < n_replay; ++i) {
            const llama_pos pos = p0 + (llama_pos) i;
            common_batch_add(batch, tok(s, pos), pos, { (llama_seq_id) s }, true);
        }
    }
    ok = llama_decode(ctx_roll, batch) == 0;
    ok = ok && llama_decode(ctx_ref, batch) == 0;
    llama_batch_free(batch);
    if (!ok) {
        fprintf(stderr, "%s : multi-seq replay decode failed\n", __func__);
        cleanup();
        return false;
    }

    // identical ubatch shapes from bit-exact states: a correct implementation
    // matches bitwise, so eps only allows backend scheduling noise
    constexpr float eps = 1e-7f;

    float    diff_max  = 0.0f;
    uint32_t seq_first = 0;
    int32_t  pos_first = -1;
    for (uint32_t i = 0; i < n_seqs*n_replay; ++i) {
        const float * l_roll = llama_get_logits_ith(ctx_roll, i);
        const float * l_ref  = llama_get_logits_ith(ctx_ref,  i);
        if (l_roll == nullptr || l_ref == nullptr) {
            fprintf(stderr, "%s : missing multi-seq logits at index %u\n", __func__, i);
            cleanup();
            return false;
        }
        for (int t = 0; t < n_vocab; ++t) {
            const float diff = logit_diff(l_roll[t], l_ref[t]);
            if (diff > eps && pos_first < 0) {
                seq_first = i/n_replay;
                pos_first = p0 + (int32_t) (i%n_replay);
            }
            diff_max = std::max(diff_max, diff);
        }
    }

    if (diff_max > eps) {
        fprintf(stderr, "%s : multi-seq split replay logits mismatch (max diff %g, first at seq %u pos %d)\n",
                __func__, (double) diff_max, seq_first, pos_first);
        cleanup();
        return false;
    }

    fprintf(stderr, "%s : multi-seq split replay matched (max diff %g)\n", __func__, (double) diff_max);

    // seq-1-only decodes must be independent of seq 0's content: diverge seq 0
    // in ctx_ref only, then compare identical seq-1-only continuations bitwise
    constexpr uint32_t n_tail = 4;

    {
        llama_batch batch_tail = llama_batch_init(n_tail, 0, 1);
        for (uint32_t i = 0; i < n_tail; ++i) {
            const llama_pos pos = p0 + (llama_pos) (n_replay + i);
            common_batch_add(batch_tail, tok(0, pos + 7), pos, { 0 }, false);
        }
        ok = llama_decode(ctx_ref, batch_tail) == 0;
        llama_batch_free(batch_tail);
    }

    float diff_tail = 0.0f;
    for (uint32_t i = 0; i < n_tail && ok; ++i) {
        const llama_pos pos = p0 + (llama_pos) (n_replay + i);
        llama_batch batch_one = llama_batch_init(1, 0, 1);
        common_batch_add(batch_one, tok(1, pos), pos, { 1 }, true);
        ok = llama_decode(ctx_roll, batch_one) == 0;
        ok = ok && llama_decode(ctx_ref, batch_one) == 0;
        llama_batch_free(batch_one);
        if (!ok) {
            break;
        }

        const float * l_roll = llama_get_logits_ith(ctx_roll, 0);
        const float * l_ref  = llama_get_logits_ith(ctx_ref,  0);
        ok = l_roll != nullptr && l_ref != nullptr;
        for (int t = 0; ok && t < n_vocab; ++t) {
            diff_tail = std::max(diff_tail, logit_diff(l_roll[t], l_ref[t]));
        }
    }

    if (!ok || diff_tail > eps) {
        fprintf(stderr, "%s : seq-1-only decode leaked seq 0 state (ok=%d, max diff %g)\n",
                __func__, ok ? 1 : 0, (double) diff_tail);
        cleanup();
        return false;
    }

    fprintf(stderr, "%s : seq-1-only decode independent of seq 0 (max diff %g)\n", __func__, (double) diff_tail);
    cleanup();
    return true;
}

//
// Recurrent rollback contract and MTP transaction parity
//
// The speculative path removes a rejected suffix with llama_memory_seq_rm(seq, p0, -1)
// after the target decoded a verify batch. These tests pin down what that call does to
// the cache, and - through the next-token logits - whether the state it leaves behind is
// the state a serial decode of the same committed tokens would have produced.
//

static llama_memory_recurrent * recr_of(llama_context * ctx) {
    llama_memory_t mem = llama_get_memory(ctx);
    if (auto * recr = dynamic_cast<llama_memory_recurrent *>(mem)) {
        return recr;
    }
    if (auto * hyb = dynamic_cast<llama_memory_hybrid *>(mem)) {
        return hyb->get_mem_recr();
    }
    return nullptr;
}

static llama_context * make_ctx_test(const common_params & params, llama_model * model,
        uint32_t n_rs, uint32_t n_ctx, llama_context_type ctx_type = LLAMA_CONTEXT_TYPE_DEFAULT) {
    auto cparams = common_context_params_to_llama(params);
    cparams.ctx_type  = ctx_type;
    cparams.n_seq_max = 1;
    cparams.n_rs_seq  = n_rs;
    cparams.n_ctx     = n_ctx;
    cparams.n_batch   = std::max<uint32_t>(cparams.n_batch,  32);
    cparams.n_ubatch  = std::max<uint32_t>(cparams.n_ubatch, 32);
    return llama_init_from_model(model, cparams);
}

// decode tokens[pos0 .. pos0+n) on seq 0, with per-row logits when asked
static bool decode_rows(llama_context * ctx, const llama_tokens & tokens, llama_pos pos0, int32_t n, bool logits) {
    llama_batch batch = llama_batch_init(n, 0, 1);
    for (int32_t i = 0; i < n; ++i) {
        common_batch_add(batch, tokens[(size_t) pos0 + i], pos0 + (llama_pos) i, { 0 }, logits);
    }
    const bool ok = llama_decode(ctx, batch) == 0;
    llama_batch_free(batch);
    return ok;
}

// the MTP draft graph consumes both a token (embedded) and the target's h_nextn row,
// so a draft-context batch carries token[] and embd[] at once
static bool decode_rows_embd(llama_context * ctx, llama_model * model, int32_t n_embd,
        const llama_tokens & tokens, llama_pos pos0, int32_t n, bool logits) {
    llama_batch batch = llama_batch_init(n, n_embd, 1);
    batch.token = (llama_token *) malloc(sizeof(llama_token) * n); // llama_batch_init allocates only one of the two

    batch.n_tokens = n;
    for (int32_t i = 0; i < n; ++i) {
        const llama_pos pos = pos0 + i;
        batch.token[i]     = tokens[(size_t) pos];
        batch.pos[i]       = pos;
        batch.n_seq_id[i]  = 1;
        batch.seq_id[i][0] = 0;
        batch.logits[i]    = logits ? 1 : 0;
        for (int32_t j = 0; j < n_embd; ++j) {
            batch.embd[(size_t) i * n_embd + j] = 0.05f * sinf(0.37f*(float) (pos + 1) + 0.11f*(float) j);
        }
    }

    const bool ok = llama_decode(ctx, batch) == 0;
    llama_batch_free(batch); // frees token[] as well
    return ok;
}

static bool gather_logits(llama_context * ctx, int32_t i_row, int n_vocab, std::vector<float> & out) {
    const float * logits = llama_get_logits_ith(ctx, i_row);
    if (logits == nullptr) {
        return false;
    }
    out.assign(logits, logits + n_vocab);
    return true;
}

static float max_logit_diff(const std::vector<float> & a, const std::vector<float> & b) {
    float res = 0.0f;
    for (size_t i = 0; i < a.size() && i < b.size(); ++i) {
        res = std::max(res, logit_diff(a[i], b[i]));
    }
    return res;
}

// Collects the state bytes so two contexts can be compared state-for-state.
struct byte_collector : llama_io_write_i {
    std::vector<uint8_t> bytes;

    void write(const void * src, size_t size) override {
        const uint8_t * p = (const uint8_t *) src;
        bytes.insert(bytes.end(), p, p + size);
    }

    void write_tensor(ggml_tensor * tensor, size_t offset, size_t size) override {
        const size_t cur = bytes.size();
        bytes.resize(cur + size);
        ggml_backend_tensor_get(tensor, bytes.data() + cur, offset, size);
    }

    size_t n_bytes() override {
        return bytes.size();
    }
};

// largest absolute difference between two state dumps read as f32 rows; the dumps are
// raw bytes, so only a layout-identical pair can be compared this way
static float max_state_diff(const byte_collector & a, const byte_collector & b) {
    if (a.bytes.size() != b.bytes.size() || a.bytes.size() % sizeof(float) != 0) {
        return std::numeric_limits<float>::infinity();
    }
    const float * fa = (const float *) a.bytes.data();
    const float * fb = (const float *) b.bytes.data();
    return max_logit_diff({ fa, fa + a.bytes.size() / sizeof(float) },
                          { fb, fb + b.bytes.size() / sizeof(float) });
}

// Return-value contract of llama_memory_recurrent::seq_rm on a cache that stores state:
// suffix removal, budget boundary, and the single-use pending rollback.
static bool test_seq_rm_contract(const common_params & params, llama_model * model, uint32_t n_rs) {
    const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));

    llama_context * ctx = make_ctx_test(params, model, n_rs, 256);
    if (ctx == nullptr) {
        fprintf(stderr, "%s : failed to init context\n", __func__);
        return false;
    }

    llama_memory_t mem = llama_get_memory(ctx);
    llama_memory_recurrent * recr = recr_of(ctx);
    if (recr == nullptr || !recr->stores_state) {
        fprintf(stderr, "%s : skipped for n_rs_seq=%u (no stateful recurrent cache)\n", __func__, n_rs);
        llama_free(ctx);
        return true;
    }

    const uint32_t N = n_rs + 4;
    llama_tokens tokens;
    for (uint32_t i = 0; i < N; ++i) {
        tokens.push_back((llama_token) ((7u*i + 31u) % (uint32_t) n_vocab));
    }

    bool ok = decode_tokens(ctx, tokens, N);
    const int32_t tail = ok ? recr->cells[0].tail : -1;

    // every step reports on its own, so a failure names the expectation that broke
    int n_fail = 0;
    const auto expect = [&](bool cond, const char * what) {
        if (!cond) {
            fprintf(stderr, "%s : n_rs_seq=%u FAILED '%s' (cell.pos=%d rs_idx=%u tail=%d)\n",
                    __func__, n_rs, what, (int) recr->cells[0].pos, recr->rs_idx[0], (int) recr->cells[0].tail);
            n_fail++;
        }
        ok = ok && cond;
    };

    expect(ok, "initial decode");
    expect(tail >= 0, "sequence has a tail cell");
    expect(tail >= 0 && recr->cells[tail].pos == (llama_pos) N - 1, "initial cell.pos == N-1");
    expect(recr->rs_idx[0] == 0, "initial rs_idx == 0");

    const auto rm = [&](llama_pos p0, llama_pos p1) {
        return llama_memory_seq_rm(mem, 0, p0, p1);
    };

    // past the tail: accepted, nothing to remove, no rollback staged
    expect(rm(N, -1), "remove nothing returns true");
    expect(recr->cells[tail].pos == (llama_pos) N - 1 && recr->rs_idx[0] == 0, "remove nothing changes nothing");

    // one token: stages exactly one snapshot and rewinds the tail by one
    expect(rm(N - 1, -1), "rollback 1 returns true");
    expect(recr->cells[tail].pos == (llama_pos) N - 2 && recr->rs_idx[0] == 1, "rollback 1 rewinds and stages");

    // the same p0 again does not enter the rollback branch, and must not disturb the pending one
    expect(rm(N - 1, -1), "repeat same p0 returns true");
    expect(recr->cells[tail].pos == (llama_pos) N - 2 && recr->rs_idx[0] == 1, "repeat same p0 is inert");

    // deeper while a rollback is pending: refused, nothing mutated
    expect(!rm(N - 2, -1), "deeper while pending is refused");
    expect(recr->cells[tail].pos == (llama_pos) N - 2 && recr->rs_idx[0] == 1, "refusal mutates nothing");

    // decoding the position the rollback removed consumes the pending rollback. It must
    // be strictly beyond the rewound position: M-RoPE init rejects a batch that starts at
    // the position the memory still holds.
    expect(decode_one(ctx, tokens[N - 1], N - 1), "decode at the removed position");
    expect(recr->cells[tail].pos == (llama_pos) N - 1 && recr->rs_idx[0] == 0, "consume clears rs_idx");

    // one beyond the budget: refused, nothing mutated
    expect(!rm((llama_pos) (N - 1) - n_rs, -1), "beyond budget is refused");
    expect(recr->cells[tail].pos == (llama_pos) N - 1 && recr->rs_idx[0] == 0, "beyond budget mutates nothing");

    // exactly at the budget: accepted
    expect(rm((llama_pos) (N - 1) - n_rs + 1, -1), "at budget returns true");
    expect(recr->cells[tail].pos == (llama_pos) ((N - 1) - n_rs) && recr->rs_idx[0] == n_rs, "at budget rewinds and stages");

    // removing everything clears the pending rollback and the cell
    expect(rm(-1, -1), "remove all returns true");
    expect(recr->cells[0].pos == -1 && recr->rs_idx[0] == 0, "remove all clears the cell");
    expect(recr->cells[0].tail == -1, "remove all clears the tail");

    fprintf(stderr, "%s : n_rs_seq=%u contract %s (%d failures)\n", __func__, n_rs, (ok && n_fail == 0) ? "PASS" : "FAIL", n_fail);
    llama_free(ctx);
    return ok && n_fail == 0;
}

// The MTP draft context is a hybrid whose recurrent half stores no state. Removing a
// rejected suffix there must be a plain rewind: it must not stage a rollback (nothing
// would ever consume it), it must be repeatable, it must really truncate the caches that
// do hold state, and a context that made the speculative excursion must end up identical
// to one that never went past the commit point.
static bool test_seq_rm_stateless_draft(const common_params & params, llama_model * model) {
    if (llama_model_n_layer_nextn(model) == 0) {
        fprintf(stderr, "%s : skipped (model has no MTP layer)\n", __func__);
        return true;
    }

    const uint32_t n_rs = 4; // what common_speculative_init_result gives the draft context
    llama_context * ctx_txn = make_ctx_test(params, model, n_rs, 256, LLAMA_CONTEXT_TYPE_MTP);
    llama_context * ctx_ref = make_ctx_test(params, model, n_rs, 256, LLAMA_CONTEXT_TYPE_MTP);
    if (ctx_txn == nullptr || ctx_ref == nullptr) {
        fprintf(stderr, "%s : failed to init MTP contexts\n", __func__);
        llama_free(ctx_txn);
        llama_free(ctx_ref);
        return false;
    }

    llama_memory_recurrent * recr = recr_of(ctx_txn);
    if (recr == nullptr || recr->stores_state) {
        fprintf(stderr, "%s : skipped (the draft half does store state on this model)\n", __func__);
        llama_free(ctx_txn);
        llama_free(ctx_ref);
        return true;
    }

    const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
    const int32_t n_embd = (int32_t) llama_model_n_embd_out(model);

    llama_tokens tokens;
    for (uint32_t i = 0; i < 16; ++i) {
        tokens.push_back((llama_token) ((11u*i + 3u) % (uint32_t) n_vocab));
    }

    constexpr llama_pos pos_commit = 6; // committed prefix
    constexpr int32_t   n_draft    = 3; // speculative draft region [6, 9)

    // A: prefix + draft region. B: the prefix only.
    bool ok = decode_rows_embd(ctx_txn, model, n_embd, tokens, 0, pos_commit, false) &&
              decode_rows_embd(ctx_txn, model, n_embd, tokens, pos_commit, n_draft, false) &&
              decode_rows_embd(ctx_ref, model, n_embd, tokens, 0, pos_commit, false);
    if (!ok) {
        fprintf(stderr, "%s : draft-context decode failed\n", __func__);
        llama_free(ctx_txn);
        llama_free(ctx_ref);
        return false;
    }

    llama_memory_t mem = llama_get_memory(ctx_txn);

    int n_fail = 0;
    const auto expect = [&](bool cond, const char * what) {
        if (!cond) {
            fprintf(stderr, "%s : FAILED '%s' (cell.pos=%d tail=%d rs_idx=%u pos_max=%d pos_min=%d)\n",
                    __func__, what, (int) recr->cells[0].pos, (int) recr->cells[0].tail, recr->rs_idx[0],
                    (int) llama_memory_seq_pos_max(mem, 0), (int) llama_memory_seq_pos_min(mem, 0));
            n_fail++;
        }
        ok = ok && cond;
    };

    // the aborting pattern: two suffix removals without a decode in between, exactly as
    // the server issues them (draft-region truncation per round, then the accept path)
    expect(llama_memory_seq_rm(mem, 0, pos_commit, -1), "first draft-region removal returns true");
    expect(recr->rs_idx[0] == 0 && recr->cells[0].tail >= 0, "first removal stages nothing and keeps the cell");
    expect(recr->cells[0].pos == pos_commit - 1, "first removal rewinds the tail");
    expect(llama_memory_seq_rm(mem, 0, pos_commit, -1), "second draft-region removal returns true");
    expect(recr->rs_idx[0] == 0, "second removal stages nothing");
    expect(llama_memory_seq_pos_max(mem, 0) == pos_commit - 1, "removal is real (pos_max reflects it)");

    // both contexts now decode the identical continuation and must agree row for row
    float diff_max = 0.0f;
    if (ok) {
        std::vector<float> la, lb;
        ok = decode_rows_embd(ctx_txn, model, n_embd, tokens, pos_commit, n_draft, true) &&
             decode_rows_embd(ctx_ref, model, n_embd, tokens, pos_commit, n_draft, true);
        for (int32_t i = 0; ok && i < n_draft; ++i) {
            ok = gather_logits(ctx_txn, i, n_vocab, la) && gather_logits(ctx_ref, i, n_vocab, lb);
            if (ok) {
                diff_max = std::max(diff_max, max_logit_diff(la, lb));
            }
        }
    }

    constexpr float eps = 1e-5f;
    if (!ok || diff_max > eps) {
        fprintf(stderr, "%s : FAIL (ok=%d, max logit diff %g)\n", __func__, ok ? 1 : 0, (double) diff_max);
        llama_free(ctx_txn);
        llama_free(ctx_ref);
        return false;
    }

    fprintf(stderr, "%s : draft-half removal repeatable and replay matched (max diff %g)\n",
            __func__, (double) diff_max);
    llama_free(ctx_txn);
    llama_free(ctx_ref);
    return true;
}

// The server's transaction in its real geometry: the target decodes a verify batch of
// n_tail tokens (drafts + the sampled slot), removes the suffix from the commit point on,
// then continues. The next-token logits must equal a context that only ever decoded the
// committed prefix. `repeats` > 0 additionally runs mixed transactions back to back on one
// context, which is where a leak that one transaction hides would accumulate.
static bool test_transaction_parity(const common_params & params, llama_model * model, uint32_t n_rs, bool repeats) {
    const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));

    constexpr uint32_t n_prefix = 8; // committed tokens before the first transaction
    constexpr uint32_t n_cont   = 4; // committed tokens decoded after each transaction
    // A token committed out of a snapshot plane is computed inside a multi-token batch,
    // while the serial reference computes it one token at a time, so these are
    // kernel-rounding bounds, not exact-equality bounds. Measured on the L4 fixture
    // (CPU oracle, 2026-09-12): state 1.30e-06 at depth 1, 7.74e-05 at depth 2,
    // 1.13e-04 at depth 3, logits an order of magnitude below that. The bounds below sit
    // ~10x above the measurement, and the one-token-error control printed beside them
    // shows by how far a real state error would miss them.
    constexpr float    eps       = 1e-3f;
    constexpr float    state_eps = 1e-3f;

    const auto tok = [&](llama_pos pos) { return (llama_token) ((7u*(uint32_t) pos + 31u) % (uint32_t) n_vocab); };

    llama_tokens tokens;
    for (llama_pos p = 0; p < (llama_pos) (n_prefix + 64); ++p) {
        tokens.push_back(tok(p));
    }

    bool  ok       = true;
    int   n_cmp    = 0;
    float diff_max = 0.0f;
    float state_max = 0.0f;

    // pattern: (n_tail, accept) with accept < n_tail, i.e. always a rejected suffix.
    // the non-repeated run isolates each case on its own context pair; the repeated run
    // chains mixed transactions on one pair, which is where a leak would accumulate
    std::vector<std::pair<uint32_t, uint32_t>> plan;
    if (repeats) {
        plan = { {3, 1}, {2, 0}, {1, 0}, {3, 2}, {3, 0}, {2, 1}, {3, 1}, {1, 0} };
    } else {
        for (uint32_t n_tail = 1; n_tail <= n_rs; ++n_tail) {
            for (uint32_t accept = 0; accept < n_tail; ++accept) {
                plan.emplace_back(n_tail, accept);
            }
        }
    }

    llama_context * ctx_txn = nullptr;
    llama_context * ctx_ref = nullptr;
    llama_pos       pos     = 0;

    if (repeats) {
        ctx_txn = make_ctx_test(params, model, n_rs, 512);
        ctx_ref = make_ctx_test(params, model, n_rs, 512);
        if (ctx_txn == nullptr || ctx_ref == nullptr) {
            fprintf(stderr, "%s : failed to init contexts\n", __func__);
            llama_free(ctx_txn);
            llama_free(ctx_ref);
            return false;
        }
    }

    for (const auto & [n_tail, accept] : plan) {
        if (!ok) {
            break;
        }

        if (!repeats) {
            ctx_txn = make_ctx_test(params, model, n_rs, 512);
            ctx_ref = make_ctx_test(params, model, n_rs, 512);
            if (ctx_txn == nullptr || ctx_ref == nullptr) {
                ok = false;
                break;
            }
            pos = 0;
        }

        const llama_pos pos_tail = pos + n_prefix;
        const llama_pos p0       = pos_tail + (llama_pos) accept;
        const uint32_t  depth    = n_tail - accept;

        // both decode the committed prefix first
        ok = ok && decode_rows(ctx_txn, tokens, pos, n_prefix, false) &&
                   decode_rows(ctx_ref, tokens, pos, n_prefix, false);

        // the transaction: speculative tail, then remove the rejected suffix
        ok = ok && decode_rows(ctx_txn, tokens, pos_tail, n_tail, false);
        ok = ok && llama_memory_seq_rm(llama_get_memory(ctx_txn), 0, p0, -1);

        // the reference is the serial world: it decodes the accepted tokens one at a time
        for (llama_pos p = pos_tail; ok && p < p0; ++p) {
            ok = decode_one(ctx_ref, tokens[p], p);
        }

        // state parity, independent of the logits: the two contexts must save the same
        // bytes once both stand on the commit point. A restored-but-wrong plane shows up
        // here as a large element difference where the logits could still be plausible.
        {
            byte_collector b_txn, b_ref;
            llama_get_memory(ctx_txn)->state_write(b_txn);
            llama_get_memory(ctx_ref)->state_write(b_ref);
            const float sd = max_state_diff(b_txn, b_ref);
            if (sd > state_max) {
                state_max = sd;
            }
            if (b_txn.bytes.size() != b_ref.bytes.size()) {
                fprintf(stderr, "%s : state layouts differ (n_tail=%u accept=%u: %zu vs %zu bytes)\n",
                        __func__, n_tail, accept, b_txn.bytes.size(), b_ref.bytes.size());
                ok = false;
            }
            if (sd > 1e-9f) {
                fprintf(stderr, "%s : n_tail=%u accept=%u depth=%u state_diff=%g\n",
                        __func__, n_tail, accept, depth, (double) sd);
            }
        }

        // and both decode the same next tokens, which must agree
        std::vector<float> la, lb;
        for (uint32_t i = 0; ok && i < n_cont; ++i) {
            const llama_pos p = p0 + (llama_pos) i;
            ok = ok && decode_rows(ctx_txn, tokens, p, 1, true) && decode_rows(ctx_ref, tokens, p, 1, true);
            ok = ok && gather_logits(ctx_txn, 0, n_vocab, la) && gather_logits(ctx_ref, 0, n_vocab, lb);
            if (ok) {
                const float d = max_logit_diff(la, lb);
                if (d > diff_max) {
                    diff_max = d;
                }
                n_cmp++;
                if (d > 1e-9f) {
                    fprintf(stderr, "%s : n_tail=%u accept=%u depth=%u pos=%d logit_diff=%g\n",
                            __func__, n_tail, accept, depth, (int) p, (double) d);
                }

            }
        }

        if (!ok) {
            break;
        }

        pos = p0 + n_cont;

        if (!repeats) {
            llama_free(ctx_txn);
            llama_free(ctx_ref);
            ctx_txn = nullptr;
            ctx_ref = nullptr;
        }
    }

    // one beyond the budget is refused: this is the case the server sends to its
    // checkpoint path (n_rollback > llama_n_rs_seq(ctx))
    if (ok && !repeats) {
        llama_context * ctx = make_ctx_test(params, model, n_rs, 256);
        ok = ctx != nullptr && decode_tokens(ctx, tokens, n_rs + 2);
        // p0 = 1 leaves a suffix of n_rs_seq + 1 tokens, one deeper than the snapshots go
        ok = ok && !llama_memory_seq_rm(llama_get_memory(ctx), 0, 1, -1);
        llama_free(ctx);
    }

    // Control: what does a one-token state error cost? The transaction is repeated with a
    // commit point one token earlier while the identical continuation tokens are decoded,
    // so the only difference to the serial reference is one token of state. The
    // transaction's own deviation has to be far below this, otherwise the tolerance above
    // would be hiding exactly the error it is meant to catch.
    float control_logit = 0.0f;
    float control_state = 0.0f;
    if (ok && !repeats && n_rs >= 3) {
        llama_context * ctx_c = make_ctx_test(params, model, n_rs, 512);
        llama_context * ctx_r = make_ctx_test(params, model, n_rs, 512);
        const llama_pos pos_tail = (llama_pos) n_prefix;
        const llama_pos p0       = pos_tail + 2; // accept 2, so the true depth is 1

        ok = ctx_c && ctx_r &&
             decode_rows(ctx_c, tokens, 0, n_prefix, false) && decode_rows(ctx_r, tokens, 0, n_prefix, false) &&
             decode_rows(ctx_c, tokens, pos_tail, 3, false);
        ok = ok && llama_memory_seq_rm(llama_get_memory(ctx_c), 0, p0 - 1, -1); // depth 2: one token too deep
        for (llama_pos p = pos_tail; ok && p < p0; ++p) {
            ok = ok && decode_one(ctx_r, tokens[p], p);
        }

        if (ok) {
            byte_collector b_c, b_r;
            llama_get_memory(ctx_c)->state_write(b_c);
            llama_get_memory(ctx_r)->state_write(b_r);
            control_state = max_state_diff(b_c, b_r);
        }

        std::vector<float> la, lb;
        for (uint32_t i = 0; ok && i < n_cont; ++i) {
            const llama_pos p = p0 + (llama_pos) i;
            ok = ok && decode_rows(ctx_c, tokens, p, 1, true) && decode_rows(ctx_r, tokens, p, 1, true);
            ok = ok && gather_logits(ctx_c, 0, n_vocab, la) && gather_logits(ctx_r, 0, n_vocab, lb);
            if (ok) {
                control_logit = std::max(control_logit, max_logit_diff(la, lb));
            }
        }

        llama_free(ctx_c);
        llama_free(ctx_r);
    }

    llama_free(ctx_txn);
    llama_free(ctx_ref);

    if (!ok) {
        fprintf(stderr, "%s : FAIL\n", __func__);
        return false;
    }

    const bool separated = repeats || (control_logit <= 1e-9f) || (diff_max <= 0.1f * control_logit);

    if (state_max > state_eps || diff_max > eps || !separated) {
        fprintf(stderr, "%s : %s parity FAIL (state %g > %g, logits %g > %g, control logits %g, separated %d)\n",
                __func__, repeats ? "repeated mixed" : "single",
                (double) state_max, (double) state_eps, (double) diff_max, (double) eps,
                (double) control_logit, separated ? 1 : 0);
        return false;
    }

    fprintf(stderr, "%s : %s parity PASS (%d logits rows compared, state %g <= %g, logits %g <= %g, "
            "one-token-error control %g (state %g))\n",
            __func__, repeats ? "repeated mixed" : "single", n_cmp,
            (double) state_max, (double) state_eps, (double) diff_max, (double) eps,
            (double) control_logit, (double) control_state);
    return true;
}

// The server's checkpoint path restores into a context that already holds state: when the
// fallback fires, the target has just decoded the verify batch. A restore must replace that
// state, not merge with it. This compares a restore into a dirty cache against a restore
// into a fresh one, at the byte level first, so a difference cannot hide in the logits.
static bool test_restore_into_dirty(const common_params & params, llama_model * model) {
    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int n_vocab = llama_vocab_n_tokens(vocab);

    const uint32_t n_rs = 8;
    constexpr uint32_t n_rollback = 3;

    auto cparams = common_context_params_to_llama(params);
    cparams.n_seq_max = 1;
    cparams.n_rs_seq  = n_rs;
    cparams.n_batch   = std::max(cparams.n_batch,  (uint32_t) (n_rs + 1));
    cparams.n_ubatch  = std::max(cparams.n_ubatch, (uint32_t) (n_rs + 1));

    llama_tokens tokens;
    if (llama_vocab_type(vocab) == LLAMA_VOCAB_TYPE_NONE) {
        for (uint32_t i = 0; i < n_rs + 1; ++i) {
            tokens.push_back((llama_token) (i + 1));
        }
    } else {
        tokens = common_tokenize(llama_model_get_vocab(model), "The quick brown fox jumps over the lazy dog", true);
        if (tokens.empty()) {
            fprintf(stderr, "%s : not enough prompt tokens\n", __func__);
            return false;
        }
        tokens.resize(n_rs + 1, tokens.back());
    }

    const llama_pos rollback_pos = (llama_pos) tokens.size() - n_rollback;

    llama_context * ctx_src   = llama_init_from_model(model, cparams);
    llama_context * ctx_clean = llama_init_from_model(model, cparams);
    llama_context * ctx_dirty = llama_init_from_model(model, cparams);
    if (ctx_src == nullptr || ctx_clean == nullptr || ctx_dirty == nullptr) {
        fprintf(stderr, "%s : failed to init contexts\n", __func__);
        return false;
    }

    bool ok = decode_tokens(ctx_src, tokens, tokens.size());
    ok = ok && llama_memory_seq_rm(llama_get_memory(ctx_src), 0, rollback_pos, -1);

    common_prompt_checkpoint ckpt;
    ckpt.update_tgt(ctx_src, 0, 0);
    ckpt.load_tgt(ctx_clean, 0, 0);

    // the dirty context: a different prompt of the same length, rolled back the same way,
    // which is what the target looks like when the checkpoint fallback fires
    llama_tokens noise = tokens;
    for (auto & t : noise) {
        t = (t + 1) % n_vocab;
    }
    ok = ok && decode_tokens(ctx_dirty, noise, noise.size());
    ok = ok && llama_memory_seq_rm(llama_get_memory(ctx_dirty), 0, rollback_pos, -1);
    ckpt.load_tgt(ctx_dirty, 0, 0);

    if (!ok) {
        fprintf(stderr, "%s : setup failed\n", __func__);
        llama_free(ctx_src); llama_free(ctx_clean); llama_free(ctx_dirty);
        return false;
    }

    byte_collector b_clean, b_dirty;
    llama_get_memory(ctx_clean)->state_write(b_clean);
    llama_get_memory(ctx_dirty)->state_write(b_dirty);

    size_t diff_at = (size_t) -1;
    for (size_t i = 0; i < b_clean.bytes.size() && i < b_dirty.bytes.size(); ++i) {
        if (b_clean.bytes[i] != b_dirty.bytes[i]) {
            diff_at = i;
            break;
        }
    }

    fprintf(stderr, "%s : restored state bytes: clean %zu, dirty %zu, first difference at %zd\n",
            __func__, b_clean.bytes.size(), b_dirty.bytes.size(), (ptrdiff_t) diff_at);

    // replay the same three tokens on both and compare the logits position by position
    std::vector<float> la, lb;
    float diff_max = 0.0f;
    for (uint32_t i = 0; i < n_rollback && ok; ++i) {
        const llama_pos pos = rollback_pos + (llama_pos) i;
        ok = decode_one(ctx_clean, tokens[pos], pos) && decode_one(ctx_dirty, tokens[pos], pos);
        ok = ok && gather_logits(ctx_clean, 0, n_vocab, la) && gather_logits(ctx_dirty, 0, n_vocab, lb);
        if (ok) {
            const float d = max_logit_diff(la, lb);
            diff_max = std::max(diff_max, d);
            fprintf(stderr, "%s :   replay pos %d: max logit diff %g\n", __func__, (int) pos, (double) d);
        }
    }

    // a second restore into the same context must land on the same state, so a leftover
    // from the first one cannot be the cause of a difference
    byte_collector b_again;
    ckpt.load_tgt(ctx_dirty, 0, 0);
    llama_get_memory(ctx_dirty)->state_write(b_again);
    const bool again_equal = b_again.bytes == b_clean.bytes;

    fprintf(stderr, "%s : %s (max logit diff %g, second restore identical to clean: %d)\n",
            __func__, diff_max <= 1e-5f ? "PASS" : "DIFFERS", (double) diff_max, again_equal ? 1 : 0);

    llama_free(ctx_src);
    llama_free(ctx_clean);
    llama_free(ctx_dirty);

    return diff_max <= 1e-5f;
}

static int test_rollback(const common_params & params, llama_model * model, uint8_t fill) {
    const llama_vocab * vocab   = llama_model_get_vocab(model);
    const int           n_vocab = llama_vocab_n_tokens(vocab);

    llama_context * ctx_src = make_ctx(params, model, fill);
    llama_context * ctx_dst = make_ctx(params, model, fill);
    if (ctx_src == nullptr || ctx_dst == nullptr) {
        fprintf(stderr, "%s : failed to init contexts\n", __func__);
        return 1;
    }

    if (llama_n_rs_seq(ctx_src) == 0) {
        fprintf(stderr, "%s : skipping because n_rs_seq is disabled\n", __func__);
        llama_free(ctx_src);
        llama_free(ctx_dst);
        return 0;
    }

    std::vector<llama_token> tokens;
    if (llama_vocab_type(vocab) == LLAMA_VOCAB_TYPE_NONE) {
        tokens = { 1, 2, 3, 4, 5, 6, 7, 8, 9 };
    } else {
        tokens = common_tokenize(ctx_src, "The quick brown fox jumps over the lazy dog", true);
    }
    const uint32_t n_rs_seq = llama_n_rs_seq(ctx_src);
    constexpr uint32_t n_rollback = 3;
    if (n_rs_seq < n_rollback) {
        fprintf(stderr, "%s : skipping because n_rs_seq is too small\n", __func__);
        llama_free(ctx_src);
        llama_free(ctx_dst);
        return 0;
    }
    if (tokens.empty()) {
        fprintf(stderr, "%s : not enough prompt tokens\n", __func__);
        return 1;
    }
    tokens.resize(n_rs_seq + 1, tokens.back());

    const uint32_t  n_tokens     = tokens.size();
    const llama_pos rollback_pos = (llama_pos) n_tokens - n_rollback;

    // Decode the full prompt on the source, then roll back three positions.
    // Replaying them crosses DSV4's ratio-4 compressor boundary.
    // Rollback leaves the recurrent memory in a snapshot state (rs_idx != 0).
    if (!decode_tokens(ctx_src, tokens, n_tokens)) {
        fprintf(stderr, "%s : failed to decode prompt\n", __func__);
        return 1;
    }
    if (!llama_memory_seq_rm(llama_get_memory(ctx_src), 0, rollback_pos, -1)) {
        fprintf(stderr, "%s : rollback failed\n", __func__);
        return 1;
    }

    // Save the rolled-back state and restore it into a fresh context.
    common_prompt_checkpoint ckpt;
    ckpt.update_tgt(ctx_src, 0, 0);
    ckpt.load_tgt(ctx_dst, 0, 0);

    constexpr float eps = 1e-5f;
    std::vector<std::vector<float>> logits_src_replay(n_rollback);
    const auto replay_and_compare = [&](const char * mode) {
        for (uint32_t i = 0; i < n_rollback; ++i) {
            const llama_pos pos = rollback_pos + i;
            if (!decode_one(ctx_src, tokens[pos], pos) ||
                !decode_one(ctx_dst, tokens[pos], pos)) {
                fprintf(stderr, "%s : %s replay failed at position %d\n", __func__, mode, pos);
                return false;
            }

            const float * logits_src = llama_get_logits_ith(ctx_src, 0);
            const float * logits_dst = llama_get_logits_ith(ctx_dst, 0);
            if (logits_src == nullptr || logits_dst == nullptr) {
                fprintf(stderr, "%s : missing %s logits at position %d\n", __func__, mode, pos);
                return false;
            }

            logits_src_replay[i].assign(logits_src, logits_src + n_vocab);
            for (int token = 0; token < n_vocab; ++token) {
                if (logit_diff(logits_src[token], logits_dst[token]) > eps) {
                    fprintf(stderr, "%s : %s logits mismatch at position %d, token %d (%g != %g)\n",
                            __func__, mode, pos, token, (double) logits_src[token], (double) logits_dst[token]);
                    return false;
                }
            }
        }
        return true;
    };
    if (!replay_and_compare("full")) {
        return 1;
    }

    if (!llama_memory_seq_rm(llama_get_memory(ctx_src), 0, rollback_pos, -1) ||
        !llama_memory_seq_rm(llama_get_memory(ctx_dst), 0, rollback_pos, -1)) {
        fprintf(stderr, "%s : partial rollback failed\n", __func__);
        return 1;
    }

    constexpr llama_state_seq_flags partial_flags = LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY;
    common_prompt_checkpoint ckpt_partial;
    ckpt_partial.update_tgt(ctx_src, 0, partial_flags);
    ckpt_partial.load_tgt(ctx_dst, 0, partial_flags);

    if (!replay_and_compare("partial")) {
        return 1;
    }

    // Repeat the load into a context that already has its own rollback state:
    // groups 1..n_rs_seq hold a different prompt's history, and rs_idx[0] is
    // non-zero at load time. The restore must wipe that state and still match.
    llama_context * ctx_dirty = make_ctx(params, model, fill);
    if (ctx_dirty == nullptr) {
        fprintf(stderr, "%s : failed to init dirty ctx\n", __func__);
        return 1;
    }

    std::vector<llama_token> noise = tokens;
    for (auto & t : noise) {
        t = (t + 1) % n_vocab;
        if (t < 0) {
            t = 0;
        }
    }
    if (!decode_tokens(ctx_dirty, noise, n_tokens)) {
        fprintf(stderr, "%s : dirty prompt decode failed\n", __func__);
        return 1;
    }
    if (!llama_memory_seq_rm(llama_get_memory(ctx_dirty), 0, rollback_pos, -1)) {
        fprintf(stderr, "%s : dirty rollback failed\n", __func__);
        return 1;
    }

    ckpt.load_tgt(ctx_dirty, 0, 0);

    for (uint32_t i = 0; i < n_rollback; ++i) {
        const llama_pos pos = rollback_pos + i;
        if (!decode_one(ctx_dirty, tokens[pos], pos)) {
            fprintf(stderr, "%s : dirty replay failed at position %d\n", __func__, pos);
            return 1;
        }

        const float * logits_dirty = llama_get_logits_ith(ctx_dirty, 0);
        if (logits_dirty == nullptr) {
            fprintf(stderr, "%s : missing dirty logits at position %d\n", __func__, pos);
            return 1;
        }

        for (int token = 0; token < n_vocab; ++token) {
            if (logit_diff(logits_src_replay[i][token], logits_dirty[token]) > eps) {
                fprintf(stderr, "%s : dirty-ctx logits mismatch at position %d, token %d (%g != %g)\n",
                        __func__, pos, token, (double) logits_src_replay[i][token], (double) logits_dirty[token]);
                return 1;
            }
        }
    }

    fprintf(stderr, "%s : recurrent rollback checkpoint restored successfully\n", __func__);
    llama_free(ctx_src);
    llama_free(ctx_dst);
    llama_free(ctx_dirty);

    if (!test_multi_seq_split_replay(params, model, n_vocab, fill)) {
        return 1;
    }

    return 0;
}

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");

    common_params params;
    params.sampling.seed = 1234;
    params.n_predict = 1;

    common_init();

    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_COMMON)) {
        return 1;
    }

    ggml_backend_load_all();

    common_init_result_ptr llama_init = common_init_from_params(params);
    llama_model * model = llama_init->model();
    if (model == nullptr) {
        fprintf(stderr, "%s : failed to init model\n", __func__);
        return 1;
    }

    if (!llama_model_is_recurrent(model) && !llama_model_is_hybrid(model)) {
        fprintf(stderr, "%s : skipping for non-recurrent model\n", __func__);
        return 0;
    }

    // the contract and the MTP transaction geometry, at the budgets the server uses.
    // these run before the upstream rollback test so a pre-existing failure there does
    // not hide their result
    for (uint32_t n_rs : { 1u, 2u, 3u, 4u, 8u }) {
        if (!test_seq_rm_contract(params, model, n_rs)) {
            return 1;
        }
    }

    if (!test_seq_rm_stateless_draft(params, model)) {
        return 1;
    }

    for (uint32_t n_rs : { 3u, 4u }) {
        if (!test_transaction_parity(params, model, n_rs, false) ||
            !test_transaction_parity(params, model, n_rs, true)) {
            return 1;
        }
    }

    if (!test_restore_into_dirty(params, model)) {
        return 1;
    }

    for (uint8_t fill : { 0, 0x3e }) {
        fprintf(stderr, "%s : testing with cache fill 0x%02x\n", __func__, fill);
        if (test_rollback(params, model, fill) != 0) {
            return 1;
        }
    }

    return 0;
}
