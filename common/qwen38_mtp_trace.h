// qwen38_mtp_trace.h — deterministic MTP transaction trace (coordinator, 2026-09-14)
//
// Purpose: produce the compact per-position table needed to localise the Flash MTP acceptance=0
// defect from first principles:
//
//   round  n_draft  n_acc  n_rollback  rows        <- one header line per speculative round
//   round  pos  draft_tok  verifier_tok  agree     <- one line per draft position
//   step  seq  row  pos  in_tok  proposed  |h|     <- one line per draft step (draft-side state)
//
// Activation (no cost when unused):
//   QWEN38_MTP_TRACE_FILE=/path/to/mtp_trace.log    enables appending to that file
//   R4X_MTP_DEBUG=1                                 (existing switch) enables the draft-step rows
//
// Header-only on purpose: no CMake change, no ABI change, no allocation in the hot path beyond
// one fopen at first use. Lines are flushed per write so a killed server still leaves a usable trail.
#pragma once

#include <cstdio>
#include <cstdlib>
#include <cstdint>

#if defined(__GNUC__)
#  define QWEN38_TRACE_PRINTF(a, b) __attribute__((format(printf, a, b)))
#else
#  define QWEN38_TRACE_PRINTF(a, b)
#endif

namespace qwen38 {

inline FILE * mtp_trace_file() {
    static FILE * f = []() -> FILE * {
        const char * p = getenv("QWEN38_MTP_TRACE_FILE");
        if (p == nullptr || p[0] == '\0') {
            return nullptr;
        }
        return fopen(p, "a");
    }();
    return f;
}

inline void mtp_trace_round(int round, int n_draft, int n_acc, int n_rollback, int rows) {
    FILE * f = mtp_trace_file();
    if (f == nullptr) {
        return;
    }
    fprintf(f, "MTPTX_ROUND round=%d n_draft=%d n_acc=%d n_rollback=%d rows=%d\n",
            round, n_draft, n_acc, n_rollback, rows);
    fflush(f);
}

inline void mtp_trace_pos(int round, int pos, int draft_tok, int verifier_tok, int agree) {
    FILE * f = mtp_trace_file();
    if (f == nullptr) {
        return;
    }
    fprintf(f, "MTPTX_POS round=%d pos=%d draft_tok=%d verifier_tok=%d agree=%d\n",
            round, pos, draft_tok, verifier_tok, agree);
    fflush(f);
}

inline void mtp_trace_draft_step(int step, int seq, int row, int pos, int in_tok, int proposed, float hnorm, float hout_norm) {
    FILE * f = mtp_trace_file();
    if (f == nullptr) {
        return;
    }
    fprintf(f, "MTPTX_STEP step=%d seq=%d row=%d pos=%d in_tok=%d proposed=%d h=%0.4f hout=%0.4f\n",
            step, seq, row, pos, in_tok, proposed, (double) hnorm, (double) hout_norm);
    fflush(f);
}

inline void mtp_trace_pair(int seq, float cos_prev_hout_vs_hin, float l2rel_prev, float lmax, float lmin, int live, int finite) {
    FILE * f = mtp_trace_file();
    if (f == nullptr) { return; }
    fprintf(f, "MTPTX_PAIR seq=%d cos_hout_prev_vs_hin=%.4f l2rel=%.4f lmax=%.2f lmin=%.2f live_logits=%d finite=%d\n",
            seq, (double) cos_prev_hout_vs_hin, (double) l2rel_prev, (double) lmax, (double) lmin, live, finite);
    fflush(f);
}

} // namespace qwen38
