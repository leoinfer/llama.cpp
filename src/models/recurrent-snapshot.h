#pragma once

#include "ggml.h"


// Shared arithmetic for the (1 + n_rs_seq) plane recurrent-state caches that
// speculative rollback restores from.
//
// A plane index IS a rollback depth: llama_memory_recurrent::s_copy() gathers
// row `rs_idx * size + src` and seq_rm() sets rs_idx = the number of tokens to
// undo, so plane r must hold the state as it was r tokens back. Both the
// S-state planes (written by ggml_gated_delta_net, documented in ggml.h:
// "slot s = state s tokens back") and the conv-state planes have to use that one
// convention or a rejected draft row restores a conv window and a recurrent
// state taken from different steps.
//
// A conv state is a sliding window of `win` inputs ending at the current token,
// and conv_x is the [history(win) | ubatch(n_tokens)] concatenation, so the
// snapshot for rollback depth `slot` is the window ending `slot` tokens before
// the end of the ubatch, i.e. at dim-0 offset `n_tokens - slot`.
//
// Returns the dim-0 offset (in elements) of the snapshot window for `slot`.
// Precondition: 0 <= slot <= n_tokens (callers only write slots that exist).
static inline int64_t llm_conv_snapshot_offset(int64_t n_tokens, int64_t slot) {
    GGML_ASSERT(slot >= 0);
    GGML_ASSERT(slot <= n_tokens);

    return n_tokens - slot;
}
