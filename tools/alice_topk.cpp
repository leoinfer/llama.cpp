// alice_topk.cpp — accumulate per-(layer, expert) route mass over a corpus.
//
// The alice_ai graph emits probe nodes named ffn_moe_topk-<il> whenever the
// ALICE_PROBE_TOPK env var is set. This hook API scans the built graph for those
// nodes after each decode, reads the selected expert IDs, and accumulates mass
// into per-layer histograms. Output is a JSON route-mass table:
//
//   { "n_layer": N, "n_expert": E, "topk": K, "tokens": T,
//     "counts": [[...512 per layer...]...] }
//
// The same run is repeated per prompt; route mass = counts / sum.
//
// build:  g++ -O2 -std=c++17 -I include -I ggml/include tools/alice_topk.cpp \
//              -L build-vk/bin -lllama -lggml -lggml-base \
//              -Wl,-rpath,$PWD/build-vk/bin
// run:    ALICE_PROBE_TOPK=1 ./alice_topk <model.gguf> <prompt-file> <out.json> [--ctx N]

#include "ggml.h"
#include "ggml-backend.h"
#include "llama.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

// ---- log capture: the probe hook logs selections here -----------------------
// We cannot reach inside the compiled graph, so instead this tool works with a
// tiny runtime patch (see below) that writes ggml_graph_get_tensor("ffn_moe_topk-<il>")
// contents to the log. Simpler: parse them from the regular log stream if
// LLAMA_DEBUG dumping is enabled. For now the tool drives decodes one token at
// a time and lets ALICE_MOE_STATS (written by the runtime on exit) record
// counts; see src/models/alice_ai.cpp for the matching accumulator.

int main(int argc, char ** argv) {
    fprintf(stderr,
        "alice_topk: route-mass collection is runtime-assisted.\n"
        "Set ALICE_MOE_STATS=<path> and run any llama-cli/bench on the model;\n"
        "the alice_ai graph accumulates per-(layer,expert) selection counts\n"
        "and writes them as JSON on exit.\n"
        "This stub documents the protocol; the accumulator lives in the graph.\n");
    return 2;
}
