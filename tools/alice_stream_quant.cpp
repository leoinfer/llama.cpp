// alice_stream_quant.cpp — pipe a tensor through llama.cpp's own quantizers.
//
// The streaming Alice build cannot hold a 162 GB intermediate GGUF, and gguf-py
// implements only *de*quantization for the K-quants. So the driver reads float32
// tensors from stdin, calls ggml_quantize_chunk() here, and appends the result
// straight into the output GGUF. Same code path llama-quantize uses, so the
// bytes are identical to a normal quantize run.
//
// Usage: alice_stream_quant <type> <nrows> <n_per_row> [imatrix.f32]
//   stdin : nrows*n_per_row float32, row-major
//   stdout: quantized block bytes (row_size(type, n_per_row) * nrows)

#include "ggml.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

int main(int argc, char ** argv) {
    if (argc < 4) {
        fprintf(stderr, "usage: %s <type> <nrows> <n_per_row> [imatrix.f32]\n", argv[0]);
        return 2;
    }
    const std::string tname = argv[1];
    const int64_t nrows = atoll(argv[2]);
    const int64_t n_per_row = atoll(argv[3]);

    enum ggml_type type = GGML_TYPE_COUNT;
    for (int t = 0; t < GGML_TYPE_COUNT; ++t) {
        if (tname == ggml_type_name((ggml_type) t)) { type = (ggml_type) t; break; }
    }
    if (type == GGML_TYPE_COUNT) {
        fprintf(stderr, "unknown ggml type '%s'\n", tname.c_str());
        return 2;
    }

    const int64_t nelem = nrows * n_per_row;
    std::vector<float> src((size_t) nelem);

    // Allow either a header-less raw stream or a simple count-prefixed stream.
    if (fread(src.data(), sizeof(float), (size_t) nelem, stdin) != (size_t) nelem) {
        fprintf(stderr, "short read: wanted %lld float32\n", (long long) nelem);
        return 3;
    }

    std::vector<float> imatrix;
    if (argc >= 5) {
        FILE * f = fopen(argv[4], "rb");
        if (!f) { fprintf(stderr, "cannot open imatrix %s\n", argv[4]); return 3; }
        imatrix.resize((size_t) n_per_row);
        if (fread(imatrix.data(), sizeof(float), (size_t) n_per_row, f) != (size_t) n_per_row) {
            fprintf(stderr, "imatrix short read (%lld)\n", (long long) n_per_row);
            return 3;
        }
        fclose(f);
    }

    const size_t row_size = ggml_row_size(type, n_per_row);
    std::vector<uint8_t> dst(row_size * (size_t) nrows);

    const size_t written = ggml_quantize_chunk(
        type, src.data(), dst.data(), 0, nrows, n_per_row,
        imatrix.empty() ? nullptr : imatrix.data());

    if (written != dst.size()) {
        fprintf(stderr, "quantize_chunk wrote %zu, expected %zu\n", written, dst.size());
        return 4;
    }
    if (fwrite(dst.data(), 1, dst.size(), stdout) != dst.size()) {
        fprintf(stderr, "short write\n");
        return 5;
    }
    return 0;
}
