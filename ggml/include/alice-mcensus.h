// ALICE master-thread census (lane/master-census) + cross-backend copy census (lane/copy-census).
//
// Attribution instrument for the *master thread* (the thread that calls llama_decode):
// per-ubatch graph construction / scheduler split & alloc / input staging / per-split
// dispatch, and the server-side decode + sampling + task loop.
//
// Design: one line per event with an absolute CLOCK_MONOTONIC timestamp, so a run can be
// sliced into settle reps by wall clock alone. Enabled by ALICE_MC=1 (log path from
// ALICE_MC_LOG). Every hook is a branch on a global int when off; the shipped build is
// unaffected when the env var is absent.
//
// Copy census (lane/copy-census): one `CP` line per cross-backend input copy performed by
// ggml_backend_sched_compute_splits, carrying the exact path the copy took and the VK
// submit/fence-wait cost it caused. The path bits are OR-ed in by the backend
// (ggml-vulkan) through alice_mc_note(); the counters alice_mc_vk_submit / alice_mc_vk_wait
// live in the Vulkan library and are snapshot-differenced per copy by the scheduler.
#ifndef ALICE_MCENSUS_H
#define ALICE_MCENSUS_H

#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "ggml.h"

#ifdef __cplusplus
extern "C" {
#endif

GGML_API int alice_mc_enabled; // set by a constructor from ALICE_MC

GGML_API uint64_t alice_mc_now_ns(void);
GGML_API void     alice_mc_ev(const char * fmt, ...);

// ---- cross-backend copy census ------------------------------------------------------------

// path bits, OR-accumulated per copy
#define AMC_PATH_HOST_MEMCPY     (1u << 0)  // dst VK buffer host-visible: plain CPU memcpy, no submit
#define AMC_PATH_PINNED_STAGE    (1u << 1)  // src/dst is a registered pinned VK host heap: GPU copyBuffer
#define AMC_PATH_TEMPCTX_SUBMIT  (1u << 2)  // temp ctx + copyBuffer + vkQueueSubmit + fence wait
#define AMC_PATH_STAGE_SUBMIT    (1u << 3)  // sync_staging + copyBuffer + submit + fence wait
#define AMC_PATH_VK_COPYBUFFER   (1u << 4)  // vk->vk copyBuffer inside an already-live ctx
#define AMC_PATH_MALLOC_FALLBACK (1u << 5)  // ggml_backend_tensor_copy malloc round trip
#define AMC_PATH_UMA_MEMCPY      (1u << 6)  // UMA device: barrier submit + fence wait + memcpy
#define AMC_PATH_MOE_GROUP       (1u << 7)  // MoE expert-group branch: per-group tensor_set_async

// rejection bits: the reason an attempted fast path was refused
#define AMC_REJ_DST_NOT_VK       (1u << 8)  // dst->buffer->buft != vk default device buffer type
#define AMC_REJ_SRC_NOT_PINNED   (1u << 9)  // ggml_vk_host_get(src->data) == nullptr
#define AMC_REJ_NOT_IDLE         (1u << 10) // ctx->submit_pending || almost_ready_fence_pending || !transfer_idle
#define AMC_REJ_DST_NOT_COHERENT (1u << 11) // dst buffer not (host-visible & host-coherent)
#define AMC_REJ_SIZE             (1u << 12) // ggml_nbytes(src) > alice_vk_cpu_copy_max()
#define AMC_REJ_SRC_NOT_HOST     (1u << 13) // src buffer is neither vk nor host
#define AMC_ACCEPT_FASTPATH      (1u << 16) // the <=128 KiB host-coherent CPU-copy path was taken
#define AMC_ACCEPT_VKASYNC       (1u << 17) // cpy_tensor_async accepted (copyBuffer into a live ctx)

#define AMC_FLAG_NOTE_MASK       (0xffffff00u)

struct alice_mc_slot {
    uint64_t t0, t1;      // ns, whole copy call incl. both synchronise() s
    uint64_t presync;     // ns spent inside the calls made before the copy itself
    uint64_t bytes;
    uint64_t sub0, wait0; // vk counter snapshots at slot begin
    uint64_t sub, wait;   // deltas over the slot
    uint32_t paths;
    int32_t  dir;         // 1 cpu->vk, 2 vk->cpu, 0 other/unknown
    int32_t  be_src, be_dst, split_id;
    int32_t  size_ok, pinned, coherent, idle;
    char     src[48], dst[48];
};

GGML_API struct alice_mc_slot alice_mc_slot;

// Vulkan-library counters, incremented by ggml-vulkan.
GGML_API uint64_t alice_mc_vk_submit;
GGML_API uint64_t alice_mc_vk_wait;

// OR mask into the current slot, recording a path taken and/or a rejection reason.
static inline void alice_mc_note(uint32_t bits) {
    alice_mc_slot.paths |= bits;
}

// a blocking Vulkan fence wait on the copy/transfer path
#define AMC_VK_WAIT() do { if (alice_mc_enabled) { alice_mc_vk_wait++; } } while (0)

#ifdef __cplusplus
}
#endif

#ifdef __cplusplus
// accumulates ns into *acc over its lifetime; ~0 cost when the census is off
struct alice_mc_timer {
    uint64_t * acc;
    uint64_t   t0;
    explicit alice_mc_timer(uint64_t * a) : acc(a), t0(alice_mc_enabled ? alice_mc_now_ns() : 0) {}
    ~alice_mc_timer() { if (alice_mc_enabled) { *acc += alice_mc_now_ns() - t0; } }
    alice_mc_timer(const alice_mc_timer &) = delete;
    alice_mc_timer & operator=(const alice_mc_timer &) = delete;
};
#define ALICE_MC_ON() (alice_mc_enabled != 0)
#define ALICE_MC_EV(...) do { if (alice_mc_enabled) { alice_mc_ev(__VA_ARGS__); } } while (0)
#endif

#endif // ALICE_MCENSUS_H
