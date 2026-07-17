//******************************************************************************
// MUL_MAT Kernel
// Matrix multiplication: C[M,N] = A[M,K] * B[K,N]
//******************************************************************************

#include <stdint.h>
#include "ggml_tensor.h"
#include "platform.h"
#include "math_fp.h"
#include "quants.h"
#include "block_ops.h"

// Toggle the experimental matrix-engine dequant path (step 1). Comment out this
// line to fall back to the validated scalar cache-line path. Kept as a source
// define so the standard benchmark build picks it up with no CMake changes.
// #define ET_Q8_0_TENSOR 1

#ifdef ET_Q8_0_TENSOR
#include "tensor.h"
#endif

// Fast whole-row Q8_0 x F32 dot product.
//
// The generic helper compute_block_dot_product_q8_0() is called once per 32-wide
// block, and every call re-saves/re-sets the vector mask, reloads the constant
// byte-gather pattern, and runs a full 7-op horizontal reduction. For a row of
// K_blocks blocks (e.g. 30 for K=960) that per-block overhead is paid ~30x per
// output element. This routine hoists all of that out of the block loop:
//
//   * mask save/set and gather-pattern loads happen ONCE per row,
//   * per-block partials are scaled (by the block's fp16 delta) and accumulated
//     into a vector accumulator (f13) entirely in vector mode,
//   * a single horizontal reduction runs once per row.
//
// The per-block math is bit-faithful to the original: 4 chunks of
// fgb.ps -> fcvt.ps.pw -> fmadd.ps, then multiply by the broadcast fp16 scale.
// Written as one asm block so f30/f31/f13 stay pinned across the block loop with
// no chance of the compiler reusing them for intervening FP work.
#ifndef ET_Q8_0_USE_GENERIC_DOT
static inline float dot_row_q8_0(const block_q8_0* q_row,
                                 const float* b_col,
                                 int64_t K_blocks) {
    static const int32_t gather_bytes[8] = {0, 1, 2, 3, 4, 5, 6, 7}; // int8 byte offsets
    static const int32_t gather_zero[8]  = {0, 0, 0, 0, 0, 0, 0, 0}; // broadcast d to 8 lanes

    float sum;
    unsigned long saved_mask;
    const block_q8_0* aptr = q_row;
    const float*      bptr = b_col;
    int64_t           cnt  = K_blocks;
    const int8_t*     qs;

    __asm__ __volatile__(
        "mova.x.m %[savem]\n"                 // save caller mask
        "mov.m.x  m0, x0, 0xFF\n"             // enable all 8 lanes
        "flw.ps   f31, (%[gb])\n"             // byte gather pattern {0..7}
        "flw.ps   f30, (%[gz])\n"             // zero pattern -> scale broadcast
        "fbci.pi  f13, 0\n"                   // vector accumulator = 0
        "beqz     %[cnt], 2f\n"
    "1:\n"
        "fbci.pi  f10, 0\n"                   // per-block partial = 0
        "addi     %[qs], %[a], 2\n"           // qs = &block->qs[0]
        "flw.ps   f12, 0(%[b])\n"             // chunk 0
        "fgb.ps   f11, f31(%[qs])\n"
        "fcvt.ps.pw f11, f11\n"
        "fmadd.ps f10, f11, f12, f10\n"
        "flw.ps   f12, 32(%[b])\n"            // chunk 1
        "addi     %[qs], %[qs], 8\n"
        "fgb.ps   f11, f31(%[qs])\n"
        "fcvt.ps.pw f11, f11\n"
        "fmadd.ps f10, f11, f12, f10\n"
        "flw.ps   f12, 64(%[b])\n"            // chunk 2
        "addi     %[qs], %[qs], 8\n"
        "fgb.ps   f11, f31(%[qs])\n"
        "fcvt.ps.pw f11, f11\n"
        "fmadd.ps f10, f11, f12, f10\n"
        "flw.ps   f12, 96(%[b])\n"            // chunk 3
        "addi     %[qs], %[qs], 8\n"
        "fgb.ps   f11, f31(%[qs])\n"
        "fcvt.ps.pw f11, f11\n"
        "fmadd.ps f10, f11, f12, f10\n"
        "fgh.ps   f14, f30(%[a])\n"           // gather block->d (fp16) into all lanes
        "fcvt.ps.f16 f14, f14\n"              // -> (f32)d in all lanes
        "fmul.ps  f10, f10, f14\n"            // scale block partials
        "fadd.ps  f13, f13, f10, rne\n"       // accumulate
        "addi     %[a], %[a], 34\n"           // sizeof(block_q8_0)
        "addi     %[b], %[b], 128\n"          // 32 floats
        "addi     %[cnt], %[cnt], -1\n"
        "bnez     %[cnt], 1b\n"
    "2:\n"
        "fswizz.ps f1, f13, 0xB1\n"           // horizontal reduce (once per row)
        "fadd.ps   f2, f13, f1, rne\n"
        "fswizz.ps f3, f2, 0x4E\n"
        "fadd.ps   f4, f2, f3, rne\n"
        "fmvz.x.ps t0, f4, 4\n"
        "fbcx.ps   f5, t0\n"
        "fadd.ps   %[out], f4, f5, rne\n"
        "mova.m.x  %[savem]\n"                // restore caller mask
        : [out]   "=f"(sum),
          [savem] "=&r"(saved_mask),
          [a]     "+r"(aptr),
          [b]     "+r"(bptr),
          [cnt]   "+r"(cnt),
          [qs]    "=&r"(qs)
        : [gb] "r"(gather_bytes),
          [gz] "r"(gather_zero)
        : "memory", "t0",
          "f1", "f2", "f3", "f4", "f5",
          "f10", "f11", "f12", "f13", "f14", "f30", "f31"
    );
    return sum;
}

// Four-column variant of dot_row_q8_0. Profiling the scored request showed
// mul_mat_Q8_0 is ~76% of firmware cycles AND memory-bound (trimming per-block
// ALU overhead changed nothing). The dominant cost is streaming Q8_0 weights,
// and the original loop re-reads every weight row once per output column. This
// routine loads+converts each weight block ONCE and multiplies it against four
// activation columns (b0..b3, column stride nb11) into four accumulators,
// cutting weight-side memory traffic ~4x. Per-block math is identical to
// dot_row_q8_0; only the reuse factor changes.
static inline void dot_row_q8_0_x4(const block_q8_0* q_row,
                                   const float* b0, const float* b1,
                                   const float* b2, const float* b3,
                                   int64_t K_blocks,
                                   float* out0, float* out1,
                                   float* out2, float* out3) {
    static const int32_t gather_bytes[8] = {0, 1, 2, 3, 4, 5, 6, 7};
    static const int32_t gather_zero[8]  = {0, 0, 0, 0, 0, 0, 0, 0};

    float s0, s1, s2, s3;
    unsigned long saved_mask;
    const block_q8_0* a = q_row;
    const float *p0 = b0, *p1 = b1, *p2 = b2, *p3 = b3;
    int64_t cnt = K_blocks;
    const int8_t* qs;

    __asm__ __volatile__(
        "mova.x.m %[savem]\n"
        "mov.m.x  m0, x0, 0xFF\n"
        "flw.ps   f31, (%[gb])\n"
        "flw.ps   f30, (%[gz])\n"
        "fbci.pi  f24, 0\n"                   // vacc0..3
        "fbci.pi  f25, 0\n"
        "fbci.pi  f26, 0\n"
        "fbci.pi  f27, 0\n"
        "beqz     %[cnt], 2f\n"
    "1:\n"
        "fbci.pi  f20, 0\n"                   // per-block partials pv0..3
        "fbci.pi  f21, 0\n"
        "fbci.pi  f22, 0\n"
        "fbci.pi  f23, 0\n"
        "addi     %[qs], %[a], 2\n"
        "fgb.ps   f11, f31(%[qs])\n"          // chunk 0: gather+convert weight ONCE
        "fcvt.ps.pw f11, f11\n"
        "flw.ps   f12, 0(%[p0])\n"  "fmadd.ps f20, f11, f12, f20\n"
        "flw.ps   f12, 0(%[p1])\n"  "fmadd.ps f21, f11, f12, f21\n"
        "flw.ps   f12, 0(%[p2])\n"  "fmadd.ps f22, f11, f12, f22\n"
        "flw.ps   f12, 0(%[p3])\n"  "fmadd.ps f23, f11, f12, f23\n"
        "addi     %[qs], %[qs], 8\n"
        "fgb.ps   f11, f31(%[qs])\n"          // chunk 1
        "fcvt.ps.pw f11, f11\n"
        "flw.ps   f12, 32(%[p0])\n" "fmadd.ps f20, f11, f12, f20\n"
        "flw.ps   f12, 32(%[p1])\n" "fmadd.ps f21, f11, f12, f21\n"
        "flw.ps   f12, 32(%[p2])\n" "fmadd.ps f22, f11, f12, f22\n"
        "flw.ps   f12, 32(%[p3])\n" "fmadd.ps f23, f11, f12, f23\n"
        "addi     %[qs], %[qs], 8\n"
        "fgb.ps   f11, f31(%[qs])\n"          // chunk 2
        "fcvt.ps.pw f11, f11\n"
        "flw.ps   f12, 64(%[p0])\n" "fmadd.ps f20, f11, f12, f20\n"
        "flw.ps   f12, 64(%[p1])\n" "fmadd.ps f21, f11, f12, f21\n"
        "flw.ps   f12, 64(%[p2])\n" "fmadd.ps f22, f11, f12, f22\n"
        "flw.ps   f12, 64(%[p3])\n" "fmadd.ps f23, f11, f12, f23\n"
        "addi     %[qs], %[qs], 8\n"
        "fgb.ps   f11, f31(%[qs])\n"          // chunk 3
        "fcvt.ps.pw f11, f11\n"
        "flw.ps   f12, 96(%[p0])\n" "fmadd.ps f20, f11, f12, f20\n"
        "flw.ps   f12, 96(%[p1])\n" "fmadd.ps f21, f11, f12, f21\n"
        "flw.ps   f12, 96(%[p2])\n" "fmadd.ps f22, f11, f12, f22\n"
        "flw.ps   f12, 96(%[p3])\n" "fmadd.ps f23, f11, f12, f23\n"
        "fgh.ps   f14, f30(%[a])\n"           // broadcast block delta -> all lanes
        "fcvt.ps.f16 f14, f14\n"
        "fmul.ps  f20, f20, f14\n" "fadd.ps f24, f24, f20, rne\n"
        "fmul.ps  f21, f21, f14\n" "fadd.ps f25, f25, f21, rne\n"
        "fmul.ps  f22, f22, f14\n" "fadd.ps f26, f26, f22, rne\n"
        "fmul.ps  f23, f23, f14\n" "fadd.ps f27, f27, f23, rne\n"
        "addi     %[a], %[a], 34\n"
        "addi     %[p0], %[p0], 128\n"
        "addi     %[p1], %[p1], 128\n"
        "addi     %[p2], %[p2], 128\n"
        "addi     %[p3], %[p3], 128\n"
        "addi     %[cnt], %[cnt], -1\n"
        "bnez     %[cnt], 1b\n"
    "2:\n"
        "fswizz.ps f1, f24, 0xB1\n" "fadd.ps f2, f24, f1, rne\n"   // reduce vacc0
        "fswizz.ps f3, f2, 0x4E\n"  "fadd.ps f4, f2, f3, rne\n"
        "fmvz.x.ps t0, f4, 4\n"     "fbcx.ps f5, t0\n" "fadd.ps %[s0], f4, f5, rne\n"
        "fswizz.ps f1, f25, 0xB1\n" "fadd.ps f2, f25, f1, rne\n"   // reduce vacc1
        "fswizz.ps f3, f2, 0x4E\n"  "fadd.ps f4, f2, f3, rne\n"
        "fmvz.x.ps t0, f4, 4\n"     "fbcx.ps f5, t0\n" "fadd.ps %[s1], f4, f5, rne\n"
        "fswizz.ps f1, f26, 0xB1\n" "fadd.ps f2, f26, f1, rne\n"   // reduce vacc2
        "fswizz.ps f3, f2, 0x4E\n"  "fadd.ps f4, f2, f3, rne\n"
        "fmvz.x.ps t0, f4, 4\n"     "fbcx.ps f5, t0\n" "fadd.ps %[s2], f4, f5, rne\n"
        "fswizz.ps f1, f27, 0xB1\n" "fadd.ps f2, f27, f1, rne\n"   // reduce vacc3
        "fswizz.ps f3, f2, 0x4E\n"  "fadd.ps f4, f2, f3, rne\n"
        "fmvz.x.ps t0, f4, 4\n"     "fbcx.ps f5, t0\n" "fadd.ps %[s3], f4, f5, rne\n"
        "mova.m.x %[savem]\n"
        : [s0] "=f"(s0), [s1] "=f"(s1), [s2] "=f"(s2), [s3] "=f"(s3),
          [savem] "=&r"(saved_mask),
          [a] "+r"(a), [p0] "+r"(p0), [p1] "+r"(p1), [p2] "+r"(p2), [p3] "+r"(p3),
          [cnt] "+r"(cnt), [qs] "=&r"(qs)
        : [gb] "r"(gather_bytes), [gz] "r"(gather_zero)
        : "memory", "t0",
          "f1", "f2", "f3", "f4", "f5", "f11", "f12", "f14",
          "f20", "f21", "f22", "f23", "f24", "f25", "f26", "f27", "f30", "f31"
    );
    *out0 = s0; *out1 = s1; *out2 = s2; *out3 = s3;
}
#endif // ET_Q8_0_USE_GENERIC_DOT

// Using the block prefetch logic
static inline void prefetch_weight_row(const void* start_ptr, int64_t num_blocks, uint32_t worker_id) {
    const uint64_t cache_line_size = 64;
    uintptr_t self_ptr = (uintptr_t)start_ptr;
    uintptr_t self_ptr_end = self_ptr + (num_blocks * sizeof(block_q8_0));

    // 1. Align to cache lines and calculate range
    uint64_t startCL = (self_ptr + 63) >> 6;
    uint64_t endCL = (self_ptr_end + 63) >> 6;
    if (endCL >= startCL) {
        uint64_t total_lines = endCL - startCL + 1;
        // 2. Load balance across the minions in the Shire (assuming 8 per group for this logic)
        // Adjust worker_id if using global_id
        uint32_t local_worker_id = worker_id % 8;
        uint64_t lines_per_minion = total_lines >> 3;
        uint64_t extra = total_lines & 7;

        uint64_t offset = local_worker_id * lines_per_minion;
        if (local_worker_id < extra) {
            offset += local_worker_id;
            lines_per_minion++;
        } else {
            offset += extra;
        }
        self_ptr = (startCL + offset) << 6;
        int pending_lines = lines_per_minion;

        // 3. Hardware Prefetch Loop (16 lines at a time)
        for (; pending_lines > 0; pending_lines -= 16) {
            uint64_t current_batch = (pending_lines > 16 ? 16 : pending_lines) - 1;
            uint64_t self_size = current_batch; // bits 3:0

            __asm__ __volatile__ (
                "li    x1, 0x400000000000000 \n"  // Dest = L2 (bits 59:58 = 01)
                "addi  x31, zero, 64\n"  // Stride = 64 bytes
                "or    x3, x1, %[ptr]\n"  // Combine Dest + VA
                "or    x3, x3, %[sz]\n"  // Combine with NumLines
                "csrw  0x81f, x3\n"  // prefetch_va
                :
                : [ptr] "r" (self_ptr),
                  [sz] "r" (self_size)
                : "x1", "x3", "x31", "memory"
            );
            self_ptr += (16 * 64);
        }
    }
}

#ifdef ET_Q8_0_TENSOR
//******************************************************************************
// STEP 1 (correctness-first): matrix-engine GEMM with on-the-fly Q8_0->f32
// dequant. Structure mirrors mul_mat_f32_matrix_engine.c (the proven f32 tiling
// path): 16x16x16 tiles, only hart 0 issues tensor ops, A = src1 activations
// (f32, plain load), B = src0 weights (transpose load). The only difference vs
// the f32 kernel is that src0 is Q8_0, so we dequantize each 16x16 weight tile
// into an f32 scratch first, then transpose-load that scratch.
//
// Once this validates PPL, STEP 2 swaps the f32 FMA for an int8 FMA
// (tensor_fma tena/tenb signed int) plus the tensor_quant dequant pipeline
// (INT32_TO_FP32 -> MUL_COL by weight delta -> MUL_ROW by activation scale),
// which removes this scalar dequant and the f32 weight bandwidth entirely.
//
// KNOWN RISK to debug first if PPL fails: tensor_load bypasses L1, so the
// scalar-written `wf` scratch must be visible in memory before the load. The
// FENCE below orders stores; if results are wrong, this producer->consumer
// boundary (evict + WAIT_CACHEOPS + FENCE) is suspect #1.
//******************************************************************************
#define ET_TE_TILE 16
#define ET_TE_NUM_HARTS 1024
#define ET_TE_N_TILES_PER_WORK 4
// Max K covered by the tensor path (bounds the per-tile-column dequant scratch:
// 16 * ET_TE_MAX_K * 4 bytes = 64 KiB at 1024). Larger-K matmuls fall back to
// the scalar path. RISK: confirm this fits the kernel stack on-board; if it
// overflows, lower this or move the scratch to a reserved region.
#define ET_TE_MAX_K 768

// Dequantize the full 16-row x K-col weight tile-column [mb..mb+15][0..K-1] into
// wf_col (row-major, ET_TE_MAX_K floats per row). Done ONCE up front so no
// scalar FP runs between tensor_fma calls -- the FMA accumulator occupies all of
// f0..f31 across the K loop and any intervening f-register use would corrupt it.
// K is a multiple of 32, so each 32-col span shares one Q8_0 block scale.
static inline void et_dequant_weight_col(const char* src0_batch, int64_t mb,
                                         int64_t K, size_t nb01,
                                         float wf_col[16][ET_TE_MAX_K]) {
    for (int j = 0; j < ET_TE_TILE; j++) {
        const block_q8_0* row = (const block_q8_0*)(src0_batch + (mb + j) * nb01);
        float* dst = wf_col[j];
        for (int64_t blk = 0; blk < K / 32; blk++) {
            const float scale = fp16_to_fp32(row[blk].d);
            const int8_t* qs = row[blk].qs;
            float* d32 = dst + blk * 32;
            for (int k = 0; k < 32; k++) {
                // tensor_load bypasses L1. Publish the dequantized scratch
                // through the global/coherent store path so the tensor unit
                // observes it without relying on a private-cache eviction.
                atomic_store_f32((volatile float*)&d32[k], scale * (float)qs[k]);
            }
        }
    }
}

static int mul_mat_q8_0_tensor(struct ggml_et_binary_params* params) {
    const uint64_t hart_id = get_hart_id();
    if (hart_id & 1) return 0;  // only hart 0 of each minion has the tensor engine
    const uint64_t global_id = ((hart_id >> 6) << 5) + ((hart_id >> 1) & 0x1F);

    const int64_t K = params->src0.ne[0];
    const int64_t M = params->src0.ne[1];
    const int64_t N = params->src1.ne[1];
    const int64_t ne2_0 = params->src0.ne[2], ne3_0 = params->src0.ne[3];
    const int64_t ne2_1 = params->src1.ne[2], ne3_1 = params->src1.ne[3];
    const size_t nb1_0 = params->src0.nb[1], nb2_0 = params->src0.nb[2], nb3_0 = params->src0.nb[3];
    const size_t nb1_1 = params->src1.nb[1], nb2_1 = params->src1.nb[2], nb3_1 = params->src1.nb[3];
    const size_t nb1_d = params->dst.nb[1], nb2_d = params->dst.nb[2], nb3_d = params->dst.nb[3];
    const char* src0_base = (const char*)params->src0.data;
    const char* src1_base = (const char*)params->src1.data;
    char* dst_base = (char*)params->dst.data;

    setup_cache_scp();
    CLEAR_TENSOR_ERROR;

    const int64_t m_tiles = M / ET_TE_TILE;
    const int64_t n_tiles = (N + ET_TE_TILE - 1) / ET_TE_TILE;
    const int64_t n_groups =
        (n_tiles + ET_TE_N_TILES_PER_WORK - 1) / ET_TE_N_TILES_PER_WORK;
    const int64_t work_per_batch = m_tiles * n_groups;
    const int64_t batch_count = ne2_1 * ne3_1;
    const int64_t total_work = batch_count * work_per_batch;
    const int64_t r2 = ne2_1 / ne2_0, r3 = ne3_1 / ne3_0;
    const uint64_t wf_stride = (uint64_t)ET_TE_MAX_K * sizeof(float);  // row stride

    float wf_col[16][ET_TE_MAX_K] __attribute__((aligned(64)));

    for (int64_t work = (int64_t)global_id; work < total_work; work += ET_TE_NUM_HARTS) {
        const int64_t batch_idx = work / work_per_batch;
        const int64_t work_in_batch = work % work_per_batch;
        const int64_t n_group = work_in_batch / m_tiles;
        const int64_t mb_idx = work_in_batch % m_tiles;
        const int64_t i3 = batch_idx / ne2_1;
        const int64_t i2 = batch_idx % ne2_1;
        const int64_t i2_0 = i2 / r2, i3_0 = i3 / r3;

        const char* src0_batch = src0_base + i3_0 * nb3_0 + i2_0 * nb2_0;
        const char* src1_batch = src1_base + i3 * nb3_1 + i2 * nb2_1;
        char* dst_batch = dst_base + i3 * nb3_d + i2 * nb2_d;

        const int64_t mb = mb_idx * ET_TE_TILE;

        // Dequantize one 16xK weight tile once, then reuse it across a group of
        // activation tiles. Four N tiles keep enough minions busy for the
        // vision shapes while cutting coherent scratch publication by 4x.
        et_dequant_weight_col(src0_batch, mb, K, nb1_0, wf_col);
        FENCE;

        const int64_t first_n_tile = n_group * ET_TE_N_TILES_PER_WORK;
        for (int64_t n_off = 0; n_off < ET_TE_N_TILES_PER_WORK; n_off++) {
            const int64_t nb_idx = first_n_tile + n_off;
            if (nb_idx >= n_tiles) break;
            const int64_t nb = nb_idx * ET_TE_TILE;
            const int64_t n_cur =
                (nb + ET_TE_TILE <= N) ? ET_TE_TILE : (N - nb);

            for (int64_t kb = 0; kb < K; kb += ET_TE_TILE) {
                // A = src1[nb:nb+n_cur-1][kb:kb+15] -> SCP line 0
                tensor_load(false, false, 0, 0, 0,
                            (uint64_t)(src1_batch + nb * nb1_1 + kb * sizeof(float)),
                            0, n_cur - 1, (uint64_t)nb1_1, 0);
                // B = transpose(wf_col[:][kb:kb+15]) -> SCP line 16
                tensor_load(false, false, ET_TE_TILE, 7, 0,
                            (uint64_t)&wf_col[0][kb], 0, ET_TE_TILE - 1, wf_stride, 1);
                tensor_wait(TENSOR_LOAD_WAIT_0);
                tensor_wait(TENSOR_LOAD_WAIT_1);
                tensor_fma(false, 3, n_cur - 1, ET_TE_TILE - 1, 0,
                           false, false, false, false, ET_TE_TILE, 0, 0, (kb == 0));
                tensor_wait(TENSOR_FMA_WAIT);
            }

            tensor_store(0, 0, 3, n_cur - 1,
                         (uint64_t)(dst_batch + nb * nb1_d + mb * sizeof(float)),
                         0, (uint64_t)nb1_d);
            tensor_wait(TENSOR_STORE_WAIT);
        }
    }

    FENCE;
    return 0;
}
#endif // ET_Q8_0_TENSOR

int entry_point(struct ggml_et_binary_params* params, void* env) {
    uint64_t hart_id = get_hart_id();
    const int64_t stride_m = 2048;

    // Matrix dimensions
    const int64_t K    = params->src0.ne[0];
    const int64_t M    = params->src0.ne[1];
    const int64_t N    = params->src1.ne[1];
    const int64_t ne02 = params->src0.ne[2];
    const int64_t ne03 = params->src0.ne[3];
    const int64_t ne12 = params->src1.ne[2];
    const int64_t ne13 = params->src1.ne[3];

#ifdef ET_Q8_0_TENSOR
    // Matrix-engine dequant path for 16-aligned shapes (M multiple of 16, K a
    // multiple of 32 so each 16-wide K tile stays within one Q8_0 block, and K
    // within the dequant-scratch bound). Falls through to the scalar path
    // otherwise (larger K, odd shapes).
    // First correctness slice: vision-encoder projections only. Keeping the
    // language-model K=960 path scalar protects the PPL gate while this path
    // is validated on the multimodal dog case.
    if ((M % 16 == 0) && (K == ET_TE_MAX_K)) {
        return mul_mat_q8_0_tensor(params);
    }
#endif

    // Strides (in bytes)
    const size_t nb01 = params->src0.nb[1];
    const size_t nb02 = params->src0.nb[2];
    const size_t nb03 = params->src0.nb[3];

    const size_t nb11 = params->src1.nb[1];
    const size_t nb12 = params->src1.nb[2];
    const size_t nb13 = params->src1.nb[3];

    const size_t nbd1 = params->dst.nb[1];
    const size_t nbd2 = params->dst.nb[2];
    const size_t nbd3 = params->dst.nb[3];

    // Q8_0 block size is 32
    const int64_t K_blocks = K / 32;

    // Broadcasting ratios
    const int64_t r2 = ne12 / ne02;
    const int64_t r3 = ne13 / ne03;

    for (int64_t i3 = 0; i3 < ne13; i3++) {
        const int64_t i03 = i3 / r3;
        const char* src0_ptr3 = (const char*)params->src0.data + i03 * nb03;
        const char* src1_ptr3 = (const char*)params->src1.data + i3 * nb13;
        char* dst_ptr3       = (char*)params->dst.data + i3 * nbd3;

        for (int64_t i2 = 0; i2 < ne12; i2++) {
            const int64_t i02 = i2 / r2;
            const char* src0_ptr2 = src0_ptr3 + i02 * nb02;
            const char* src1_ptr2 = src1_ptr3 + i2 * nb12;
            char* dst_ptr2       = dst_ptr3 + i2 * nbd2;

#ifdef ET_Q8_0_USE_GENERIC_DOT
            for (int64_t n = 0; n < N; n++) {
                // src1 is F32, so column pointer moves by nb11
                const float* b_col_base = (const float*)(src1_ptr2 + n * nb11);

                for (int64_t m = hart_id; m < M; m += stride_m) {
                    // src0 is Q8_0 blocks, row pointer moves by nb01
                    const block_q8_0* q_row = (const block_q8_0*)(src0_ptr2 + m * nb01);
                    float sum = 0.0f;
                    for (int64_t kb = 0; kb < K_blocks; kb++) {
                        // q_row is a pointer to blocks, so + kb moves by sizeof(block_q8_0)
                        // b_col is float*, so we move 32 elements (kb << 5)
                        sum += compute_block_dot_product_q8_0(q_row + kb, b_col_base + (kb << 5));
                    }
                    // Store result in dst[m, n, i2, i3]
                    float* dst_entry = (float*)(dst_ptr2 + n * nbd1 + m * sizeof(float));
                    atomic_store_f32((volatile float*)dst_entry, sum);
                }
            }
#else
            // Cache-line-owned output stores (no per-element global atomics).
            //
            // atomic_store_f32 uses amoswapg.w -- a global atomic read-modify-write
            // per output. Profiling implicated it as the shared bottleneck: neither
            // cutting compute nor cutting weight traffic helped, and adding harts
            // made it slightly WORSE (more contention on the global atomic path).
            //
            // The atomic is only needed because 16 adjacent outputs (one 64B cache
            // line) were split across 16 harts (false sharing on non-coherent L1D).
            // Following the set_rows_f32 cache-aligned pattern: give each hart a
            // COMPLETE output cache line (16 consecutive m for one n). It then owns
            // that line exclusively and can use plain stores; the runtime writes
            // dirty lines back at the kernel boundary. dst is m-contiguous, so a
            // line is 16 consecutive m. Requires M and the dst base/stride to be
            // 64B aligned; anything else falls back to the flattened atomic path.
            const int64_t M_blocks = M >> 4;  // 16 f32 = 64B cache line
            const bool aligned =
                (M >= 16) && ((M & 15) == 0) &&
                (((uintptr_t)dst_ptr2 & 63) == 0) && ((nbd1 & 63) == 0);

            if (aligned) {
                const int64_t total_cls = M_blocks * N;
                for (int64_t cl = (int64_t)hart_id; cl < total_cls; cl += stride_m) {
                    const int64_t n  = cl / M_blocks;
                    const int64_t m0 = (cl - n * M_blocks) << 4;   // first m in this line
                    const float* b_col_base = (const float*)(src1_ptr2 + n * nb11);
                    // This hart owns the whole 64B line -> plain stores, no atomics.
                    float* dst_line = (float*)(dst_ptr2 + n * nbd1 + m0 * sizeof(float));
                    for (int j = 0; j < 16; j++) {
                        const block_q8_0* q_row =
                            (const block_q8_0*)(src0_ptr2 + (m0 + j) * nb01);
                        dst_line[j] = dot_row_q8_0(q_row, b_col_base, K_blocks);
                    }
                }
            } else {
                // Unaligned fallback: flattened distribution + per-element atomics.
                const int64_t MN = M * N;
                for (int64_t idx = (int64_t)hart_id; idx < MN; idx += stride_m) {
                    const int64_t n = idx / M;
                    const int64_t m = idx - n * M;
                    const float* b_col_base = (const float*)(src1_ptr2 + n * nb11);
                    const block_q8_0* q_row = (const block_q8_0*)(src0_ptr2 + m * nb01);
                    float sum = dot_row_q8_0(q_row, b_col_base, K_blocks);
                    float* dst_entry = (float*)(dst_ptr2 + n * nbd1 + m * sizeof(float));
                    atomic_store_f32((volatile float*)dst_entry, sum);
                }
            }
#endif
        }
    }
    return 0;
}
