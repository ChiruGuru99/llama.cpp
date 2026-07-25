
// Fused RMS Norm + MUL F32 Kernel

#include <stdint.h>
#include <string.h>
#include <assert.h>
#include "ggml_tensor.h"
#include "platform.h"
#include "math_fp.h"

// Fused RMS norm + MUL kernel parameters structure
struct ggml_et_rms_norm_mul_params {
    struct ggml_tensor src0;  // F32 input tensor (to be normalized)
    struct ggml_tensor src1;  // F32 weights tensor (element-wise multiply)
    struct ggml_tensor dst;   // F32 output tensor
    float eps;                // Epsilon for numerical stability
};

int entry_point(struct ggml_et_rms_norm_mul_params* params, void* env) {
    kernel_environment_t* kernel_env = (kernel_environment_t*)env;

    if (!kernel_env) {
        return -1;
    }

    int thread_id = get_relative_thread_id(kernel_env->shire_mask);
    int num_threads = get_num_threads(kernel_env->shire_mask);

    if (thread_id < 0) {
        return 0;
    }

    if (params == 0 || ((uint64_t)params & 0x7) != 0) {
        return -1; // Invalid pointer
    }

    struct ggml_tensor* src0 = &params->src0;
    struct ggml_tensor* src1 = &params->src1;
    struct ggml_tensor* dst = &params->dst;
    float eps = params->eps;

    if (src0->type != GGML_TYPE_F32 || src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        return -1; // Unsupported type combination
    }

    float* src0_data = (float*)src0->data;
    float* src1_data = (float*)src1->data;
    float* dst_data = (float*)dst->data;

    if (!src0_data || !src1_data || !dst_data) {
        return -1; // Null data pointer
    }

    if (eps < 0.0f) {
        return -1; // Invalid epsilon
    }

    const int64_t ne0 = dst->ne[0];  // Inner dimension (row size)
    const int64_t ne1 = dst->ne[1];  // Dimension 1
    const int64_t ne2 = dst->ne[2];  // Dimension 2
    const int64_t ne3 = dst->ne[3];  // Dimension 3

    // Get dst strides (in bytes)
    const size_t nb0 = dst->nb[0], nb1 = dst->nb[1], nb2 = dst->nb[2], nb3 = dst->nb[3];

    // Get src0 strides (in bytes)
    const size_t nb00 = src0->nb[0], nb01 = src0->nb[1], nb02 = src0->nb[2], nb03 = src0->nb[3];

    // Get src1 (weights) strides (in bytes) — supports broadcasting in dims 1,2,3
    const size_t nb10 = src1->nb[0], nb11 = src1->nb[1], nb12 = src1->nb[2], nb13 = src1->nb[3];

    // Verify that src0 and dst have same shape (required for RMS norm)
    if (src0->ne[0] != ne0 || src0->ne[1] != ne1 || src0->ne[2] != ne2 || src0->ne[3] != ne3) {
        return -1; // Shape mismatch
    }

    // Specialized two-hart path for one contiguous 2048-element row
    if (ne0 == 2048 && ne1 == 1 && ne2 == 1 && ne3 == 1 &&
        src1->ne[0] == 2048 && src1->ne[1] == 1 && src1->ne[2] == 1 && src1->ne[3] == 1 &&
        nb00 == 4 && nb10 == 4 && nb0 == 4) {
        if (thread_id >= 2) {
            return 0;
        }

        const int is_hart1 = get_hart_id() & 1;
        const int32_t start = is_hart1 ? 1024 : 0;
        const float* src_ptr = src0_data + start;
        const float* wgt_ptr = src1_data + start;
        float* dst_ptr = dst_data + start;

        const int64_t local_minion = (get_hart_id() >> 1) & 0x1F;
        volatile float* l2scp_slot =
            (volatile float*)et_shire_l2scp_local(local_minion * 64);

        float zero = 0.0f;
        __asm__ volatile("fbc.ps f10, %[z]\n" : : [z] "m"(zero) : "f10");

        for (int32_t i0 = 0; i0 < 1024; i0 += 8) {
            __asm__ volatile(
                "flw.ps f11, %[x_vec]\n"
                "fmadd.ps f10, f11, f11, f10\n"
                :
                : [x_vec] "m"(*(const float(*)[8])&src_ptr[i0])
                : "f10", "f11"
            );
        }

        float partial;
        __asm__ __volatile__(
            "fswizz.ps f1, f10, 0xB1 \n\t"
            "fadd.ps   f2, f10, f1, rne \n\t"
            "fswizz.ps f3, f2, 0x4E \n\t"
            "fadd.ps   f4, f2, f3, rne \n\t"
            "fmvz.x.ps t0, f4, 4 \n\t"
            "fbcx.ps   f5, t0 \n\t"
            "fadd.ps   %[vout], f4, f5, rne \n\t"
            : [vout] "=f" (partial)
            :: "t0", "f1", "f2", "f3", "f4", "f5"
        );

        float scale;
        if (is_hart1) {
            *l2scp_slot = partial;
            FENCE;
            flush_to_l2((const void*)l2scp_slot, 1, 64);
            WAIT_CACHEOPS;
            et_sem_post(ET_BARRIER_MINION);
            et_sem_wait(ET_BARRIER_MINION);

            // Hart 0 replaced our partial with the scale. Discard the cached
            // partial so this load fetches hart 0's updated value from L2 SCP.
            FENCE;
            evict_to_l2((const void*)l2scp_slot, 1, 64);
            WAIT_CACHEOPS;
            FENCE;
            scale = *l2scp_slot;
        } else {
            et_sem_wait(ET_BARRIER_MINION);

            // This slot may still contain the scale from a previous launch.
            // Discard the local copy before reading hart 1's new partial.
            FENCE;
            evict_to_l2((const void*)l2scp_slot, 1, 64);
            WAIT_CACHEOPS;
            FENCE;

            const float sum = partial + *l2scp_slot;
            const float mean = et_fdiv(sum, 2048.0f);
            scale = et_powf(mean + eps, -0.5f);
            *l2scp_slot = scale;
            FENCE;
            flush_to_l2((const void*)l2scp_slot, 1, 64);
            WAIT_CACHEOPS;
            et_sem_post(ET_BARRIER_MINION);
        }

        if (!(scale > 0.0f)) {
            return -1;
        }

        for (int32_t i0 = 0; i0 < 1024; i0 += 8) {
            __asm__ volatile(
                "flw.ps f12, %[x_vec]\n"
                "fbc.ps f13, %[scale_ptr]\n"
                "fmul.ps f14, f12, f13\n"
                "flw.ps f15, %[w_vec]\n"
                "fmul.ps f14, f14, f15\n"
                "fsw.ps f14, %[result]\n"
                : [result] "=m"(*(float(*)[8])&dst_ptr[i0])
                : [x_vec] "m"(*(const float(*)[8])&src_ptr[i0]),
                  [scale_ptr] "m"(scale),
                  [w_vec] "m"(*(const float(*)[8])&wgt_ptr[i0])
                : "f12", "f13", "f14", "f15"
            );
        }

        return 0;
    }

    // RMS norm processes rows independently
    // Parallelize across rows using simple striding
    for (int64_t i3 = 0; i3 < ne3; i3++) {
        for (int64_t i2 = 0; i2 < ne2; i2++) {
            for (int64_t i1 = thread_id; i1 < ne1; i1 += num_threads) {

            // Calculate base pointers for this row using stride-based addressing
            const float* src_ptr = (const float*)((const char*)src0_data + i3*nb03 + i2*nb02 + i1*nb01);
            float* dst_ptr = (float*)((char*)dst_data + i3*nb3 + i2*nb2 + i1*nb1);

            // Weights pointer with broadcasting support (dims 1,2,3 may be size 1)
            const float* wgt_ptr = (const float*)((const char*)src1_data
                + (i3 % src1->ne[3])*nb13
                + (i2 % src1->ne[2])*nb12
                + (i1 % src1->ne[1])*nb11);

            // Step 1: Compute sum of squares for this row using 8-wide vectors
            // ne0 is guaranteed to be a multiple of 16 (cache-aligned)

            // Zero the accumulator register
            float zero = 0.0f;
            __asm__ volatile("fbc.ps f10, %[z]\n" : : [z] "m"(zero) : "f10");

            for (int32_t i0 = 0; i0 < (int32_t)ne0; i0 += 8) {
                __asm__ volatile(
                    "flw.ps f11, %[x_vec]\n"            // Load 8 input values
                    "fmadd.ps f10, f11, f11, f10\n"     // acc += x * x (fused multiply-add)
                    :
                    : [x_vec] "m"(*(const float(*)[8])&src_ptr[i0])
                    : "f10", "f11"
                );
            }

            // Horizontal sum of 8 accumulated values in f10
            float sum;
            __asm__ __volatile__(
                "fswizz.ps f1, f10, 0xB1 \n\t"         // Swaps: e0<->e1 and e2<->e3
                "fadd.ps   f2, f10, f1, rne \n\t"
                "fswizz.ps f3, f2, 0x4E \n\t"           // Swaps: e0,e1 <-> e2,e3
                "fadd.ps   f4, f2, f3, rne \n\t"
                "fmvz.x.ps t0, f4, 4 \n\t"              // Move upper 128b half to scalar
                "fbcx.ps   f5, t0 \n\t"                  // Broadcast to vector
                "fadd.ps   %[vout], f4, f5, rne \n\t"
                : [vout] "=f" (sum)
                :: "t0", "f1", "f2", "f3", "f4", "f5"
            );

            // Step 2: Compute mean of squares and scale factor
            const float mean = et_fdiv(sum, (float)(int32_t)ne0);
            const float scale = et_powf(mean + eps, -0.5f);

            // Numerical stability check
            if (!(scale > 0.0f)) {
                return -1; // Invalid scale factor
            }

            // Step 3: Apply scaling and multiply by weights using 8-wide vectors
            // Fused: dst[i] = src[i] * scale * weights[i]
            for (int32_t i0 = 0; i0 < (int32_t)ne0; i0 += 8) {
                __asm__ volatile(
                    "flw.ps f12, %[x_vec]\n"            // Load 8 input values
                    "fbc.ps f13, %[scale_ptr]\n"        // Broadcast scale to all 8 elements
                    "fmul.ps f14, f12, f13\n"           // x * scale (8-wide)
                    "flw.ps f15, %[w_vec]\n"            // Load 8 weight values
                    "fmul.ps f14, f14, f15\n"           // (x * scale) * weights (8-wide)
                    "fsw.ps f14, %[result]\n"           // Store 8 results

                    : [result] "=m"(*(float(*)[8])&dst_ptr[i0])
                    : [x_vec] "m"(*(const float(*)[8])&src_ptr[i0]),
                      [scale_ptr] "m"(scale),
                      [w_vec] "m"(*(const float(*)[8])&wgt_ptr[i0])
                    : "f12", "f13", "f14", "f15"
                );
            }
            }
        }
    }

    return 0;
}
