// Contiguous F32 unary operations required by vision encoders.

#include <stdint.h>
#include "ggml_tensor.h"
#include "platform.h"

enum { GGML_UNARY_OP_GELU = 8 };

struct ggml_et_unary_params {
    struct ggml_tensor src0;
    struct ggml_tensor dst;
    int32_t unary_op;
};

int entry_point(struct ggml_et_unary_params * params, void * env) {
    kernel_environment_t * kernel_env = (kernel_environment_t *) env;
    if (!kernel_env || !params || ((uint64_t) params & 0x7) != 0) {
        return -1;
    }

    const int thread_id = get_relative_thread_id(kernel_env->shire_mask);
    const int num_threads = get_num_threads(kernel_env->shire_mask);
    if (thread_id < 0) {
        return 0;
    }

    const struct ggml_tensor * src = &params->src0;
    const struct ggml_tensor * dst = &params->dst;
    if (params->unary_op != GGML_UNARY_OP_GELU ||
        src->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32 ||
        !src->data || !dst->data || src->nb[0] != sizeof(float) || dst->nb[0] != sizeof(float)) {
        return -1;
    }

    const int64_t elements = dst->ne[0] * dst->ne[1] * dst->ne[2] * dst->ne[3];
    if (elements <= 0 || elements % 16 != 0) {
        return -1;
    }

    const float * x = (const float *) src->data;
    float * y = (float *) dst->data;
    const int64_t cache_lines = elements / 16;
    const int64_t lines_per_thread = (cache_lines + num_threads - 1) / num_threads;
    const int64_t first = thread_id * lines_per_thread;
    int64_t last = first + lines_per_thread;
    if (last > cache_lines) {
        last = cache_lines;
    }
    if (first >= last) {
        return 0;
    }

    const float one = 1.0f;
    const float coef = 0.044715f;
    const float sqrt_2_over_pi = 0.7978845608028654f;
    const float two_log2e = 2.8853900817779268f;
    unsigned long saved_mask;
    __asm__ volatile("mova.x.m %0" : "=r"(saved_mask));
    __asm__ volatile("mov.m.x m0, x0, 0xFF");

    for (int64_t line = first; line < last; ++line) {
        for (int offset = 0; offset < 16; offset += 8) {
            const int64_t i = line * 16 + offset;
            __asm__ volatile(
                "flw.ps f10, %[x]\n"
                "fbc.ps f20, %[one]\n"
                "fbc.ps f21, %[coef]\n"
                "fbc.ps f22, %[sqrt]\n"
                "fbc.ps f23, %[two_log2e]\n"
                "fmul.ps f11, f10, f10\n"
                "fmadd.ps f12, f21, f11, f20\n"
                "fmul.ps f13, f22, f10\n"
                "fmul.ps f13, f13, f12\n"
                "fmul.ps f14, f13, f23\n"
                "fexp.ps f14, f14\n"
                "fadd.ps f15, f14, f20\n"
                "frcp.ps f15, f15\n"
                "fsub.ps f15, f20, f15\n"
                "fmul.ps f16, f10, f15\n"
                "fsw.ps f16, %[y]\n"
                : [y] "=m"(*(float (*)[8]) &y[i])
                : [x] "m"(*(const float (*)[8]) &x[i]),
                  [one] "m"(one), [coef] "m"(coef),
                  [sqrt] "m"(sqrt_2_over_pi), [two_log2e] "m"(two_log2e)
                : "f10", "f11", "f12", "f13", "f14", "f15", "f16",
                  "f20", "f21", "f22", "f23");
        }
    }
    __asm__ volatile("mova.m.x %0" : : "r"(saved_mask));
    return 0;
}
