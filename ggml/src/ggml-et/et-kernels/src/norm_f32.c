// F32 LayerNorm over dimension 0: (x - mean) / sqrt(variance + eps).

#include <stdint.h>
#include "ggml_tensor.h"
#include "platform.h"
#include "math_fp.h"

struct ggml_et_norm_params {
    struct ggml_tensor src0;
    struct ggml_tensor dst;
    float eps;
};

static inline float reduce_f32x8(void) {
    float sum;
    __asm__ __volatile__(
        "fswizz.ps f1, f10, 0xB1 \n\t"
        "fadd.ps   f2, f10, f1, rne \n\t"
        "fswizz.ps f3, f2, 0x4E \n\t"
        "fadd.ps   f4, f2, f3, rne \n\t"
        "fmvz.x.ps t0, f4, 4 \n\t"
        "fbcx.ps   f5, t0 \n\t"
        "fadd.ps   %[sum], f4, f5, rne \n\t"
        : [sum] "=f"(sum)
        :
        : "t0", "f1", "f2", "f3", "f4", "f5");
    return sum;
}

int entry_point(struct ggml_et_norm_params * params, void * env) {
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
    if (src->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32 || params->eps < 0.0f) {
        return -1;
    }
    if (!src->data || !dst->data || src->nb[0] != sizeof(float) || dst->nb[0] != sizeof(float)) {
        return -1;
    }
    for (int i = 0; i < 4; ++i) {
        if (src->ne[i] != dst->ne[i]) {
            return -1;
        }
    }

    const int64_t ne0 = dst->ne[0];
    const int64_t rows = dst->ne[1] * dst->ne[2] * dst->ne[3];
    if (ne0 <= 0 || ne0 % 16 != 0) {
        return -1;
    }

    for (int64_t row = thread_id; row < rows; row += num_threads) {
        const int64_t i1 = row % dst->ne[1];
        const int64_t outer = row / dst->ne[1];
        const int64_t i2 = outer % dst->ne[2];
        const int64_t i3 = outer / dst->ne[2];
        const float * x = (const float *) ((const char *) src->data +
            i1 * src->nb[1] + i2 * src->nb[2] + i3 * src->nb[3]);
        float * y = (float *) ((char *) dst->data +
            i1 * dst->nb[1] + i2 * dst->nb[2] + i3 * dst->nb[3]);

        const float zero = 0.0f;
        __asm__ volatile("fbc.ps f10, %[zero]\n" : : [zero] "m"(zero) : "f10");
        for (int32_t i = 0; i < (int32_t) ne0; i += 8) {
            __asm__ volatile(
                "flw.ps f11, %[x]\n"
                "fadd.ps f10, f10, f11, rne\n"
                :
                : [x] "m"(*(const float (*)[8]) &x[i])
                : "f10", "f11");
        }
        const float row_size = (float) (int32_t) ne0;
        const float mean = et_fdiv(reduce_f32x8(), row_size);

        __asm__ volatile("fbc.ps f10, %[zero]\n" : : [zero] "m"(zero) : "f10");
        for (int32_t i = 0; i < (int32_t) ne0; i += 8) {
            __asm__ volatile(
                "flw.ps f11, %[x]\n"
                "fbc.ps f12, %[mean]\n"
                "fsub.ps f11, f11, f12\n"
                "fmadd.ps f10, f11, f11, f10\n"
                :
                : [x] "m"(*(const float (*)[8]) &x[i]), [mean] "m"(mean)
                : "f10", "f11", "f12");
        }
        const float variance = et_fdiv(reduce_f32x8(), row_size);
        const float scale = et_powf(variance + params->eps, -0.5f);
        if (!(scale > 0.0f)) {
            return -1;
        }

        for (int32_t i = 0; i < (int32_t) ne0; i += 8) {
            __asm__ volatile(
                "flw.ps f11, %[x]\n"
                "fbc.ps f12, %[mean]\n"
                "fbc.ps f13, %[scale]\n"
                "fsub.ps f11, f11, f12\n"
                "fmul.ps f14, f11, f13\n"
                "fsw.ps f14, %[y]\n"
                : [y] "=m"(*(float (*)[8]) &y[i])
                : [x] "m"(*(const float (*)[8]) &x[i]),
                  [mean] "m"(mean), [scale] "m"(scale)
                : "f11", "f12", "f13", "f14");
        }
    }
    return 0;
}
