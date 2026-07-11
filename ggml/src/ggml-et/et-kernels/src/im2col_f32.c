// General F32 2D im2col used by CLIP/SigLIP patch embeddings.

#include <stdint.h>
#include "ggml_tensor.h"
#include "platform.h"

struct ggml_et_im2col_params {
    struct ggml_tensor src0;
    struct ggml_tensor src1;
    struct ggml_tensor dst;
    int32_t s0;
    int32_t s1;
    int32_t p0;
    int32_t p1;
    int32_t d0;
    int32_t d1;
    int32_t is_2d;
};

int entry_point(struct ggml_et_im2col_params * params, void * env) {
    kernel_environment_t * kernel_env = (kernel_environment_t *) env;
    if (!kernel_env || !params || ((uint64_t) params & 0x7) != 0) {
        return -1;
    }

    const int thread_id = get_relative_thread_id(kernel_env->shire_mask);
    const int num_threads = get_num_threads(kernel_env->shire_mask);
    if (thread_id < 0) {
        return 0;
    }

    const struct ggml_tensor * kernel = &params->src0;
    const struct ggml_tensor * image = &params->src1;
    const struct ggml_tensor * dst = &params->dst;
    if (params->is_2d != 1 || image->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32 ||
        !image->data || !dst->data || image->nb[0] != sizeof(float) || dst->nb[0] != sizeof(float)) {
        return -1;
    }

    const int64_t kw = kernel->ne[0];
    const int64_t kh = kernel->ne[1];
    const int64_t channels = image->ne[2];
    const int64_t iw = image->ne[0];
    const int64_t ih = image->ne[1];
    const int64_t batches = image->ne[3];
    const int64_t ow = dst->ne[1];
    const int64_t oh = dst->ne[2];
    const int64_t patch = channels * kh * kw;
    const int64_t elements = patch * ow * oh * batches;
    if (kw <= 0 || kh <= 0 || kernel->ne[2] != channels ||
        dst->ne[0] != patch || dst->ne[3] != batches || elements % 16 != 0) {
        return -1;
    }

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

    float * output = (float *) dst->data;
    for (int64_t line = first; line < last; ++line) {
        const int64_t begin = line * 16;
        for (int offset = 0; offset < 16; ++offset) {
            const int64_t index = begin + offset;
            const int64_t patch_index = index / patch;
            const int64_t kernel_index = index - patch_index * patch;
            const int64_t batch = patch_index / (oh * ow);
            const int64_t spatial = patch_index - batch * oh * ow;
            const int64_t out_y = spatial / ow;
            const int64_t out_x = spatial - out_y * ow;
            const int64_t channel = kernel_index / (kh * kw);
            const int64_t kernel_spatial = kernel_index - channel * kh * kw;
            const int64_t kernel_y = kernel_spatial / kw;
            const int64_t kernel_x = kernel_spatial - kernel_y * kw;
            const int64_t in_x = out_x * params->s0 + kernel_x * params->d0 - params->p0;
            const int64_t in_y = out_y * params->s1 + kernel_y * params->d1 - params->p1;

            float value = 0.0f;
            if (in_x >= 0 && in_x < iw && in_y >= 0 && in_y < ih) {
                const char * address = (const char *) image->data +
                    batch * image->nb[3] + channel * image->nb[2] +
                    in_y * image->nb[1] + in_x * image->nb[0];
                value = *(const float *) address;
            }
            output[index] = value;
        }
    }
    return 0;
}
