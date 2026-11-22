#include "mini_llama/nn.h"

#include <cmath>

namespace mini_llama::nn {

void rmsnorm(float* out, const float* x, const float* weight, int size) {
    float sum_squares = 0.0f;
    for (int i = 0; i < size; ++i) {
        sum_squares += x[i] * x[i];
    }
    sum_squares /= static_cast<float>(size);
    sum_squares += 1e-5f;
    const float scale = 1.0f / std::sqrt(sum_squares);
    for (int i = 0; i < size; ++i) {
        out[i] = weight[i] * (scale * x[i]);
    }
}

void softmax(float* x, int size) {
    float max_value = x[0];
    for (int i = 1; i < size; ++i) {
        if (x[i] > max_value) {
            max_value = x[i];
        }
    }

    float sum = 0.0f;
    for (int i = 0; i < size; ++i) {
        x[i] = std::exp(x[i] - max_value);
        sum += x[i];
    }
    for (int i = 0; i < size; ++i) {
        x[i] /= sum;
    }
}

void matmul(float* out, const float* x, const float* w, int n, int d) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int row = 0; row < d; ++row) {
        const float* weights = w + static_cast<std::size_t>(row) * n;
        float sum = 0.0f;
        for (int col = 0; col < n; ++col) {
            sum += weights[col] * x[col];
        }
        out[row] = sum;
    }
}

void rope(float* q, float* k, int pos, int dim, int kv_dim, int head_size) {
    for (int i = 0; i < dim; i += 2) {
        const int head_dim = i % head_size;
        const float freq = 1.0f / std::pow(
            10000.0f, static_cast<float>(head_dim) / static_cast<float>(head_size));
        const float angle = static_cast<float>(pos) * freq;
        const float cos_value = std::cos(angle);
        const float sin_value = std::sin(angle);

        float q0 = q[i];
        float q1 = q[i + 1];
        q[i] = q0 * cos_value - q1 * sin_value;
        q[i + 1] = q0 * sin_value + q1 * cos_value;

        if (i < kv_dim) {
            float k0 = k[i];
            float k1 = k[i + 1];
            k[i] = k0 * cos_value - k1 * sin_value;
            k[i + 1] = k0 * sin_value + k1 * cos_value;
        }
    }
}

}  // namespace mini_llama::nn
