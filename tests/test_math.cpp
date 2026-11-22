#include "mini_llama/nn.h"
#include "mini_llama/sampler.h"

#include <cmath>
#include <cstdio>
#include <vector>

namespace {

bool close(float a, float b, float tolerance = 1e-5f) {
    return std::fabs(a - b) <= tolerance;
}

bool test_rmsnorm() {
    const float x[] = {3.0f, 4.0f};
    const float weight[] = {1.0f, 2.0f};
    float out[2];
    mini_llama::nn::rmsnorm(out, x, weight, 2);

    const float scale = 1.0f / std::sqrt((9.0f + 16.0f) / 2.0f + 1e-5f);
    return close(out[0], 3.0f * scale) && close(out[1], 8.0f * scale);
}

bool test_softmax() {
    float values[] = {0.0f, std::log(2.0f)};
    mini_llama::nn::softmax(values, 2);
    return close(values[0], 1.0f / 3.0f) && close(values[1], 2.0f / 3.0f);
}

bool test_matmul() {
    const float w[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    const float x[] = {7.0f, 8.0f};
    float out[3];
    mini_llama::nn::matmul(out, x, w, 2, 3);
    return close(out[0], 23.0f) && close(out[1], 53.0f) && close(out[2], 83.0f);
}

bool test_rope() {
    float q[] = {0.5f, -1.0f, 2.0f, 0.25f};
    float k[] = {1.0f, 0.5f, -0.5f, 3.0f};
    const float q_original[] = {0.5f, -1.0f, 2.0f, 0.25f};
    const float k_original[] = {1.0f, 0.5f, -0.5f, 3.0f};

    mini_llama::nn::rope(q, k, 0, 4, 4, 4);
    if (!close(q[0], q_original[0]) || !close(k[3], k_original[3])) {
        return false;
    }

    const float q_energy = q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3];
    const float q_original_energy =
        q_original[0] * q_original[0] + q_original[1] * q_original[1]
        + q_original[2] * q_original[2] + q_original[3] * q_original[3];
    const float k_energy = k[0] * k[0] + k[1] * k[1] + k[2] * k[2] + k[3] * k[3];
    const float k_original_energy =
        k_original[0] * k_original[0] + k_original[1] * k_original[1]
        + k_original[2] * k_original[2] + k_original[3] * k_original[3];

    mini_llama::nn::rope(q, k, 7, 4, 4, 4);
    const float q_energy_after = q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3];
    const float k_energy_after = k[0] * k[0] + k[1] * k[1] + k[2] * k[2] + k[3] * k[3];

    return close(q_energy, q_original_energy, 1e-4f)
        && close(k_energy, k_original_energy, 1e-4f)
        && close(q_energy_after, q_original_energy, 1e-4f)
        && close(k_energy_after, k_original_energy, 1e-4f);
}

bool test_sampler() {
    {
        mini_llama::Sampler greedy(4, 0.0f, 0.9f, 1);
        float logits[] = {-2.0f, 3.0f, 1.0f, 0.0f};
        if (greedy.sample(logits) != 1) {
            return false;
        }
    }

    float first[8] = {0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f};
    float second[8] = {0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f};
    mini_llama::Sampler a(8, 0.8f, 0.9f, 1234);
    mini_llama::Sampler b(8, 0.8f, 0.9f, 1234);
    for (int i = 0; i < 16; ++i) {
        if (a.sample(first) != b.sample(second)) {
            return false;
        }
    }

    float uniform[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    mini_llama::Sampler narrow(4, 1.0f, 0.01f, 1);
    const int sampled = narrow.sample(uniform);
    return sampled >= 0 && sampled < 4;
}

}  // namespace

int main() {
    const bool ok = test_rmsnorm() && test_softmax() && test_matmul()
        && test_rope() && test_sampler();
    if (!ok) {
        std::fprintf(stderr, "math tests failed\n");
        return 1;
    }
    std::puts("math tests passed");
    return 0;
}
