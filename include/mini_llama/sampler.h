#pragma once

#include <cstdint>
#include <vector>

namespace mini_llama {

class Sampler {
public:
    Sampler(int vocab_size, float temperature, float topp, std::uint64_t seed);

    // logits is modified in place when temperature > 0.
    int sample(float* logits);

    std::uint64_t rng_state() const noexcept { return rng_state_; }

private:
    struct ProbIndex {
        float prob;
        int index;
    };

    int sample_argmax(const float* logits) const;
    int sample_mult(const float* probabilities, float coin) const;
    int sample_topp(const float* probabilities, float coin);

    int vocab_size_;
    float temperature_;
    float topp_;
    std::uint64_t rng_state_;
    std::vector<ProbIndex> probindex_;

    std::uint32_t random_u32();
    float random_f32();
};

}  // namespace mini_llama
