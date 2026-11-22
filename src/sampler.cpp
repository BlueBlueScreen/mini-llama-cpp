#include "mini_llama/sampler.h"

#include "mini_llama/nn.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace mini_llama {

Sampler::Sampler(int vocab_size, float temperature, float topp, std::uint64_t seed)
    : vocab_size_(vocab_size),
      temperature_(std::max(temperature, 0.0f)),
      topp_((topp < 0.0f || topp > 1.0f) ? 0.9f : topp),
      rng_state_(seed == 0 ? 0x9E3779B97F4A7C15ULL : seed),
      probindex_(static_cast<std::size_t>(vocab_size > 0 ? vocab_size : 0)) {
    if (vocab_size_ <= 0) {
        throw std::runtime_error("sampler vocab_size must be positive");
    }
}

std::uint32_t Sampler::random_u32() {
    rng_state_ ^= rng_state_ >> 12;
    rng_state_ ^= rng_state_ << 25;
    rng_state_ ^= rng_state_ >> 27;
    return static_cast<std::uint32_t>(
        (rng_state_ * 0x2545F4914F6CDD1DULL) >> 32);
}

float Sampler::random_f32() {
    return static_cast<float>(random_u32() >> 8) / 16777216.0f;
}

int Sampler::sample_argmax(const float* logits) const {
    int best_index = 0;
    float best_value = logits[0];
    for (int i = 1; i < vocab_size_; ++i) {
        if (logits[i] > best_value) {
            best_index = i;
            best_value = logits[i];
        }
    }
    return best_index;
}

int Sampler::sample_mult(const float* probabilities, float coin) const {
    float cdf = 0.0f;
    for (int i = 0; i < vocab_size_; ++i) {
        cdf += probabilities[i];
        if (coin < cdf) {
            return i;
        }
    }
    return vocab_size_ - 1;
}

int Sampler::sample_topp(const float* probabilities, float coin) {
    if (vocab_size_ == 1) {
        return 0;
    }

    const float cutoff = (1.0f - topp_) / static_cast<float>(vocab_size_ - 1);

    std::size_t candidate_count = 0;
    for (int i = 0; i < vocab_size_; ++i) {
        if (probabilities[i] >= cutoff) {
            probindex_[candidate_count].index = i;
            probindex_[candidate_count].prob = probabilities[i];
            ++candidate_count;
        }
    }

    if (candidate_count == 0) {
        return sample_argmax(probabilities);
    }

    std::sort(probindex_.begin(),
              probindex_.begin() + static_cast<std::ptrdiff_t>(candidate_count),
              [](const ProbIndex& left, const ProbIndex& right) {
                  if (left.prob != right.prob) {
                      return left.prob > right.prob;
                  }
                  return left.index < right.index;
              });

    float cumulative_prob = 0.0f;
    std::size_t last_index = candidate_count - 1;
    for (std::size_t i = 0; i < candidate_count; ++i) {
        cumulative_prob += probindex_[i].prob;
        if (cumulative_prob > topp_) {
            last_index = i;
            break;
        }
    }

    const float target = coin * cumulative_prob;
    float cdf = 0.0f;
    for (std::size_t i = 0; i <= last_index; ++i) {
        cdf += probindex_[i].prob;
        if (target < cdf) {
            return probindex_[i].index;
        }
    }
    return probindex_[last_index].index;
}

int Sampler::sample(float* logits) {
    if (temperature_ == 0.0f) {
        return sample_argmax(logits);
    }

    for (int i = 0; i < vocab_size_; ++i) {
        logits[i] /= temperature_;
    }
    nn::softmax(logits, vocab_size_);

    const float coin = random_f32();
    if (topp_ <= 0.0f || topp_ >= 1.0f) {
        return sample_mult(logits, coin);
    }
    return sample_topp(logits, coin);
}

}  // namespace mini_llama
