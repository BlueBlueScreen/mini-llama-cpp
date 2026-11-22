#include "mini_llama/model.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: test_model <checkpoint>\n");
        return 1;
    }

    try {
        mini_llama::Transformer transformer(argv[1]);
        const mini_llama::Config& config = transformer.config();
        const mini_llama::RunState& state = transformer.state();
        const int kv_dim = config.kv_dim();

        float* logits = transformer.forward(1, 0);
        int finite_logits = 0;
        for (int i = 0; i < config.vocab_size; ++i) {
            if (std::isfinite(logits[i])) {
                ++finite_logits;
            }
        }
        if (finite_logits != config.vocab_size) {
            std::fprintf(stderr, "non-finite logits at position 0\n");
            return 1;
        }

        std::vector<float> first_key(
            state.key_cache.begin(),
            state.key_cache.begin() + kv_dim);
        std::vector<float> first_value(
            state.value_cache.begin(),
            state.value_cache.begin() + kv_dim);

        transformer.forward(1, 1);

        if (std::memcmp(first_key.data(), state.key_cache.data(),
                        first_key.size() * sizeof(float)) != 0) {
            std::fprintf(stderr, "position 0 key cache changed after position 1\n");
            return 1;
        }
        if (std::memcmp(first_value.data(), state.value_cache.data(),
                        first_value.size() * sizeof(float)) != 0) {
            std::fprintf(stderr, "position 0 value cache changed after position 1\n");
            return 1;
        }

        bool second_row_nonzero = false;
        for (int i = 0; i < kv_dim; ++i) {
            if (state.key_cache[static_cast<std::size_t>(kv_dim) + i] != 0.0f) {
                second_row_nonzero = true;
                break;
            }
        }
        if (!second_row_nonzero) {
            std::fprintf(stderr, "position 1 key cache row was not written\n");
            return 1;
        }
    } catch (const std::exception& error) {
        std::fprintf(stderr, "model test failed: %s\n", error.what());
        return 1;
    }

    std::puts("model tests passed");
    return 0;
}
