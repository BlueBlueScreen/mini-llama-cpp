#pragma once

#include <cstddef>

namespace mini_llama {

struct Config {
    int dim = 0;
    int hidden_dim = 0;
    int n_layers = 0;
    int n_heads = 0;
    int n_kv_heads = 0;
    int vocab_size = 0;
    int seq_len = 0;
    bool shared_weights = true;

    int head_size() const noexcept { return dim / n_heads; }
    int kv_dim() const noexcept { return dim * n_kv_heads / n_heads; }
    int kv_mul() const noexcept { return n_heads / n_kv_heads; }
    std::size_t kv_cache_floats() const noexcept;

    void validate() const;
};

}  // namespace mini_llama
