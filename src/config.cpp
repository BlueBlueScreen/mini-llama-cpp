#include "mini_llama/config.h"

#include <stdexcept>
#include <string>

namespace mini_llama {

std::size_t Config::kv_cache_floats() const noexcept {
    return static_cast<std::size_t>(n_layers) * static_cast<std::size_t>(seq_len)
         * static_cast<std::size_t>(kv_dim());
}

void Config::validate() const {
    auto require = [](bool condition, const char* message) {
        if (!condition) {
            throw std::runtime_error(std::string("invalid model config: ") + message);
        }
    };

    require(dim > 0, "dim must be positive");
    require(hidden_dim > 0, "hidden_dim must be positive");
    require(n_layers > 0, "n_layers must be positive");
    require(n_heads > 0, "n_heads must be positive");
    require(n_kv_heads > 0, "n_kv_heads must be positive");
    require(vocab_size > 0, "vocab_size must be positive");
    require(seq_len > 0, "seq_len must be positive");
    require(dim % n_heads == 0, "dim must be divisible by n_heads");
    require(n_heads % n_kv_heads == 0, "n_heads must be divisible by n_kv_heads");
    require((dim / n_heads) % 2 == 0, "head_size must be even for RoPE");
    require(kv_dim() == n_kv_heads * (dim / n_heads), "kv_dim must equal n_kv_heads * head_size");
}

}  // namespace mini_llama
