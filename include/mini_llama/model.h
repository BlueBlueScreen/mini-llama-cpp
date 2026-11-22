#pragma once

#include "mini_llama/checkpoint.h"
#include "mini_llama/config.h"

#include <string>
#include <vector>

namespace mini_llama {

struct TransformerWeights {
    const float* token_embedding_table = nullptr;  // (vocab_size, dim)
    const float* rms_att_weight = nullptr;         // (n_layers, dim)
    const float* wq = nullptr;                     // (n_layers, dim, dim)
    const float* wk = nullptr;                     // (n_layers, dim, kv_dim)
    const float* wv = nullptr;                     // (n_layers, dim, kv_dim)
    const float* wo = nullptr;                     // (n_layers, dim, dim)
    const float* rms_ffn_weight = nullptr;         // (n_layers, dim)
    const float* w1 = nullptr;                     // (n_layers, hidden_dim, dim)
    const float* w2 = nullptr;                     // (n_layers, dim, hidden_dim)
    const float* w3 = nullptr;                     // (n_layers, hidden_dim, dim)
    const float* rms_final_weight = nullptr;       // (dim)
    const float* wcls = nullptr;                   // (vocab_size, dim)
};

struct RunState {
    std::vector<float> x;
    std::vector<float> xb;
    std::vector<float> xb2;
    std::vector<float> hb;
    std::vector<float> hb2;
    std::vector<float> q;
    std::vector<float> att;
    std::vector<float> logits;
    std::vector<float> key_cache;
    std::vector<float> value_cache;

    static RunState create(const Config& config);
};

class Transformer {
public:
    explicit Transformer(const std::string& checkpoint_path);

    const Config& config() const noexcept { return checkpoint_.config(); }
    const TransformerWeights& weights() const noexcept { return weights_; }
    const RunState& state() const noexcept { return state_; }

    // Returns a pointer to the logits buffer, valid until the next forward call.
    float* forward(int token, int pos);

private:
    void map_weights();

    Checkpoint checkpoint_;
    TransformerWeights weights_;
    RunState state_;
};

}  // namespace mini_llama
