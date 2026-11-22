#include "mini_llama/model.h"

#include "mini_llama/nn.h"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>

namespace mini_llama {

RunState RunState::create(const Config& config) {
    RunState state;
    const auto dim = static_cast<std::size_t>(config.dim);
    const auto hidden = static_cast<std::size_t>(config.hidden_dim);
    const auto vocab = static_cast<std::size_t>(config.vocab_size);

    state.x.assign(dim, 0.0f);
    state.xb.assign(dim, 0.0f);
    state.xb2.assign(dim, 0.0f);
    state.hb.assign(hidden, 0.0f);
    state.hb2.assign(hidden, 0.0f);
    state.q.assign(dim, 0.0f);
    state.att.assign(static_cast<std::size_t>(config.n_heads)
                         * static_cast<std::size_t>(config.seq_len),
                     0.0f);
    state.logits.assign(vocab, 0.0f);
    state.key_cache.assign(config.kv_cache_floats(), 0.0f);
    state.value_cache.assign(config.kv_cache_floats(), 0.0f);
    return state;
}

Transformer::Transformer(const std::string& checkpoint_path)
    : checkpoint_(checkpoint_path),
      weights_(),
      state_(RunState::create(checkpoint_.config())) {
    map_weights();
}

void Transformer::map_weights() {
    const Config& c = config();
    const float* cursor = checkpoint_.weights();

    auto take = [&cursor](std::uint64_t count) {
        const float* result = cursor;
        cursor += static_cast<std::size_t>(count);
        return result;
    };

    const std::uint64_t dim = static_cast<std::uint64_t>(c.dim);
    const std::uint64_t hidden = static_cast<std::uint64_t>(c.hidden_dim);
    const std::uint64_t layers = static_cast<std::uint64_t>(c.n_layers);
    const std::uint64_t vocab = static_cast<std::uint64_t>(c.vocab_size);
    const std::uint64_t seq = static_cast<std::uint64_t>(c.seq_len);
    const std::uint64_t head = static_cast<std::uint64_t>(c.head_size());
    const std::uint64_t kv = static_cast<std::uint64_t>(c.kv_dim());

    weights_.token_embedding_table = take(vocab * dim);
    weights_.rms_att_weight = take(layers * dim);
    weights_.wq = take(layers * dim * dim);
    weights_.wk = take(layers * dim * kv);
    weights_.wv = take(layers * dim * kv);
    weights_.wo = take(layers * dim * dim);
    weights_.rms_ffn_weight = take(layers * dim);
    weights_.w1 = take(layers * hidden * dim);
    weights_.w2 = take(layers * dim * hidden);
    weights_.w3 = take(layers * hidden * dim);
    weights_.rms_final_weight = take(dim);
    take(seq * head / 2);  // legacy freq_cis_real is not used by run.c
    take(seq * head / 2);  // legacy freq_cis_imag is not used by run.c
    weights_.wcls = c.shared_weights ? weights_.token_embedding_table : take(vocab * dim);

    const float* expected_end = checkpoint_.weights() + checkpoint_.weight_floats();
    if (cursor != expected_end) {
        throw std::runtime_error("internal error: weight mapping did not consume the checkpoint");
    }
}

float* Transformer::forward(int token, int pos) {
    const Config& c = config();
    const TransformerWeights& w = weights_;
    RunState& s = state_;

    if (token < 0 || token >= c.vocab_size) {
        throw std::out_of_range("token id is outside the vocabulary");
    }
    if (pos < 0 || pos >= c.seq_len) {
        throw std::out_of_range("position is outside the KV cache");
    }

    const int dim = c.dim;
    const int kv_dim = c.kv_dim();
    const int kv_mul = c.kv_mul();
    const int hidden_dim = c.hidden_dim;
    const int head_size = c.head_size();
    const int seq_len = c.seq_len;

    std::memcpy(s.x.data(),
                w.token_embedding_table + static_cast<std::size_t>(token) * dim,
                static_cast<std::size_t>(dim) * sizeof(float));

    for (int layer = 0; layer < c.n_layers; ++layer) {
        const std::size_t l = static_cast<std::size_t>(layer);
        const std::size_t layer_dim = l * static_cast<std::size_t>(dim);
        const std::size_t layer_dim_dim = l * static_cast<std::size_t>(dim) * dim;
        const std::size_t layer_dim_hidden =
            l * static_cast<std::size_t>(dim) * hidden_dim;
        const std::size_t kv_layer_offset =
            l * static_cast<std::size_t>(seq_len) * kv_dim;

        nn::rmsnorm(s.xb.data(), s.x.data(),
                    w.rms_att_weight + layer_dim, dim);

        float* k = s.key_cache.data() + kv_layer_offset
                 + static_cast<std::size_t>(pos) * kv_dim;
        float* v = s.value_cache.data() + kv_layer_offset
                 + static_cast<std::size_t>(pos) * kv_dim;

        nn::matmul(s.q.data(), s.xb.data(),
                   w.wq + layer_dim_dim, dim, dim);
        nn::matmul(k, s.xb.data(),
                   w.wk + l * static_cast<std::size_t>(dim) * kv_dim, dim, kv_dim);
        nn::matmul(v, s.xb.data(),
                   w.wv + l * static_cast<std::size_t>(dim) * kv_dim, dim, kv_dim);

        nn::rope(s.q.data(), k, pos, dim, kv_dim, head_size);

#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
        for (int head = 0; head < c.n_heads; ++head) {
            float* q_head = s.q.data() + head * head_size;
            float* att = s.att.data()
                       + static_cast<std::size_t>(head) * seq_len;
            const int kv_head = head / kv_mul;

            const float head_scale = std::sqrt(static_cast<float>(head_size));
            for (int t = 0; t <= pos; ++t) {
                const float* k_head = s.key_cache.data() + kv_layer_offset
                                    + static_cast<std::size_t>(t) * kv_dim
                                    + kv_head * head_size;
                float score = 0.0f;
                for (int i = 0; i < head_size; ++i) {
                    score += q_head[i] * k_head[i];
                }
                att[t] = score / head_scale;
            }

            nn::softmax(att, pos + 1);

            float* out_head = s.xb.data() + head * head_size;
            std::memset(out_head, 0, static_cast<std::size_t>(head_size) * sizeof(float));
            for (int t = 0; t <= pos; ++t) {
                const float* v_head = s.value_cache.data() + kv_layer_offset
                                    + static_cast<std::size_t>(t) * kv_dim
                                    + kv_head * head_size;
                const float weight = att[t];
                for (int i = 0; i < head_size; ++i) {
                    out_head[i] += weight * v_head[i];
                }
            }
        }

        nn::matmul(s.xb2.data(), s.xb.data(),
                   w.wo + layer_dim_dim, dim, dim);
        for (int i = 0; i < dim; ++i) {
            s.x[i] += s.xb2[i];
        }

        nn::rmsnorm(s.xb.data(), s.x.data(),
                    w.rms_ffn_weight + layer_dim, dim);

        nn::matmul(s.hb.data(), s.xb.data(),
                   w.w1 + layer_dim_hidden, dim, hidden_dim);
        nn::matmul(s.hb2.data(), s.xb.data(),
                   w.w3 + layer_dim_hidden, dim, hidden_dim);

        for (int i = 0; i < hidden_dim; ++i) {
            float value = s.hb[i];
            value *= 1.0f / (1.0f + std::exp(-value));
            value *= s.hb2[i];
            s.hb[i] = value;
        }

        nn::matmul(s.xb.data(), s.hb.data(),
                   w.w2 + layer_dim_hidden, hidden_dim, dim);
        for (int i = 0; i < dim; ++i) {
            s.x[i] += s.xb[i];
        }
    }

    nn::rmsnorm(s.x.data(), s.x.data(), w.rms_final_weight, dim);
    nn::matmul(s.logits.data(), s.x.data(), w.wcls, dim, c.vocab_size);
    return s.logits.data();
}

}  // namespace mini_llama
