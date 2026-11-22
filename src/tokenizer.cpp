#include "mini_llama/tokenizer.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>

namespace mini_llama {
namespace {

void require_read(const std::ifstream& file, const std::string& path) {
    if (!file) {
        throw std::runtime_error("truncated tokenizer file: " + path);
    }
}

}  // namespace

Tokenizer::Tokenizer(const std::string& path, int vocab_size)
    : vocab_size_(vocab_size) {
    if (vocab_size_ <= 0) {
        throw std::runtime_error("tokenizer vocab_size must be positive");
    }

    std::ifstream file(path, std::ios::binary);
    if (!file) {
        throw std::runtime_error("cannot open tokenizer file '" + path + "'");
    }

    file.read(reinterpret_cast<char*>(&max_token_length_), sizeof(std::int32_t));
    require_read(file, path);
    if (max_token_length_ <= 0) {
        throw std::runtime_error("invalid max_token_length in tokenizer file");
    }

    scores_.resize(static_cast<std::size_t>(vocab_size_));
    vocab_.resize(static_cast<std::size_t>(vocab_size_));

    for (int i = 0; i < vocab_size_; ++i) {
        float score = 0.0f;
        std::int32_t length = 0;

        file.read(reinterpret_cast<char*>(&score), sizeof(score));
        require_read(file, path);
        file.read(reinterpret_cast<char*>(&length), sizeof(length));
        require_read(file, path);
        if (length < 0) {
            throw std::runtime_error("negative token length in tokenizer file");
        }

        std::string token(static_cast<std::size_t>(length), '\0');
        if (length > 0) {
            file.read(token.data(), length);
            require_read(file, path);
        }

        scores_[static_cast<std::size_t>(i)] = score;
        vocab_[static_cast<std::size_t>(i)] = std::move(token);
    }

    sorted_ids_.resize(static_cast<std::size_t>(vocab_size_));
    std::iota(sorted_ids_.begin(), sorted_ids_.end(), 0);
    std::sort(sorted_ids_.begin(), sorted_ids_.end(), [this](int a, int b) {
        const std::string& left = vocab_[static_cast<std::size_t>(a)];
        const std::string& right = vocab_[static_cast<std::size_t>(b)];
        if (left != right) {
            return left < right;
        }
        return a < b;
    });
}

int Tokenizer::lookup(std::string_view text) const {
    auto it = std::lower_bound(
        sorted_ids_.begin(), sorted_ids_.end(), text,
        [this](int id, std::string_view key) {
            return std::string_view(vocab_[static_cast<std::size_t>(id)]) < key;
        });
    if (it != sorted_ids_.end()
        && std::string_view(vocab_[static_cast<std::size_t>(*it)]) == text) {
        return *it;
    }
    return -1;
}

std::vector<int> Tokenizer::encode(std::string_view text, bool bos, bool eos) const {
    std::vector<int> tokens;
    tokens.reserve(text.size() + 3);

    if (bos) {
        tokens.push_back(1);
    }

    if (!text.empty()) {
        const int dummy_prefix = lookup(" ");
        if (dummy_prefix < 0) {
            throw std::runtime_error("tokenizer vocabulary has no dummy-prefix token");
        }
        tokens.push_back(dummy_prefix);
    }

    std::string code_point;
    for (std::size_t i = 0; i < text.size(); ++i) {
        const auto byte = static_cast<unsigned char>(text[i]);
        if ((byte & 0xC0) != 0x80) {
            code_point.clear();
        }
        code_point.push_back(static_cast<char>(byte));

        const bool next_is_continuation =
            (i + 1 < text.size()
             && (static_cast<unsigned char>(text[i + 1]) & 0xC0) == 0x80);
        if (next_is_continuation && code_point.size() < 4) {
            continue;
        }

        const int id = lookup(code_point);
        if (id != -1) {
            tokens.push_back(id);
        } else {
            for (unsigned char c : code_point) {
                tokens.push_back(static_cast<int>(c) + 3);
            }
        }
        code_point.clear();
    }

    while (true) {
        float best_score = -1e10f;
        int best_id = -1;
        std::size_t best_index = 0;

        for (std::size_t i = 0; i + 1 < tokens.size(); ++i) {
            const std::string merged =
                vocab_[static_cast<std::size_t>(tokens[i])]
                + vocab_[static_cast<std::size_t>(tokens[i + 1])];
            const int id = lookup(merged);
            if (id != -1 && scores_[static_cast<std::size_t>(id)] > best_score) {
                best_score = scores_[static_cast<std::size_t>(id)];
                best_id = id;
                best_index = i;
            }
        }

        if (best_id == -1) {
            break;
        }

        tokens[best_index] = best_id;
        tokens.erase(tokens.begin() + static_cast<std::ptrdiff_t>(best_index) + 1);
    }

    if (eos) {
        tokens.push_back(2);
    }
    return tokens;
}

std::string Tokenizer::decode(int prev_token, int token) const {
    std::string piece = vocab_.at(static_cast<std::size_t>(token));
    if (prev_token == 1 && !piece.empty() && piece[0] == ' ') {
        piece.erase(0, 1);
    }

    unsigned int byte_value = 0;
    if (std::sscanf(piece.c_str(), "<0x%02X>", &byte_value) == 1) {
        return std::string(1, static_cast<char>(static_cast<unsigned char>(byte_value)));
    }
    return piece;
}

}  // namespace mini_llama
