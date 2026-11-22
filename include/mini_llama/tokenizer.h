#pragma once

#include <string>
#include <string_view>
#include <vector>

namespace mini_llama {

class Tokenizer {
public:
    Tokenizer(const std::string& path, int vocab_size);

    std::vector<int> encode(std::string_view text, bool bos, bool eos) const;
    std::string decode(int prev_token, int token) const;

    int vocab_size() const noexcept { return vocab_size_; }
    const std::string& piece(int id) const { return vocab_.at(static_cast<std::size_t>(id)); }

private:
    int lookup(std::string_view text) const;

    int vocab_size_ = 0;
    int max_token_length_ = 0;
    std::vector<float> scores_;
    std::vector<std::string> vocab_;
    std::vector<int> sorted_ids_;
};

}  // namespace mini_llama
