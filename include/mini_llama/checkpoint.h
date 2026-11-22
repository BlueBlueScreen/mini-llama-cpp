#pragma once

#include "mini_llama/config.h"

#include <cstddef>
#include <string>

namespace mini_llama {

class Checkpoint {
public:
    explicit Checkpoint(const std::string& path);
    ~Checkpoint();

    Checkpoint(const Checkpoint&) = delete;
    Checkpoint& operator=(const Checkpoint&) = delete;
    Checkpoint(Checkpoint&& other) noexcept;
    Checkpoint& operator=(Checkpoint&& other) noexcept;

    const Config& config() const noexcept { return config_; }
    const float* weights() const noexcept { return data_; }
    std::size_t weight_floats() const noexcept { return weight_floats_; }
    std::size_t file_size() const noexcept { return file_size_; }

private:
    void close() noexcept;

    void* mapping_ = nullptr;
    const float* data_ = nullptr;
    std::size_t file_size_ = 0;
    std::size_t weight_floats_ = 0;
    Config config_;
    int fd_ = -1;
};

}  // namespace mini_llama
