#include "mini_llama/checkpoint.h"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cerrno>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

namespace mini_llama {
namespace {

constexpr std::size_t kHeaderBytes = 7 * sizeof(std::int32_t);

[[noreturn]] void throw_errno(const std::string& what) {
    throw std::runtime_error(what + ": " + std::strerror(errno));
}

std::uint64_t expected_weight_floats(const Config& c) {
    const std::uint64_t dim = static_cast<std::uint64_t>(c.dim);
    const std::uint64_t hidden = static_cast<std::uint64_t>(c.hidden_dim);
    const std::uint64_t layers = static_cast<std::uint64_t>(c.n_layers);
    const std::uint64_t vocab = static_cast<std::uint64_t>(c.vocab_size);
    const std::uint64_t seq = static_cast<std::uint64_t>(c.seq_len);
    const std::uint64_t head = static_cast<std::uint64_t>(c.head_size());
    const std::uint64_t kv = static_cast<std::uint64_t>(c.kv_dim());

    std::uint64_t count = 0;
    count += vocab * dim;
    count += layers * dim;
    count += layers * dim * dim;
    count += layers * dim * kv;
    count += layers * dim * kv;
    count += layers * dim * dim;
    count += layers * dim;
    count += layers * dim * hidden;
    count += layers * hidden * dim;
    count += layers * dim * hidden;
    count += dim;
    count += seq * head / 2;
    count += seq * head / 2;
    if (!c.shared_weights) {
        count += vocab * dim;
    }
    return count;
}

}  // namespace

Checkpoint::Checkpoint(const std::string& path) {
    fd_ = ::open(path.c_str(), O_RDONLY);
    if (fd_ < 0) {
        throw_errno("cannot open checkpoint '" + path + "'");
    }

    struct stat info {};
    if (::fstat(fd_, &info) != 0) {
        const int saved = errno;
        ::close(fd_);
        fd_ = -1;
        throw std::runtime_error("cannot stat checkpoint '" + path
                                 + "': " + std::strerror(saved));
    }
    if (info.st_size < 0 || static_cast<std::uint64_t>(info.st_size) < kHeaderBytes) {
        ::close(fd_);
        fd_ = -1;
        throw std::runtime_error("checkpoint is smaller than its 7-int header");
    }
    file_size_ = static_cast<std::size_t>(info.st_size);

    mapping_ = ::mmap(nullptr, file_size_, PROT_READ, MAP_PRIVATE, fd_, 0);
    if (mapping_ == MAP_FAILED) {
        mapping_ = nullptr;
        const int saved = errno;
        ::close(fd_);
        fd_ = -1;
        throw std::runtime_error("mmap failed: " + std::string(std::strerror(saved)));
    }

    try {
        std::int32_t header[7];
        std::memcpy(header, mapping_, sizeof(header));

        constexpr std::int32_t kVersion1Magic = 0x616b3432;
        if (header[0] == kVersion1Magic) {
            throw std::runtime_error(
                "checkpoint uses llama2.c v1 header; this port supports the legacy v0 format");
        }

        const std::int64_t vocab = header[5];
        if (vocab == std::numeric_limits<std::int32_t>::min()
            || vocab > std::numeric_limits<int>::max()
            || vocab < -std::numeric_limits<int>::max()) {
            throw std::runtime_error("checkpoint has an invalid vocab_size");
        }

        config_.dim = header[0];
        config_.hidden_dim = header[1];
        config_.n_layers = header[2];
        config_.n_heads = header[3];
        config_.n_kv_heads = header[4];
        config_.vocab_size = static_cast<int>(vocab < 0 ? -vocab : vocab);
        config_.seq_len = header[6];
        config_.shared_weights = vocab > 0;
        config_.validate();

        weight_floats_ = static_cast<std::size_t>(expected_weight_floats(config_));
        const std::uint64_t expected_size = static_cast<std::uint64_t>(kHeaderBytes)
            + static_cast<std::uint64_t>(weight_floats_) * sizeof(float);
        if (expected_size != file_size_) {
            throw std::runtime_error(
                "checkpoint size mismatch: header says " + std::to_string(expected_size)
                + " bytes, file has " + std::to_string(file_size_));
        }
    } catch (...) {
        ::munmap(mapping_, file_size_);
        mapping_ = nullptr;
        ::close(fd_);
        fd_ = -1;
        throw;
    }

    data_ = reinterpret_cast<const float*>(
        static_cast<const char*>(mapping_) + kHeaderBytes);
}

Checkpoint::~Checkpoint() {
    close();
}

Checkpoint::Checkpoint(Checkpoint&& other) noexcept
    : mapping_(other.mapping_),
      data_(other.data_),
      file_size_(other.file_size_),
      weight_floats_(other.weight_floats_),
      config_(other.config_),
      fd_(other.fd_) {
    other.mapping_ = nullptr;
    other.data_ = nullptr;
    other.file_size_ = 0;
    other.weight_floats_ = 0;
    other.fd_ = -1;
}

Checkpoint& Checkpoint::operator=(Checkpoint&& other) noexcept {
    if (this != &other) {
        close();
        mapping_ = other.mapping_;
        data_ = other.data_;
        file_size_ = other.file_size_;
        weight_floats_ = other.weight_floats_;
        config_ = other.config_;
        fd_ = other.fd_;

        other.mapping_ = nullptr;
        other.data_ = nullptr;
        other.file_size_ = 0;
        other.weight_floats_ = 0;
        other.fd_ = -1;
    }
    return *this;
}

void Checkpoint::close() noexcept {
    if (mapping_ != nullptr) {
        ::munmap(mapping_, file_size_);
        mapping_ = nullptr;
    }
    data_ = nullptr;
    if (fd_ >= 0) {
        ::close(fd_);
        fd_ = -1;
    }
}

}  // namespace mini_llama
