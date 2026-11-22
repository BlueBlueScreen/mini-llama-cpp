#include "mini_llama/generator.h"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cstdio>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace mini_llama {
namespace {

void print_piece(const std::string& piece) {
    if (piece.empty()) {
        return;
    }
    if (piece.size() == 1) {
        const auto byte = static_cast<unsigned char>(piece[0]);
        if (!(std::isprint(byte) || std::isspace(byte))) {
            return;
        }
    }
    std::fwrite(piece.data(), 1, piece.size(), stdout);
    std::fflush(stdout);
}

std::string read_line_with_prompt(const char* prompt) {
    std::fputs(prompt, stdout);
    std::fflush(stdout);
    std::string line;
    std::getline(std::cin, line);
    return line;
}

}  // namespace

void generate(Transformer& transformer, Tokenizer& tokenizer, Sampler& sampler,
              std::string_view prompt, int steps) {
    const Config& config = transformer.config();
    if (steps > config.seq_len) {
        steps = config.seq_len;
    }
    if (steps < 0) {
        steps = 0;
    }

    const std::vector<int> prompt_tokens = tokenizer.encode(prompt, true, false);
    if (prompt_tokens.empty()) {
        throw std::runtime_error("expected at least one prompt token");
    }

    int token = prompt_tokens[0];
    int next = 0;
    int pos = 0;
    std::chrono::steady_clock::time_point start{};
    bool timing_started = false;

    while (pos < steps) {
        float* logits = transformer.forward(token, pos);

        if (pos < static_cast<int>(prompt_tokens.size()) - 1) {
            next = prompt_tokens[static_cast<std::size_t>(pos) + 1];
        } else {
            next = sampler.sample(logits);
        }
        ++pos;

        if (next == 1) {
            break;
        }

        print_piece(tokenizer.decode(token, next));
        token = next;

        if (!timing_started) {
            start = std::chrono::steady_clock::now();
            timing_started = true;
        }
    }

    std::fputc('\n', stdout);
    std::fflush(stdout);

    if (pos > 1 && timing_started) {
        const auto end = std::chrono::steady_clock::now();
        const double seconds = std::chrono::duration<double>(end - start).count();
        if (seconds > 0.0) {
            std::fprintf(stderr, "achieved tok/s: %.2f\n",
                         static_cast<double>(pos - 1) / seconds);
        }
    }
}

void chat(Transformer& transformer, Tokenizer& tokenizer, Sampler& sampler,
          const std::optional<std::string>& cli_user_prompt,
          const std::optional<std::string>& cli_system_prompt, int steps) {
    const Config& config = transformer.config();
    if (steps > config.seq_len) {
        steps = config.seq_len;
    }
    if (steps < 0) {
        steps = 0;
    }

    std::string system_prompt;
    std::string user_prompt;
    std::vector<int> prompt_tokens;

    bool user_turn = true;
    int next = 0;
    int token = 0;
    int pos = 0;
    std::size_t user_index = 0;

    while (pos < steps) {
        if (user_turn) {
            if (pos == 0) {
                system_prompt = cli_system_prompt
                    ? *cli_system_prompt
                    : read_line_with_prompt("Enter system prompt (optional): ");
            }

            if (pos == 0 && cli_user_prompt) {
                user_prompt = *cli_user_prompt;
            } else {
                user_prompt = read_line_with_prompt("User: ");
            }

            std::string rendered_prompt;
            if (pos == 0 && !system_prompt.empty()) {
                rendered_prompt = "[INST] <<SYS>>\n" + system_prompt
                    + "\n<</SYS>>\n\n" + user_prompt + " [/INST]";
            } else {
                rendered_prompt = "[INST] " + user_prompt + " [/INST]";
            }

            prompt_tokens = tokenizer.encode(rendered_prompt, true, false);
            user_index = 0;
            user_turn = false;
            std::fputs("Assistant: ", stdout);
            std::fflush(stdout);
        }

        if (user_index < prompt_tokens.size()) {
            token = prompt_tokens[user_index++];
        } else {
            token = next;
        }

        if (token == 2) {
            user_turn = true;
        }

        float* logits = transformer.forward(token, pos);
        next = sampler.sample(logits);
        ++pos;

        if (user_index >= prompt_tokens.size() && next != 2) {
            print_piece(tokenizer.decode(token, next));
        }
        if (next == 2) {
            std::fputc('\n', stdout);
            std::fflush(stdout);
        }
    }

    std::fputc('\n', stdout);
    std::fflush(stdout);
}

}  // namespace mini_llama
