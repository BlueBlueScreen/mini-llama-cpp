#include "mini_llama/generator.h"
#include "mini_llama/model.h"
#include "mini_llama/sampler.h"
#include "mini_llama/tokenizer.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <optional>
#include <stdexcept>
#include <string>

namespace {

struct Options {
    std::string checkpoint_path;
    std::string tokenizer_path = "tokenizer.bin";
    float temperature = 1.0f;
    float topp = 0.9f;
    int steps = 256;
    std::optional<std::string> prompt;
    std::uint64_t seed = 0;
    std::string mode = "generate";
    std::optional<std::string> system_prompt;
};

[[noreturn]] void usage() {
    std::fprintf(stderr,
        "Usage:   mini_llama <checkpoint> [options]\n"
        "Example: mini_llama stories15M.bin -n 256 -i \"Once upon a time\"\n"
        "Options:\n"
        "  -t <float>  temperature in [0,inf], default 1.0\n"
        "  -p <float>  p value in top-p (nucleus) sampling in [0,1], default 0.9\n"
        "  -s <int>    random seed, default time(NULL)\n"
        "  -n <int>    number of steps, default 256; 0 = max_seq_len\n"
        "  -i <string> input prompt\n"
        "  -z <string> optional path to custom tokenizer\n"
        "  -m <string> mode: generate|chat, default: generate\n"
        "  -y <string> optional system prompt in chat mode\n");
    std::exit(EXIT_FAILURE);
}

template <typename T, typename Parser>
T parse_number(const std::string& flag, const std::string& text, Parser parser) {
    try {
        std::size_t consumed = 0;
        const T value = parser(text, &consumed);
        if (consumed != text.size()) {
            throw std::invalid_argument("trailing characters");
        }
        return value;
    } catch (const std::exception&) {
        throw std::runtime_error("invalid value for " + flag + ": " + text);
    }
}

Options parse_options(int argc, char** argv) {
    if (argc < 2) {
        usage();
    }

    Options options;
    options.checkpoint_path = argv[1];

    for (int i = 2; i < argc; i += 2) {
        if (i + 1 >= argc) {
            usage();
        }
        const std::string flag = argv[i];
        if (flag.size() != 2 || flag[0] != '-') {
            usage();
        }
        const std::string value = argv[i + 1];

        switch (flag[1]) {
            case 't':
                options.temperature =
                    parse_number<float>(flag, value, [](const std::string& s, std::size_t* p) {
                        return std::stof(s, p);
                    });
                break;
            case 'p':
                options.topp =
                    parse_number<float>(flag, value, [](const std::string& s, std::size_t* p) {
                        return std::stof(s, p);
                    });
                break;
            case 's': {
                const long long seed =
                    parse_number<long long>(flag, value, [](const std::string& s, std::size_t* p) {
                        return std::stoll(s, p);
                    });
                options.seed = seed == 0 ? 0 : static_cast<std::uint64_t>(seed);
                break;
            }
            case 'n':
                options.steps =
                    parse_number<int>(flag, value, [](const std::string& s, std::size_t* p) {
                        return std::stoi(s, p);
                    });
                break;
            case 'i':
                options.prompt = value;
                break;
            case 'z':
                options.tokenizer_path = value;
                break;
            case 'm':
                options.mode = value;
                break;
            case 'y':
                options.system_prompt = value;
                break;
            default:
                usage();
        }
    }

    if (options.seed == 0) {
        options.seed = static_cast<std::uint64_t>(std::time(nullptr));
    }
    if (options.temperature < 0.0f) {
        options.temperature = 0.0f;
    }
    if (options.topp < 0.0f || options.topp > 1.0f) {
        options.topp = 0.9f;
    }
    if (options.steps < 0) {
        options.steps = 0;
    }

    return options;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        Options options = parse_options(argc, argv);

        mini_llama::Transformer transformer(options.checkpoint_path);
        int steps = options.steps;
        if (steps == 0 || steps > transformer.config().seq_len) {
            steps = transformer.config().seq_len;
        }

        mini_llama::Tokenizer tokenizer(options.tokenizer_path,
                                        transformer.config().vocab_size);
        mini_llama::Sampler sampler(transformer.config().vocab_size,
                                    options.temperature, options.topp, options.seed);

        if (options.mode == "generate") {
            mini_llama::generate(transformer, tokenizer, sampler,
                                 options.prompt.value_or(""), steps);
        } else if (options.mode == "chat") {
            mini_llama::chat(transformer, tokenizer, sampler,
                             options.prompt, options.system_prompt, steps);
        } else {
            std::fprintf(stderr, "unknown mode: %s\n", options.mode.c_str());
            usage();
        }
    } catch (const std::exception& error) {
        std::fprintf(stderr, "error: %s\n", error.what());
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
