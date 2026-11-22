#include "mini_llama/tokenizer.h"

#include <cstdio>
#include <string>
#include <vector>

namespace {

struct Case {
    std::string prompt;
    std::vector<int> expected;
};

bool check_case(const mini_llama::Tokenizer& tokenizer, const Case& test_case) {
    const std::vector<int> actual = tokenizer.encode(test_case.prompt, true, false);
    if (actual == test_case.expected) {
        return true;
    }

    std::fprintf(stderr, "token mismatch for prompt: \"%s\"\n  expected:",
                 test_case.prompt.c_str());
    for (int token : test_case.expected) {
        std::fprintf(stderr, " %d", token);
    }
    std::fprintf(stderr, "\n  actual:  ");
    for (int token : actual) {
        std::fprintf(stderr, " %d", token);
    }
    std::fprintf(stderr, "\n");
    return false;
}

}  // namespace

int main(int argc, char** argv) {
    const std::string path = argc > 1 ? argv[1] : "tokenizer.bin";

    try {
        mini_llama::Tokenizer tokenizer(path, 32000);

        const std::vector<Case> cases = {
            {"", {1}},
            {"I believe the meaning of life is",
             {1, 306, 4658, 278, 6593, 310, 2834, 338}},
            {"Simply put, the theory of relativity states that ",
             {1, 3439, 17632, 1925, 29892, 278, 6368, 310, 14215, 537, 5922, 393, 29871}},
            {"A brief message congratulating the team on the launch:\n\n        Hi everyone,\n\n        I just ",
             {1, 319, 11473, 2643, 378, 629, 271, 18099, 278, 3815, 373,
              278, 6826, 29901, 13, 13, 4706, 6324, 14332, 29892, 13, 13,
              4706, 306, 925, 29871}},
            {"Translate English to French:\n\n        sea otter => loutre de mer\n        peppermint => menthe poivrée\n        plush girafe => girafe peluche\n        cheese =>",
             {1, 4103, 9632, 4223, 304, 5176, 29901, 13, 13, 4706, 7205, 4932,
              357, 1149, 301, 449, 276, 316, 2778, 13, 4706, 1236, 407, 837,
              524, 1149, 6042, 354, 772, 440, 29878, 1318, 13, 4706, 715, 1878,
              330, 3055, 1725, 1149, 330, 3055, 1725, 4639, 28754, 13, 4706,
              923, 968, 1149}},
        };

        for (const Case& test_case : cases) {
            if (!check_case(tokenizer, test_case)) {
                return 1;
            }
        }
    } catch (const std::exception& error) {
        std::fprintf(stderr, "tokenizer test failed: %s\n", error.what());
        return 1;
    }

    std::puts("tokenizer tests passed");
    return 0;
}
