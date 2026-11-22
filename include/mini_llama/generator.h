#pragma once

#include "mini_llama/model.h"
#include "mini_llama/sampler.h"
#include "mini_llama/tokenizer.h"

#include <optional>
#include <string>
#include <string_view>

namespace mini_llama {

void generate(Transformer& transformer, Tokenizer& tokenizer, Sampler& sampler,
              std::string_view prompt, int steps);

void chat(Transformer& transformer, Tokenizer& tokenizer, Sampler& sampler,
          const std::optional<std::string>& cli_user_prompt,
          const std::optional<std::string>& cli_system_prompt, int steps);

}  // namespace mini_llama
