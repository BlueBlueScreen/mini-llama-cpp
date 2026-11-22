cmake_minimum_required(VERSION 3.16)
project(mini_llama LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(MINI_LLAMA_ENABLE_OPENMP "Parallelize matrix rows and attention heads" OFF)
option(MINI_LLAMA_ENABLE_SANITIZERS "Build with address and undefined-behavior sanitizers" OFF)

add_library(mini_llama_core STATIC
    src/config.cpp
    src/checkpoint.cpp
    src/model.cpp
    src/nn.cpp
    src/tokenizer.cpp
    src/sampler.cpp
    src/generator.cpp
)

target_include_directories(mini_llama_core PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_compile_options(mini_llama_core PRIVATE -Wall -Wextra -Wpedantic)

if(MINI_LLAMA_ENABLE_OPENMP)
    find_package(OpenMP REQUIRED)
    target_link_libraries(mini_llama_core PUBLIC OpenMP::OpenMP_CXX)
endif()

if(MINI_LLAMA_ENABLE_SANITIZERS)
    target_compile_options(mini_llama_core PUBLIC -fsanitize=address,undefined -fno-omit-frame-pointer)
    target_link_options(mini_llama_core PUBLIC -fsanitize=address,undefined)
endif()

add_executable(mini_llama apps/main.cpp)
target_link_libraries(mini_llama PRIVATE mini_llama_core)

enable_testing()

add_executable(test_tokenizer tests/test_tokenizer.cpp)
target_link_libraries(test_tokenizer PRIVATE mini_llama_core)
target_compile_options(test_tokenizer PRIVATE -Wall -Wextra -Wpedantic)
add_test(NAME tokenizer COMMAND test_tokenizer tokenizer.bin)
set_tests_properties(tokenizer PROPERTIES
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)

add_executable(test_math tests/test_math.cpp)
target_link_libraries(test_math PRIVATE mini_llama_core)
target_compile_options(test_math PRIVATE -Wall -Wextra -Wpedantic)
add_test(NAME math COMMAND test_math)

set(MODEL_PATH "${CMAKE_CURRENT_SOURCE_DIR}/models/stories15M.bin" CACHE FILEPATH
    "Path to a legacy llama2.c checkpoint used by test_model")
set(MODEL_GQA_PATH "${CMAKE_CURRENT_SOURCE_DIR}/models/stories260K.bin" CACHE FILEPATH
    "Path to a GQA llama2.c checkpoint used by test_model")
if(EXISTS "${MODEL_PATH}" OR EXISTS "${MODEL_GQA_PATH}")
    add_executable(test_model tests/test_model.cpp)
    target_link_libraries(test_model PRIVATE mini_llama_core)
    target_compile_options(test_model PRIVATE -Wall -Wextra -Wpedantic)
endif()
if(EXISTS "${MODEL_PATH}")
    add_test(NAME model COMMAND test_model "${MODEL_PATH}")
endif()
if(EXISTS "${MODEL_GQA_PATH}")
    add_test(NAME model_gqa COMMAND test_model "${MODEL_GQA_PATH}")
endif()
