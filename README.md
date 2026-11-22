# mini-llama-cpp

[![CI](../../actions/workflows/ci.yml/badge.svg)](../../actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
![C++17](https://img.shields.io/badge/C%2B%2B-17-blue.svg)

**LLM inference in ~1,600 lines of dependency-free C++17** — a from-scratch modular
refactor of [karpathy/llama2.c](https://github.com/karpathy/llama2.c)'s `run.c`,
with RAII resource management, exception-safe error handling, and a test suite.
Standard library only; OpenMP is optional.

**Correctness is verified, not assumed:** greedy decoding *and* seeded top-p
sampling produce **byte-identical output to upstream `run.c`** on both a standard
and a GQA checkpoint. One command reproduces the check:
[`scripts/verify_upstream.sh`](scripts/verify_upstream.sh).

中文文档：[README.zh-CN.md](README.zh-CN.md)

## Demo

Real output, `stories15M.bin`, temperature 0.8, seed 42 (truncated):

```text
$ ./build/mini_llama models/stories15M.bin -i "Once upon a time" -n 256 -s 42
Once upon a time, there was a little boy named Timmy. Timmy loved to play
outside in the garden. One day, he saw a spider crawling on the ground. Timmy
thought the spider was funny, so he tried to catch it. But the spider was too
fast and ran away.
Timmy felt sad because he couldn't catch the spider. Suddenly, he saw a bird
flying in the sky. ...
achieved tok/s: 148.12
```

## What changed from C to C++17

The numerics are unchanged; everything around them is restructured:

| Concern | upstream `run.c` | this refactor |
|---|---|---|
| Structure | one ~1,000-line file | 7 modules behind `include/mini_llama/`, `namespace mini_llama` |
| Checkpoint loading | `fread` into a `malloc` buffer | zero-copy `mmap`; weights are `const float*` views |
| Resource cleanup | manual `free`/`fclose` | RAII `Checkpoint`: move-only, exception-safe constructor |
| Error handling | `fprintf` + `exit` | exceptions with context; config, file size, and token/position bounds all validated |
| Vocabulary lookup | linear scan per lookup | sorted ids + `std::lower_bound` — O(log V) |
| Run state | raw `malloc` arrays | `std::vector<float>`, sized once from `Config` |
| Parallelism | OpenMP `#ifdef` baked in | opt-in CMake flag; output bit-identical to serial |
| Tests | none | CTest: golden BPE cases, math invariants, KV-cache invariants |

## Architecture

```text
apps/main.cpp            CLI: argument parsing, wiring
        │
        ▼
generator                generate() / chat() loops, streaming output, tok/s
        │
        ├──► model       Transformer: weight mapping + forward(token, pos)
        │      ├──► nn         rmsnorm / softmax / matmul / rope
        │      └──► checkpoint RAII mmap of the .bin file
        │             └──► config   header parsing + validation
        ├──► tokenizer   BPE encode/decode, binary-search vocabulary
        └──► sampler     argmax / multinomial / top-p, xorshift64* RNG
```

Per-token data flow (identical to upstream):

```text
token ─► embedding ─► [ RMSNorm ─► Q/K/V ─► RoPE ─► attention over KV cache ─► +residual
                      ─► RMSNorm ─► SwiGLU FFN ─► +residual ] × n_layers
      ─► final RMSNorm ─► classifier ─► logits ─► sampler ─► next token
```

Grouped-query attention (GQA) is supported: `n_kv_heads` may be smaller than
`n_heads`, and each KV head is shared by `n_heads / n_kv_heads` query heads.
The bundled 260K model (8 query heads / 4 KV heads) exercises this path in CI.

## Quickstart

Requires CMake ≥ 3.16, a C++17 compiler, and a POSIX system (Linux/macOS).

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j

# Runs out of the box: a 260K-parameter GQA model and its tokenizer are bundled.
./build/mini_llama models/stories260K.bin -z models/tok512.bin -i "One day" -n 256

# For coherent stories, download the 15M model (~58 MB):
./scripts/download_model.sh
./build/mini_llama models/stories15M.bin -i "Once upon a time" -n 256

# Chat mode (Llama-2 chat template):
./build/mini_llama models/stories15M.bin -m chat -i "Tell me a story"

# Tests:
ctest --test-dir build --output-on-failure
```

`HF_ENDPOINT=https://hf-mirror.com ./scripts/download_model.sh` uses a mirror
where Hugging Face is slow or blocked. `42M` and `110M` checkpoints also work.

### CLI (same flags as upstream `run.c`)

| Flag | Meaning | Default |
|---|---|---|
| `-t` | temperature (0 = greedy) | 1.0 |
| `-p` | top-p (nucleus) cutoff | 0.9 |
| `-s` | RNG seed (0 = time-based) | 0 |
| `-n` | max steps (0 = `seq_len`) | 256 |
| `-i` | prompt | — |
| `-z` | tokenizer path | `tokenizer.bin` |
| `-m` | `generate` \| `chat` | `generate` |
| `-y` | system prompt (chat mode) | — |

## Correctness

Three layers of verification, all runnable locally:

1. **Differential testing against upstream.**
   [`scripts/verify_upstream.sh`](scripts/verify_upstream.sh) clones llama2.c,
   builds `run.c`, and diffs generated text for both greedy decoding and seeded
   top-p sampling (same xorshift64\* RNG, so sampling matches too):

   ```text
   PASS: greedy output is byte-identical to upstream run.c
   PASS: sampled output is byte-identical to upstream run.c
   ```

2. **Unit tests** (`ctest`):
   - `test_tokenizer` — golden BPE encodings, including multi-byte UTF-8;
   - `test_math` — RMSNorm/softmax/matmul against hand-computed values, RoPE
     norm preservation, sampler determinism under a fixed seed;
   - `test_model` — logits are finite; earlier KV-cache rows stay untouched as
     generation advances (the core KV-cache invariant). Runs on the bundled GQA
     model, and on `stories15M.bin` when present.

3. **Sanitizers.** The whole suite passes under ASan + UBSan:

   ```bash
   cmake -S . -B build-asan -DCMAKE_BUILD_TYPE=Debug -DMINI_LLAMA_ENABLE_SANITIZERS=ON
   cmake --build build-asan -j
   ctest --test-dir build-asan --output-on-failure
   ```

CI runs gcc/clang × {Release, OpenMP, ASan+UBSan} on every push.

## Performance

Optional OpenMP parallelizes matmul rows and attention heads — the two
embarrassingly parallel loops — without changing reduction order, so output is
bit-identical to serial:

```bash
cmake -S . -B build-omp -DCMAKE_BUILD_TYPE=Release -DMINI_LLAMA_ENABLE_OPENMP=ON
cmake --build build-omp -j
OMP_NUM_THREADS=8 ./build-omp/mini_llama models/stories15M.bin -n 256
```

Measured on an Intel Core Ultra 5 225H (Ubuntu 22.04 / WSL2, gcc 11.4, `-O3`),
`stories15M.bin`, 256 steps:

| Build | Threads | tok/s |
|---|---|---|
| Release, serial | 1 | ~148 |
| Release + OpenMP | 4 | ~444 |
| Release + OpenMP | 8 | ~673 |

On hybrid CPUs, using *all* logical cores can be slower than 8 threads (E-core
scheduling); tune `OMP_NUM_THREADS` to your performance cores. Numbers are
machine-specific — measure your own with the built-in `achieved tok/s` report.

## Limitations

- CPU-only, fp32, no SIMD/BLAS — this is a readability-first reference, not a
  speed record (see [llama.cpp](https://github.com/ggml-org/llama.cpp) for that).
- Legacy llama2.c **v0** checkpoint format; v1 files are rejected with a clear error.
- POSIX `mmap` backend (Linux/macOS); Windows would need a `CreateFileMapping` port.
- BPE encoding uses the same O(n²) merge loop as upstream — fine for prompts.

Possible next steps: AVX2 matmul, a Windows mapping backend, v1/safetensors
loading, int8 quantization.

## Repository layout

```text
apps/        CLI entry point
include/     public headers (one per module)
src/         implementation
tests/       CTest unit tests
scripts/     model download + upstream differential test
models/      bundled 260K GQA demo model + its tokenizer (large ones downloaded)
```

## Acknowledgments

- [Andrej Karpathy](https://github.com/karpathy) for
  [llama2.c](https://github.com/karpathy/llama2.c) (MIT) — the upstream this
  refactor is verified against — and the
  [TinyStories](https://huggingface.co/karpathy/tinyllamas) checkpoints.
- The bundled `tokenizer.bin`, `models/stories260K.bin`, and `models/tok512.bin`
  are redistributed from those projects under MIT.

## License

MIT — see [LICENSE](LICENSE) (upstream copyright preserved).
