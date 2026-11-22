# mini-llama-cpp

[![CI](../../actions/workflows/ci.yml/badge.svg)](../../actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
![C++17](https://img.shields.io/badge/C%2B%2B-17-blue.svg)

**约 1600 行、零第三方依赖的 C++17 LLM 推理引擎** —— 对
[karpathy/llama2.c](https://github.com/karpathy/llama2.c) 中 `run.c`
的从零模块化重构：RAII 资源管理、异常安全的错误处理、完整单元测试。
只依赖标准库，OpenMP 可选。

**正确性是验证出来的，不是声称的**：贪心解码与固定种子的 top-p 采样，
在普通模型和 GQA 模型上都与上游 `run.c` **逐字节一致**。一条命令即可复现：
[`scripts/verify_upstream.sh`](scripts/verify_upstream.sh)。

English: [README.md](README.md)

## 演示

真实输出，`stories15M.bin`，temperature 0.8，seed 42（有截断）：

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

## 从 C 到 C++17 改了什么

数值计算完全不变，重构的是它周围的一切：

| 关注点 | 上游 `run.c` | 本重构 |
|---|---|---|
| 代码组织 | 单个约 1000 行文件 | 7 个模块，头文件在 `include/mini_llama/`，`namespace mini_llama` |
| checkpoint 加载 | `fread` 进 `malloc` 缓冲区 | 零拷贝 `mmap`，权重是 `const float*` 视图 |
| 资源清理 | 手动 `free`/`fclose` | RAII `Checkpoint`：仅可移动，构造函数异常安全 |
| 错误处理 | `fprintf` + `exit` | 带上下文的异常；config、文件大小、token/position 边界全部校验 |
| 词表查找 | 每次线性扫描 | 排序 id + `std::lower_bound`，O(log V) |
| 运行状态 | 裸 `malloc` 数组 | `std::vector<float>`，按 `Config` 一次性分配 |
| 并行 | OpenMP `#ifdef` 写死 | CMake 可选开关；输出与串行逐位一致 |
| 测试 | 无 | CTest：BPE 黄金用例、数学不变量、KV cache 不变量 |

## 架构

```text
apps/main.cpp            CLI：参数解析、组装
        │
        ▼
generator                generate() / chat() 循环、流式输出、tok/s 统计
        │
        ├──► model       Transformer：权重映射 + forward(token, pos)
        │      ├──► nn         rmsnorm / softmax / matmul / rope
        │      └──► checkpoint .bin 文件的 RAII mmap
        │             └──► config   头部解析 + 校验
        ├──► tokenizer   BPE 编解码、二分查找词表
        └──► sampler     argmax / 多项式 / top-p 采样，xorshift64* RNG
```

每个 token 的数据流（与上游一致）：

```text
token ─► embedding ─► [ RMSNorm ─► Q/K/V ─► RoPE ─► 基于 KV cache 的 attention ─► +残差
                      ─► RMSNorm ─► SwiGLU FFN ─► +残差 ] × n_layers
      ─► 最终 RMSNorm ─► 分类头 ─► logits ─► sampler ─► 下一个 token
```

支持分组查询注意力（GQA）：`n_kv_heads` 可以小于 `n_heads`，每个 KV 头被
`n_heads / n_kv_heads` 个查询头共享。仓库自带的 260K 模型（8 查询头 / 4 KV 头）
在 CI 中覆盖这条路径。

## 快速开始

需要 CMake ≥ 3.16、C++17 编译器、POSIX 系统（Linux/macOS）。

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j

# 开箱即用：仓库自带 260K 参数的 GQA 模型及其 tokenizer
./build/mini_llama models/stories260K.bin -z models/tok512.bin -i "One day" -n 256

# 想要连贯的故事，下载 15M 模型（约 58 MB）：
./scripts/download_model.sh
./build/mini_llama models/stories15M.bin -i "Once upon a time" -n 256

# 聊天模式（Llama-2 chat 模板）：
./build/mini_llama models/stories15M.bin -m chat -i "Tell me a story"

# 测试：
ctest --test-dir build --output-on-failure
```

Hugging Face 慢或被墙时可用镜像：
`HF_ENDPOINT=https://hf-mirror.com ./scripts/download_model.sh`。
`42M`、`110M` 的 checkpoint 同样支持。

### 命令行参数（与上游 `run.c` 相同）

| 参数 | 含义 | 默认值 |
|---|---|---|
| `-t` | temperature（0 = 贪心） | 1.0 |
| `-p` | top-p（核采样）截断 | 0.9 |
| `-s` | 随机种子（0 = 按时间） | 0 |
| `-n` | 最大步数（0 = `seq_len`） | 256 |
| `-i` | 提示词 | — |
| `-z` | tokenizer 路径 | `tokenizer.bin` |
| `-m` | `generate` \| `chat` | `generate` |
| `-y` | 系统提示词（chat 模式） | — |

## 正确性

三层验证，全部可在本地运行：

1. **与上游差分对比。**
   [`scripts/verify_upstream.sh`](scripts/verify_upstream.sh) 会克隆 llama2.c、
   编译 `run.c`，对贪心解码和固定种子 top-p 采样分别 diff 生成文本
   （RNG 同为 xorshift64\*，因此采样路径也一致）：

   ```text
   PASS: greedy output is byte-identical to upstream run.c
   PASS: sampled output is byte-identical to upstream run.c
   ```

2. **单元测试**（`ctest`）：
   - `test_tokenizer` —— BPE 黄金编码用例，含多字节 UTF-8；
   - `test_math` —— RMSNorm/softmax/matmul 对手算值、RoPE 保范数、
     固定种子下采样器确定性；
   - `test_model` —— logits 有限；生成推进时早期 KV cache 行不被改写
     （KV cache 的核心不变量）。在自带 GQA 模型上运行，存在
     `stories15M.bin` 时也在其上运行。

3. **Sanitizer。** 全套测试在 ASan + UBSan 下通过：

   ```bash
   cmake -S . -B build-asan -DCMAKE_BUILD_TYPE=Debug -DMINI_LLAMA_ENABLE_SANITIZERS=ON
   cmake --build build-asan -j
   ctest --test-dir build-asan --output-on-failure
   ```

CI 在每次 push 时运行 gcc/clang × {Release, OpenMP, ASan+UBSan}。

## 性能

可选的 OpenMP 并行化矩阵乘的行与注意力头这两个天然可并行的循环，
不改变归约顺序，因此输出与串行逐位一致：

```bash
cmake -S . -B build-omp -DCMAKE_BUILD_TYPE=Release -DMINI_LLAMA_ENABLE_OPENMP=ON
cmake --build build-omp -j
OMP_NUM_THREADS=8 ./build-omp/mini_llama models/stories15M.bin -n 256
```

实测环境：Intel Core Ultra 5 225H（Ubuntu 22.04 / WSL2，gcc 11.4，`-O3`），
`stories15M.bin`，256 步：

| 构建 | 线程数 | tok/s |
|---|---|---|
| Release 串行 | 1 | ~148 |
| Release + OpenMP | 4 | ~444 |
| Release + OpenMP | 8 | ~673 |

混合架构 CPU 上开满全部逻辑核可能反而比 8 线程慢（E 核调度），
建议按性能核数设置 `OMP_NUM_THREADS`。数字与机器强相关，
请用程序自带的 `achieved tok/s` 输出实测你自己的机器。

## 局限

- 仅 CPU、fp32、无 SIMD/BLAS —— 这是可读性优先的参考实现，不是性能纪录
  （性能取向请看 [llama.cpp](https://github.com/ggml-org/llama.cpp)）。
- 仅支持 llama2.c 旧版 **v0** checkpoint 格式；v1 文件会报明确错误。
- POSIX `mmap` 后端（Linux/macOS）；Windows 需要 `CreateFileMapping` 移植。
- BPE 编码沿用上游的 O(n²) 合并循环 —— 对提示词长度足够。

可能的后续方向：AVX2 矩阵乘、Windows 映射后端、v1/safetensors 加载、int8 量化。

## 目录结构

```text
apps/        CLI 入口
include/     公开头文件（每模块一个）
src/         实现
tests/       CTest 单元测试
scripts/     模型下载 + 上游差分验证
models/      自带 260K GQA 演示模型及其 tokenizer（大模型自行下载）
```

## 致谢

- [Andrej Karpathy](https://github.com/karpathy) 的
  [llama2.c](https://github.com/karpathy/llama2.c)（MIT）—— 本重构的差分验证基准，
  以及 [TinyStories](https://huggingface.co/karpathy/tinyllamas) 系列 checkpoint。
- 仓库中的 `tokenizer.bin`、`models/stories260K.bin`、`models/tok512.bin`
  按 MIT 协议从上述项目再分发。

## 许可证

MIT —— 见 [LICENSE](LICENSE)（保留上游版权声明）。
