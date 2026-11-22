#pragma once

namespace mini_llama::nn {

void rmsnorm(float* out, const float* x, const float* weight, int size);

void softmax(float* x, int size);

// out(d) = W(d, n) * x(n), with W stored row-major as d rows of n floats.
void matmul(float* out, const float* x, const float* w, int n, int d);

void rope(float* q, float* k, int pos, int dim, int kv_dim, int head_size);

}  // namespace mini_llama::nn
