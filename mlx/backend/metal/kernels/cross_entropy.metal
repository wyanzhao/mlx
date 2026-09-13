// Copyright © 2026 Apple Inc.

#include <metal_common>
#include <metal_simdgroup>
#include "mlx/backend/metal/kernels/utils.h"

using namespace metal;

template <typename T, bool backward>
[[kernel]] void cross_entropy(
    const device T* x,
    const device int* targets,
    device conditional_t<backward, T, float>* out,
    constant int& classes,
    const device float* cotangent,
    uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint threads [[threads_per_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd [[simdgroup_index_in_threadgroup]]) {
  threadgroup float partial[32];
  threadgroup float row_max;
  threadgroup float row_sum;
  x += size_t(row) * classes;
  float max_value = -INFINITY;
  for (size_t c = tid; c < size_t(classes); c += threads) {
    max_value = max(max_value, float(x[c]));
  }
  max_value = simd_max(max_value);
  if (lane == 0) {
    partial[simd] = max_value;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (simd == 0) {
    max_value = simd_max(lane < threads / 32 ? partial[lane] : -INFINITY);
    if (lane == 0) {
      row_max = max_value;
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float sum = 0;
  for (size_t c = tid; c < size_t(classes); c += threads) {
    sum += exp(float(x[c]) - row_max);
  }
  sum = simd_sum(sum);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (lane == 0) {
    partial[simd] = sum;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (simd == 0) {
    sum = simd_sum(lane < threads / 32 ? partial[lane] : 0.0f);
    if (lane == 0) {
      row_sum = sum;
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  int target = targets[row];
  target += target < 0 ? classes : 0;
  if (target < 0 || target >= classes) {
    if constexpr (backward) {
      out += size_t(row) * classes;
      for (size_t c = tid; c < size_t(classes); c += threads) {
        out[c] = T(NAN);
      }
    } else if (tid == 0) {
      out[row] = NAN;
    }
    return;
  }
  if constexpr (backward) {
    out += size_t(row) * classes;
    for (size_t c = tid; c < size_t(classes); c += threads) {
      float p = exp(float(x[c]) - row_max) / row_sum;
      out[c] = T(cotangent[row] * (p - float(c == size_t(target))));
    }
  } else if (tid == 0) {
    float gap = row_max - float(x[target]);
    out[row] = isinf(row_max) ? gap : gap + log(row_sum);
  }
}

#define instantiate_cross_entropy(name, type)                            \
  instantiate_kernel("cross_entropy_" #name, cross_entropy, type, false) \
      instantiate_kernel(                                                \
          "cross_entropy_vjp_" #name, cross_entropy, type, true)

instantiate_cross_entropy(float32, float)
    instantiate_cross_entropy(float16, half)
        instantiate_cross_entropy(bfloat16, bfloat16_t)
