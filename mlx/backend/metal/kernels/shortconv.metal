// Copyright © 2026 Apple Inc.

#include <metal_stdlib>

#include "mlx/backend/metal/kernels/bf16.h"
#include "mlx/backend/metal/kernels/defines.h"

using namespace metal;

template <typename T>
inline float round_shortconv(float value);

template <>
inline float round_shortconv<bfloat16_t>(float value) {
  uint bits = as_type<uint>(value);
  bits += 0x7fffu + ((bits >> 16) & 1u);
  bits &= 0xffff0000u;
  return as_type<float>(bits);
}

template <>
inline float round_shortconv<half>(float value) {
  return float(half(value));
}

template <>
inline float round_shortconv<float>(float value) {
  return value;
}

template <typename T>
[[kernel]] void fused_shortconv_step(
    const device T* bcx [[buffer(0)]],
    const device T* state [[buffer(1)]],
    const device T* weight [[buffer(2)]],
    device T* y [[buffer(3)]],
    device T* state_out [[buffer(4)]],
    constant const uint& channels [[buffer(5)]],
    constant const uint& kernel_size [[buffer(6)]],
    uint gid [[thread_position_in_grid]]) {
  // state_out may alias a dead state input; each thread reads its channel
  // before shifting that same channel toward lower tap indices.
  const uint batch = gid / channels;
  const uint channel = gid % channels;
  const device T* gates = bcx + batch * 3 * channels;
  const float bx = round_shortconv<T>(
      float(gates[channel]) * float(gates[2 * channels + channel]));

  const device T* state_row = state + batch * (kernel_size - 1) * channels;
  const device T* channel_weight = weight + channel * kernel_size;
  float acc = 0.0f;
  for (uint tap = 0; tap + 1 < kernel_size; ++tap) {
    acc +=
        float(state_row[tap * channels + channel]) * float(channel_weight[tap]);
  }
  acc += bx * float(channel_weight[kernel_size - 1]);
  const float conv = round_shortconv<T>(acc);
  y[batch * channels + channel] = static_cast<T>(
      round_shortconv<T>(float(gates[channels + channel]) * conv));

  device T* next_state = state_out + batch * (kernel_size - 1) * channels;
  for (uint tap = 0; tap + 2 < kernel_size; ++tap) {
    next_state[tap * channels + channel] =
        state_row[(tap + 1) * channels + channel];
  }
  next_state[(kernel_size - 2) * channels + channel] = static_cast<T>(bx);
}

template <typename T>
[[kernel]] void fused_shortconv_step_c2048_k3(
    const device T* bcx [[buffer(0)]],
    const device T* state [[buffer(1)]],
    const device T* weight [[buffer(2)]],
    device T* y [[buffer(3)]],
    device T* state_out [[buffer(4)]],
    uint gid [[thread_position_in_grid]]) {
  constexpr uint channels = 2048;
  const uint batch = gid / channels;
  const uint channel = gid % channels;
  const device T* gates = bcx + batch * 3 * channels;
  const float bx = round_shortconv<T>(
      float(gates[channel]) * float(gates[2 * channels + channel]));

  const device T* state_row = state + batch * 2 * channels;
  const device T* channel_weight = weight + channel * 3;
  // Keep the generic path's initial +0 so signed-zero bits remain identical.
  float acc = 0.0f;
  acc += float(state_row[channel]) * float(channel_weight[0]);
  acc += float(state_row[channels + channel]) * float(channel_weight[1]);
  acc += bx * float(channel_weight[2]);
  const float conv = round_shortconv<T>(acc);
  y[batch * channels + channel] = static_cast<T>(
      round_shortconv<T>(float(gates[channels + channel]) * conv));

  device T* next_state = state_out + batch * 2 * channels;
  next_state[channel] = state_row[channels + channel];
  next_state[channels + channel] = static_cast<T>(bx);
}

// clang-format off
#define instantiate_shortconv(name, type) \
  instantiate_kernel(                    \
      "fused_shortconv_step_" #name, fused_shortconv_step, type)

instantiate_shortconv(float32, float)
instantiate_shortconv(float16, half)
instantiate_shortconv(bfloat16, bfloat16_t)

#define instantiate_shortconv_c2048_k3(name, type) \
  instantiate_kernel(                             \
      "fused_shortconv_step_c2048_k3_" #name,     \
      fused_shortconv_step_c2048_k3,               \
      type)

instantiate_shortconv_c2048_k3(float32, float)
instantiate_shortconv_c2048_k3(float16, half)
instantiate_shortconv_c2048_k3(bfloat16, bfloat16_t) // clang-format on
