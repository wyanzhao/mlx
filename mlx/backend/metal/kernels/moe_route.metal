// Copyright © 2026 Apple Inc.

#include <metal_simdgroup>
#include <metal_stdlib>

#include "mlx/backend/metal/kernels/bf16.h"
#include "mlx/backend/metal/kernels/defines.h"

using namespace metal;

// Sparse-MoE routing, one threadgroup per token.
//
// Replaces the ~10 serial dispatches an MoE block spends deciding WHICH
// experts to run: router matmul, sigmoid, expert-bias add, top-k partition,
// score gather, sum-normalise, scale, and the casts between them. None of
// those move meaningful bytes -- the router weight is 32 x 2048 -- so their
// cost is the launch chain, and on LFM2.5-8B-A1B that chain measured 13.12%
// of the decode step.
//
// The selection reads scores WITH the expert bias and the emitted weights come
// from the UNBIASED probabilities, which is the reference's behaviour: the
// bias steers load balancing without distorting the mixture.
//
// Indices are emitted in descending selected-score order. argpartition leaves
// the order inside the top-k unspecified, so a fixed descending order is the
// stable contract; the expert SET matches, and the block output is invariant
// to the permutation up to floating-point summation order.

#define MOE_ROUTE_MAX_EXPERTS 1024
#define MOE_ROUTE_MAX_TOPK 32
#define MOE_ROUTE_SIMDS 8
#define MOE_ROUTE_THREADS (MOE_ROUTE_SIMDS * 32)

template <typename T>
[[kernel]] void moe_route(
    const device T* x [[buffer(0)]],
    const device float* gate_w [[buffer(1)]],
    // Same dtype as x, which is how a checkpoint stores it -- LFM2.5-8B-A1B
    // carries a bfloat16 expert_bias next to a float32 router weight, and
    // reading it as float32 silently corrupts every selection.
    const device T* expert_bias [[buffer(2)]],
    device uint32_t* inds_out [[buffer(3)]],
    device T* scores_out [[buffer(4)]],
    constant const uint& dims [[buffer(5)]],
    constant const uint& num_experts [[buffer(6)]],
    constant const uint& top_k [[buffer(7)]],
    constant const float& routed_scaling [[buffer(8)]],
    constant const uint& flags [[buffer(9)]],
    uint token [[threadgroup_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  const bool has_bias = (flags & 1u) != 0u;
  const bool norm_topk = (flags & 2u) != 0u;

  threadgroup float probs[MOE_ROUTE_MAX_EXPERTS];

  const device T* xt = x + uint64_t(token) * dims;

  // One simdgroup per expert, strided so num_experts need not divide the
  // simdgroup count. Each lane walks the row with a stride of 32, which keeps
  // consecutive lanes on consecutive floats -- the coalesced pattern.
  for (uint e = simd_gid; e < num_experts; e += MOE_ROUTE_SIMDS) {
    const device float* row = gate_w + uint64_t(e) * dims;
    float partial = 0.0f;
    for (uint d = simd_lid; d < dims; d += 32) {
      partial = metal::fma(float(xt[d]), row[d], partial);
    }
    const float dot = simd_sum(partial);
    if (simd_lid == 0) {
      probs[e] = 1.0f / (1.0f + metal::precise::exp(-dot));
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // Selecting 4 of 32 is ~128 comparisons; a parallel reduction would cost
  // more in barriers than it saves, so one thread does it.
  if (lid != 0) {
    return;
  }

  uint chosen[MOE_ROUTE_MAX_TOPK];
  float weights[MOE_ROUTE_MAX_TOPK];
  float total = 0.0f;

  for (uint slot = 0; slot < top_k; ++slot) {
    uint best = 0;
    float best_score = -INFINITY;
    for (uint e = 0; e < num_experts; ++e) {
      bool taken = false;
      for (uint prev = 0; prev < slot; ++prev) {
        taken = taken || (chosen[prev] == e);
      }
      if (taken) {
        continue;
      }
      // Bias steers SELECTION only; the weight below comes from probs[e].
      const float score =
          has_bias ? probs[e] + float(expert_bias[e]) : probs[e];
      if (score > best_score) {
        best_score = score;
        best = e;
      }
    }
    chosen[slot] = best;
    weights[slot] = probs[best];
    total += probs[best];
  }

  const float denom = norm_topk ? (total + 1e-6f) : 1.0f;
  device uint32_t* inds = inds_out + uint64_t(token) * top_k;
  device T* scores = scores_out + uint64_t(token) * top_k;
  for (uint slot = 0; slot < top_k; ++slot) {
    inds[slot] = chosen[slot];
    scores[slot] = static_cast<T>(weights[slot] / denom * routed_scaling);
  }
}

// clang-format off
#define instantiate_moe_route(name, type) \
  instantiate_kernel("moe_route_" #name, moe_route, type)

instantiate_moe_route(float32, float)
instantiate_moe_route(float16, half)
instantiate_moe_route(bfloat16, bfloat16_t) // clang-format on
