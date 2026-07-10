// Copyright © 2023-2024 Apple Inc.

// clang-format off
#include "mlx/backend/metal/kernels/utils.h"
#include "mlx/backend/metal/kernels/steel/gemm/gemm.h"
#include "mlx/backend/metal/kernels/steel/gemm/nax.h"
#include "mlx/backend/metal/kernels/steel/gemm/loader.h"
#include "mlx/backend/metal/kernels/quantized_nax.h"

#define instantiate_quantized(name, type, group_size, bits, bm, bn, bk, wm, wn)  \
  instantiate_kernel(                                                    \
      #name "_" #type "_gs_" #group_size "_b_" #bits,                    \
      name,                                                              \
      type,                                                              \
      group_size,                                                        \
      bits, bm, bk, bn, wm, wn)

#define instantiate_quantized_batched(name, type, group_size, bits, bm, bn, bk, wm, wn, batched)     \
  instantiate_kernel(                                                    \
      #name "_" #type "_gs_" #group_size "_b_" #bits "_bm" #bm "_bn" #bn "_bk" #bk "_wm" #wm "_wn" #wn "_batch_" #batched, \
      name,                                                              \
      type,                                                              \
      group_size,                                                        \
      bits,                                                              \
      batched, bm, bk, bn, wm, wn)

#define instantiate_quantized_aligned(name, type, group_size, bits, bm, bn, bk, wm, wn, aligned)     \
  instantiate_kernel(                                                                     \
      #name "_" #type "_gs_" #group_size "_b_" #bits "_bm" #bm "_bn" #bn "_bk" #bk "_wm" #wm "_wn" #wn "_alN_" #aligned, \
      name,                                                                  \
      type,                                                                  \
      group_size,                                                            \
      bits,                                                                  \
      aligned, bm, bk, bn, wm, wn)

#define instantiate_quantized_aligned_batched(name, type, group_size, bits, bm, bn, bk, wm, wn, aligned, batched)     \
  instantiate_kernel(                                                                     \
      #name "_" #type "_gs_" #group_size "_b_" #bits "_bm" #bm "_bn" #bn "_bk" #bk "_wm" #wm "_wn" #wn "_alN_" #aligned "_batch_" #batched, \
      name,                                                                  \
      type,                                                                  \
      group_size,                                                            \
      bits,                                                                  \
      aligned,                                                               \
      batched, bm, bk, bn, wm, wn)

#define instantiate_gather_qmm_rhs(func, name, type, group_size, bits, bm, bn, bk, wm, wn, transpose)        \
  instantiate_kernel(                                                                                        \
      #name "_" #type "_gs_" #group_size "_b_" #bits "_bm_" #bm "_bn_" #bn "_bk_" #bk "_wm_" #wm "_wn_" #wn, \
      func,                                                         \
      type,                                                         \
      group_size,                                                   \
      bits,                                                         \
      bm,                                                           \
      bn,                                                           \
      bk,                                                           \
      wm,                                                           \
      wn,                                                           \
      transpose)

#define instantiate_quantized_batched_wrap(name, type, group_size, bits) \
  instantiate_quantized_batched(name, type, group_size, bits, 64, 64, 64, 2, 2, 1)      \
  instantiate_quantized_batched(name, type, group_size, bits, 64, 64, 64, 2, 2, 0)

#define instantiate_quantized_all_batched(type, group_size, bits) \
  instantiate_quantized_batched_wrap(affine_qmm_n_nax, type, group_size, bits)


#define instantiate_quantized_all_single(type, group_size, bits) \
  instantiate_quantized(affine_gather_qmm_n_nax, type, group_size, bits, 64, 64, 64, 2, 2)

#define instantiate_quantized_all_aligned(type, group_size, bits)   \
  instantiate_quantized_aligned(affine_gather_qmm_t_nax, type, group_size, bits, 64, 64, 64, 2, 2, true) \
  instantiate_quantized_aligned(affine_gather_qmm_t_nax, type, group_size, bits, 64, 64, 64, 2, 2, false) \
  instantiate_quantized_aligned_batched(affine_qmm_t_nax, type, group_size, bits, 64, 64, 64, 2, 2, true, 1) \
  instantiate_quantized_aligned_batched(affine_qmm_t_nax, type, group_size, bits, 64, 64, 64, 2, 2, true, 0) \
  instantiate_quantized_aligned_batched(affine_qmm_t_nax, type, group_size, bits, 64, 64, 64, 2, 2, false, 1) \
  instantiate_quantized_aligned_batched(affine_qmm_t_nax, type, group_size, bits, 64, 64, 64, 2, 2, false, 0)

#define instantiate_quantized_all_rhs(type, group_size, bits) \
  instantiate_gather_qmm_rhs(affine_gather_qmm_rhs_nax, affine_gather_qmm_rhs_nax_nt, type, group_size, bits, 64, 64, 64, 2, 2, true) \
  instantiate_gather_qmm_rhs(affine_gather_qmm_rhs_nax, affine_gather_qmm_rhs_nax_nn, type, group_size, bits, 64, 64, 64, 2, 2, false) \
  instantiate_gather_qmm_rhs(affine_gather_qmm_rhs_seg_nax, affine_gather_qmm_rhs_seg_nax_nt, type, group_size, bits, 64, 64, 64, 2, 2, true)

#define instantiate_quantized_funcs(type, group_size, bits) \
  instantiate_quantized_all_batched(type, group_size, bits) \
  instantiate_quantized_all_aligned(type, group_size, bits) \
  instantiate_quantized_all_rhs(type, group_size, bits)

#define instantiate_quantized_types(group_size, bits)       \
  instantiate_quantized_funcs(float, group_size, bits)      \
  instantiate_quantized_funcs(float16_t, group_size, bits)  \
  instantiate_quantized_funcs(bfloat16_t, group_size, bits)  

#define instantiate_quantized_groups(bits) \
  instantiate_quantized_types(128, bits)   \
  instantiate_quantized_types(64, bits)    \
  instantiate_quantized_types(32, bits)

#define instantiate_quantized_all() \
  instantiate_quantized_groups(2) \
  instantiate_quantized_groups(3) \
  instantiate_quantized_groups(4) \
  instantiate_quantized_groups(5) \
  instantiate_quantized_groups(6) \
  instantiate_quantized_groups(8)

instantiate_quantized_all() // clang-format on

// Build per-expert row segments and BM-tile offsets from sorted rhs indices.
// Single threadgroup; thread e lower-bounds expert e over the sorted indices.
[[kernel]] void moe_build_segments(
    const device uint32_t* indices [[buffer(0)]],
    device uint32_t* seg_offsets [[buffer(1)]],
    device uint32_t* tile_offsets [[buffer(2)]],
    const constant int& R [[buffer(3)]],
    const constant int& n_experts [[buffer(4)]],
    const constant int& bm [[buffer(5)]],
    uint tid [[thread_position_in_grid]]) {
  if ((int)tid <= n_experts) {
    uint32_t target = tid;
    int lo = 0, hi = R;
    while (lo < hi) {
      int mid = (lo + hi) >> 1;
      if (indices[mid] < target) {
        lo = mid + 1;
      } else {
        hi = mid;
      }
    }
    seg_offsets[tid] = (uint32_t)lo;
  }
  threadgroup_barrier(mem_flags::mem_device);
  if (tid == 0) {
    uint32_t acc = 0;
    for (int e = 0; e < n_experts; ++e) {
      tile_offsets[e] = acc;
      uint32_t c = seg_offsets[e + 1] - seg_offsets[e];
      acc += (c + bm - 1) / bm;
    }
    tile_offsets[n_experts] = acc;
  }
}
