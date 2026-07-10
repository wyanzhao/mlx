// Copyright © 2026 Apple Inc.

#pragma once

#include <metal_stdlib>

using namespace metal;
using namespace mlx::steel;

// Phase-A-only gated-delta WY preparation for the Qwen3.5/3.6 shape used by
// the local M5 optimization run.  The implementation is deliberately
// specialized: B=1, Hk=16, Hv=32, Dk=Dv=128, chunk size 64.
//
// For a chunk, with D(i,j) = product(g[j+1:i+1]), it computes
//
//   A(i,j) = beta[i] * D(i,j) * dot(k[i], k[j]), i > j
//   (I + A) w = beta * gamma * k
//   (I + A) u = beta * v
//
// where gamma[i] = product(g[:i+1]).  Exact zero decays are resets.  We keep
// both a finite prefix-log (zeros contribute 0) and an inclusive zero count,
// so interval products never evaluate an undefined 0 / 0 gamma ratio.  The
// caller guarantees finite 0 <= g <= 1.  This M5-targeted debug
// specialization requires NAX and exactly 32 KiB of threadgroup memory.

template <typename T, int C, int D, int Hk, int Hv>
[[kernel]] void gated_delta_wy_prepare_nax(
    const device T* k [[buffer(0)]],
    const device T* v [[buffer(1)]],
    const device float* g [[buffer(2)]],
    const device T* beta [[buffer(3)]],
    device float* w [[buffer(4)]],
    device float* u [[buffer(5)]],
    device float* log_gamma [[buffer(6)]],
    uint3 tgp [[threadgroup_position_in_grid]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]],
    uint simd_lane_id [[thread_index_in_simdgroup]]) {
  static_assert(C == 64, "gated_delta_wy_prepare_nax requires C=64");
  static_assert(D == 128, "gated_delta_wy_prepare_nax requires D=128");
  static_assert(Hk == 16, "gated_delta_wy_prepare_nax requires Hk=16");
  static_assert(Hv == 32, "gated_delta_wy_prepare_nax requires Hv=32");

  constexpr int kFrag = 16;
  const int chunk = int(tgp.x);
  const int hv = int(tgp.y);
  const int hk = hv / (Hv / Hk);
  const int t0 = chunk * C;
  const int lid = int(simd_group_id * 32 + simd_lane_id);

  // A and a 64-column RHS slab stay fp32 through forward substitution.  The
  // two arrays consume exactly 32 KiB of threadgroup memory.  Before the solve,
  // the RHS allocation is reused for prefix metadata; gamma is then preserved
  // on A's otherwise-unused unit diagonal.
  threadgroup float mat[C * C];
  threadgroup float rhs_scratch[C * 64];
  threadgroup float* prefix_log = rhs_scratch;
  threadgroup ushort* prefix_zeros =
      reinterpret_cast<threadgroup ushort*>(rhs_scratch + C);

  // A single lane builds the exact reset-aware chunk prefix.  This is small
  // compared with the matrix work and avoids a separate prep dispatch.
  if (lid == 0) {
    float lp = 0.0f;
    ushort zeros = 0;
    for (int i = 0; i < C; ++i) {
      const float gi = g[(t0 + i) * Hv + hv];
      if (gi == 0.0f) {
        ++zeros;
      } else {
        lp += metal::precise::log(gi);
      }
      prefix_log[i] = lp;
      prefix_zeros[i] = zeros;
      log_gamma[(t0 + i) * Hv + hv] =
          zeros == 0 ? lp : -metal::numeric_limits<float>::infinity();
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // K K^T.  Two SIMDgroups independently produce one 64x32 column slice.
  // Keeping TN=2 uses NAX's paired-N path and avoids the paired-M path's
  // non-contiguous row-fragment layout.
  if (simd_group_id < 2) {
    NAXTile<float, 4, 2> kkt;
    kkt.clear();
    const device T* k_chunk = k + (size_t(t0) * Hk + hk) * D;
    for (int kk = 0; kk < D; kk += 32) {
      NAXTile<T, 4, 2> ka;
      NAXTile<T, 2, 2> kb;
      ka.load(k_chunk + kk, Hk * D);
      kb.load(
          k_chunk + size_t(simd_group_id) * 2 * kFrag * Hk * D + kk, Hk * D);
      tile_matmad_nax(
          kkt,
          ka,
          metal::bool_constant<false>{},
          kb,
          metal::bool_constant<true>{});
    }

    // Apply the strictly-lower mask, beta, and reset-safe interval decay.
    // For D(i,j), differing inclusive zero counts detect a reset in [j+1,i].
    for (short br = 0; br < 4; ++br) {
      for (short bc = 0; bc < 2; ++bc) {
        thread auto& frag = kkt.frag_at(br, bc);
        for (short e = 0; e < BaseNAXFrag::kElemsPerFrag; ++e) {
          const short2 coord = BaseNAXFrag::get_coord(e);
          const int row = int(br) * kFrag + int(coord.y);
          const int col =
              int(simd_group_id) * 2 * kFrag + int(bc) * kFrag + int(coord.x);
          float value = 0.0f;
          if (row > col && prefix_zeros[row] == prefix_zeros[col]) {
            const float decay =
                metal::precise::exp(prefix_log[row] - prefix_log[col]);
            value = float(beta[(t0 + row) * Hv + hv]) * decay * frag[e];
          }
          mat[row * C + col] = value;
        }
      }
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // The triangular solve never reads A's unit diagonal, so preserve gamma
  // there before rhs_scratch is reused for solved RHS blocks.
  if (lid < C) {
    mat[lid * C + lid] =
        prefix_zeros[lid] == 0 ? metal::precise::exp(prefix_log[lid]) : 0.0f;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // Solve in four 64-column waves (two for W, two for U).  Each SIMDgroup owns
  // 32 columns.  Off-diagonal 16x16 block updates use NAX's paired-N path;
  // the diagonal 16x16 triangular solve remains strict fp32 scalar arithmetic.
  for (int out_wave = 0; out_wave < 4; ++out_wave) {
    const bool make_w = out_wave < 2;
    const int col_half = make_w ? out_wave : out_wave - 2;
    const int out_col = col_half * 64 + int(simd_group_id) * 32;
    const int rhs_heads = make_w ? Hk : Hv;
    const int rhs_head = make_w ? hk : hv;
    const device T* rhs = make_w ? k : v;
    device float* dst = make_w ? w : u;

    for (int block_i = 0; block_i < 4; ++block_i) {
      const int row_base = block_i * kFrag;
      NAXTile<float, 1, 2> result;
      result.load(
          rhs + (size_t(t0 + row_base) * rhs_heads + rhs_head) * D + out_col,
          rhs_heads * D);

      // Scale the device RHS in registers before applying prior block rows.
      for (short bc = 0; bc < 2; ++bc) {
        thread auto& frag = result.frag_at(0, bc);
        for (short e = 0; e < BaseNAXFrag::kElemsPerFrag; ++e) {
          const short2 coord = BaseNAXFrag::get_coord(e);
          const int row = row_base + int(coord.y);
          float scale = float(beta[(t0 + row) * Hv + hv]);
          if (make_w) {
            scale *= mat[row * C + row];
          }
          frag[e] *= scale;
        }
      }

      for (int block_j = 0; block_j < block_i; ++block_j) {
        NAXTile<float, 1, 1> a_tile;
        NAXTile<float, 1, 2> x_tile;
        a_tile.template load<float, C, 1>(mat + row_base * C + block_j * kFrag);
        x_tile.template load<float, 64, 1>(
            rhs_scratch + block_j * kFrag * 64 + int(simd_group_id) * 32);
        thread auto& a_frag = a_tile.frag_at(0, 0);
        for (short e = 0; e < BaseNAXFrag::kElemsPerFrag; ++e) {
          a_frag[e] = -a_frag[e];
        }
        tile_matmad_nax(
            result,
            a_tile,
            metal::bool_constant<false>{},
            x_tile,
            metal::bool_constant<false>{});
      }

      result.template store<float, 64, 1>(
          rhs_scratch + row_base * 64 + int(simd_group_id) * 32);
      simdgroup_barrier(mem_flags::mem_threadgroup);

      // Strict fp32 forward substitution inside the diagonal 16x16 block.
      const int lane_col = int(simd_lane_id);
      for (int i = 0; i < kFrag; ++i) {
        float value = rhs_scratch
            [(row_base + i) * 64 + int(simd_group_id) * 32 + lane_col];
        for (int j = 0; j < i; ++j) {
          value -= mat[(row_base + i) * C + row_base + j] *
              rhs_scratch
                  [(row_base + j) * 64 + int(simd_group_id) * 32 + lane_col];
        }
        rhs_scratch[(row_base + i) * 64 + int(simd_group_id) * 32 + lane_col] =
            value;
        dst[(size_t(t0 + row_base + i) * Hv + hv) * D + out_col + lane_col] =
            value;
      }
      // The next NAX block load remaps matrix elements across lanes, so one
      // barrier is required after the complete diagonal block, not per row.
      simdgroup_barrier(mem_flags::mem_threadgroup);
    }
  }
}
