// Copyright © 2026 Apple Inc.

#include <algorithm>

#include "mlx/backend/gpu/copy.h"
#include "mlx/backend/metal/device.h"
#include "mlx/backend/metal/utils.h"
#include "mlx/fast_primitives.h"

namespace mlx::core::fast {

namespace {

array ensure_row_contiguous(const array& x, const Stream& s) {
  if (x.flags().row_contiguous) {
    return x;
  }
  auto copy = contiguous_copy_gpu(x, s);
  metal::get_command_encoder(s).add_temporary(copy);
  return copy;
}

} // namespace

bool MoERoute::use_fallback(Stream s) {
  return s.device == Device::cpu;
}

void MoERoute::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  assert(inputs.size() == 2 || inputs.size() == 3);
  assert(outputs.size() == 2);

  auto& s = stream();
  auto& d = metal::device(s.device);
  auto& compute_encoder = metal::get_command_encoder(s);

  // House convention across the metal backend: never encode a zero-dimension
  // grid.
  if (outputs[0].size() == 0) {
    outputs[0].set_data(allocator::malloc(0));
    outputs[1].set_data(allocator::malloc(0));
    return;
  }

  auto x = ensure_row_contiguous(inputs[0], s);
  auto gate_w = ensure_row_contiguous(inputs[1], s);
  const bool has_bias = inputs.size() == 3;
  // With no bias the kernel never dereferences this, but it is typed as x's
  // dtype, so bind x rather than the float32 router weight.
  auto expert_bias = has_bias ? ensure_row_contiguous(inputs[2], s) : x;

  auto& inds = outputs[0];
  auto& scores = outputs[1];
  inds.set_data(allocator::malloc(inds.nbytes()));
  scores.set_data(allocator::malloc(scores.nbytes()));

  uint32_t dims = x.shape(-1);
  uint32_t num_experts = gate_w.shape(0);
  uint32_t top_k = top_k_;
  uint32_t tokens = x.size() / std::max<uint32_t>(dims, 1);
  float routed_scaling = routed_scaling_;
  uint32_t flags = (has_bias ? 1u : 0u) | (norm_topk_prob_ ? 2u : 0u);

  auto kernel = d.get_kernel("moe_route_" + type_to_name(scores.dtype()));
  compute_encoder.set_compute_pipeline_state(kernel);
  compute_encoder.set_input_array(x, 0);
  compute_encoder.set_input_array(gate_w, 1);
  compute_encoder.set_input_array(expert_bias, 2);
  compute_encoder.set_output_array(inds, 3);
  compute_encoder.set_output_array(scores, 4);
  compute_encoder.set_bytes(dims, 5);
  compute_encoder.set_bytes(num_experts, 6);
  compute_encoder.set_bytes(top_k, 7);
  compute_encoder.set_bytes(routed_scaling, 8);
  compute_encoder.set_bytes(flags, 9);

  // One threadgroup per token; the kernel fixes the group at 8 simdgroups and
  // strides experts across them, so the shape of the router does not change
  // the launch geometry.
  constexpr uint32_t kGroup = 8 * 32;
  compute_encoder.dispatch_threadgroups(
      MTL::Size(tokens, 1, 1), MTL::Size(kGroup, 1, 1));
}

} // namespace mlx::core::fast
