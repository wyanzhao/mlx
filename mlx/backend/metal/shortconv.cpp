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

bool FusedShortConvStep::use_fallback(Stream s) {
  return s.device == Device::cpu;
}

void FusedShortConvStep::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  assert(inputs.size() == 3);
  assert(outputs.size() == 2);

  auto& s = stream();
  auto& d = metal::device(s.device);
  auto& compute_encoder = metal::get_command_encoder(s);

  auto bcx = ensure_row_contiguous(inputs[0], s);
  bool donate_state =
      inputs[1].flags().row_contiguous &&
      inputs[1].is_donatable(outputs.size());
  auto state = ensure_row_contiguous(inputs[1], s);
  auto weight = ensure_row_contiguous(inputs[2], s);
  auto& y = outputs[0];
  auto& state_out = outputs[1];
  if (donate_state) {
    y.set_data(allocator::malloc(y.nbytes()));
    state_out.copy_shared_buffer(state);
  } else {
    y.set_data(allocator::malloc(y.nbytes() + state_out.nbytes()));
    state_out.copy_shared_buffer(
        y, state_out.strides(), state_out.flags(), state_out.size(), y.size());
  }

  uint32_t channels = state.shape(-1);
  uint32_t kernel_size = state.shape(1) + 1;
  uint32_t nthreads = state.shape(0) * channels;
  bool use_c2048_k3 = channels == 2048 && kernel_size == 3;
  std::string kernel_name = (use_c2048_k3 ? "fused_shortconv_step_c2048_k3_"
                                          : "fused_shortconv_step_") +
      type_to_name(y.dtype());
  auto kernel = d.get_kernel(kernel_name);

  compute_encoder.set_compute_pipeline_state(kernel);
  compute_encoder.set_input_array(bcx, 0);
  compute_encoder.set_input_array(state, 1);
  compute_encoder.set_input_array(weight, 2);
  compute_encoder.set_output_array(y, 3);
  compute_encoder.set_output_array(state_out, 4);
  if (!use_c2048_k3) {
    compute_encoder.set_bytes(channels, 5);
    compute_encoder.set_bytes(kernel_size, 6);
  }

  auto group_size = std::min<uint32_t>(256, nthreads);
  compute_encoder.dispatch_threads(
      MTL::Size(nthreads, 1, 1), MTL::Size(group_size, 1, 1));
}

} // namespace mlx::core::fast
