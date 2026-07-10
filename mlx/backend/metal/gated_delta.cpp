// Copyright © 2026 Apple Inc.

#include "mlx/backend/common/compiled.h"
#include "mlx/backend/gpu/copy.h"
#include "mlx/backend/metal/device.h"
#include "mlx/backend/metal/kernels.h"
#include "mlx/backend/metal/utils.h"
#include "mlx/fast_primitives.h"

namespace mlx::core::fast {

bool GatedDeltaWYPrepare::use_fallback(Stream s) {
  return s.device == Device::cpu || !metal::is_nax_available();
}

void GatedDeltaWYPrepare::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  auto& s = stream();
  auto& d = metal::device(s.device);
  auto& compute_encoder = metal::get_command_encoder(s);

  std::vector<array> copies;
  copies.reserve(inputs.size());
  auto ensure_row_contiguous =
      [&copies, &compute_encoder, &s](const array& x) -> array {
    if (x.flags().row_contiguous) {
      return x;
    }
    auto copy = contiguous_copy_gpu(x, s);
    compute_encoder.add_temporary(copy);
    copies.push_back(copy);
    return copies.back();
  };

  const auto k = ensure_row_contiguous(inputs[0]);
  const auto v = ensure_row_contiguous(inputs[1]);
  const auto g = ensure_row_contiguous(inputs[2]);
  const auto beta = ensure_row_contiguous(inputs[3]);

  for (auto& out : outputs) {
    out.set_data(allocator::malloc(out.nbytes()));
  }

  constexpr int C = 64;
  constexpr int D = 128;
  constexpr int Hk = 16;
  constexpr int Hv = 32;
  const int T = static_cast<int>(k.shape(1));
  const std::string kernel_name =
      "gated_delta_wy_prepare_nax_bfloat16_t_c64_d128_hk16_hv32";
  const auto template_def = get_template_definition(
      kernel_name,
      "gated_delta_wy_prepare_nax",
      get_type_string(k.dtype()),
      C,
      D,
      Hk,
      Hv);
  auto kernel = get_gated_delta_nax_kernel(d, kernel_name, template_def);

  const MTL::Size group_dims(32, 2, 1);
  const MTL::Size grid_dims(T / C, Hv, 1);
  check_kernel_threadgroup_size(kernel, group_dims, kernel_name);

  compute_encoder.set_compute_pipeline_state(kernel);
  compute_encoder.set_input_array(k, 0);
  compute_encoder.set_input_array(v, 1);
  compute_encoder.set_input_array(g, 2);
  compute_encoder.set_input_array(beta, 3);
  compute_encoder.set_output_array(outputs[0], 4);
  compute_encoder.set_output_array(outputs[1], 5);
  compute_encoder.set_output_array(outputs[2], 6);
  compute_encoder.dispatch_threadgroups(grid_dims, group_dims);
}

} // namespace mlx::core::fast
