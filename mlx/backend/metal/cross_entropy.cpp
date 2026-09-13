// Copyright © 2026 Apple Inc.

#include <cstdlib>
#include <string_view>

#include "mlx/backend/gpu/copy.h"
#include "mlx/backend/metal/utils.h"
#include "mlx/fast_primitives.h"

namespace mlx::core::fast {

bool CrossEntropy::use_fallback(Stream s) {
  const char* enabled = std::getenv("MLX_METAL_CROSS_ENTROPY");
  return s.device == Device::cpu || !enabled || std::string_view(enabled) != "1";
}

namespace {
void cross_entropy_eval(
    const std::vector<array>& inputs,
    array& out,
    Stream s,
    bool backward) {
  out.set_data(allocator::malloc(out.nbytes()));
  if (out.size() == 0) {
    return;
  }
  auto& encoder = metal::get_command_encoder(s);
  auto contiguous = [&](const array& x) {
    if (x.flags().row_contiguous) {
      return x;
    }
    auto copy = contiguous_copy_gpu(x, s);
    encoder.add_temporary(copy);
    return copy;
  };
  auto x = contiguous(inputs[0]);
  auto y = contiguous(inputs[1]);
  auto& d = metal::device(s.device);
  auto name = std::string(backward ? "cross_entropy_vjp_" : "cross_entropy_") +
      type_to_name(x);
  auto kernel = d.get_kernel(name);
  int classes = x.shape(-1);
  size_t threads = std::min<size_t>(256, kernel->maxTotalThreadsPerThreadgroup());
  encoder.set_compute_pipeline_state(kernel);
  encoder.set_input_array(x, 0);
  encoder.set_input_array(y, 1);
  encoder.set_output_array(out, 2);
  encoder.set_bytes(classes, 3);
  if (backward) {
    auto g = contiguous(inputs[3]);
    encoder.set_input_array(g, 4);
  }
  encoder.dispatch_threads(
      MTL::Size(y.size() * threads, 1, 1), MTL::Size(threads, 1, 1));
}
} // namespace

void CrossEntropy::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  cross_entropy_eval(inputs, outputs[0], stream(), false);
}

void CrossEntropyVJP::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  cross_entropy_eval(inputs, outputs[0], stream(), true);
}

} // namespace mlx::core::fast
