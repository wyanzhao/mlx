// Copyright © 2026 Apple Inc.

#include "mlx/backend/metal/kernels/utils.h"
#include "mlx/backend/metal/kernels/steel/gemm/nax.h"
#include "mlx/backend/metal/kernels/gated_delta_nax.h"

instantiate_kernel(
    "gated_delta_wy_prepare_nax_bfloat16_t_c64_d128_hk16_hv32",
    gated_delta_wy_prepare_nax,
    bfloat16_t,
    64,
    128,
    16,
    32)
