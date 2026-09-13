# Copyright © 2026 Apple Inc.

import os
import unittest
from unittest.mock import patch

import mlx.core as mx
import mlx_tests
import numpy as np


class TestEdgePartition(mlx_tests.MLXTestCase):
    def setUp(self):
        super().setUp()
        if not mx.metal.is_available() or mx.default_device() != mx.gpu:
            self.skipTest("Metal edge partition requires the GPU device")

    def check_partition(self, x, kth, axis=-1):
        original = np.array(x.astype(mx.float32))
        axis %= original.ndim
        n = original.shape[axis]
        k = kth % n
        partitioned = mx.partition(x, kth, axis)
        values = np.array(partitioned.astype(mx.float32))
        indices = np.array(mx.argpartition(x, kth, axis))
        if x.dtype in (mx.float32, mx.float16):
            bits = np.uint32 if x.dtype == mx.float32 else np.uint16
            native_input = np.array(x).view(bits)
            native_output = np.array(partitioned).view(bits)
            np.testing.assert_array_equal(
                native_output, np.take_along_axis(native_input, indices, axis)
            )
        gathered = np.take_along_axis(original, indices, axis)
        np.testing.assert_array_equal(values.view(np.uint32), gathered.view(np.uint32))
        expected = np.sort(original, axis=axis)
        expected_indices = np.broadcast_to(
            np.arange(n), np.moveaxis(original, axis, -1).shape
        )
        expected_indices = np.moveaxis(expected_indices, -1, axis)
        np.testing.assert_array_equal(np.sort(indices, axis=axis), expected_indices)
        for result in (values, gathered):
            # Compare bit multisets so signs of zero and NaN payloads survive.
            np.testing.assert_array_equal(
                np.sort(result.view(np.uint32), axis=axis),
                np.sort(original.view(np.uint32), axis=axis),
            )
            np.testing.assert_array_equal(
                np.take(result, k, axis), np.take(expected, k, axis)
            )
            left = np.take(result, np.arange(k), axis)
            right = np.take(result, np.arange(k + 1, n), axis)
            pivot = np.take(result, [k], axis)
            self.assertTrue(np.all((left <= pivot) | np.isnan(pivot)))
            self.assertTrue(np.all((right >= pivot) | np.isnan(right)))
        self.assertEqual(mx.argpartition(x, kth, axis).dtype, mx.uint32)
        self.assertEqual(mx.partition(x, kth, axis).dtype, x.dtype)

    def test_edges(self):
        rng = np.random.default_rng(14)
        with patch.dict(os.environ, {"MLX_METAL_EDGE_PARTITION": "1"}):
            for dtype in (mx.float32, mx.float16, mx.bfloat16):
                for width in (32, 33, 64, 128, 160, 256):
                    for batch in (1, 8, 1024):
                        x = mx.array(rng.normal(size=(batch, width)), dtype)
                        for kth in (0, 1, 7, -1, -2, -8, -width):
                            with self.subTest(
                                dtype=dtype, width=width, batch=batch, kth=kth
                            ):
                                self.check_partition(x, kth)

    def test_special_values(self):
        bits = np.array(
            [
                0,
                0x80000000,
                0x7F800000,
                0xFF800000,
                0x7FC00001,
                0x7FC00002,
                0xFFC00001,
                0x3F800000,
            ],
            dtype=np.uint32,
        )
        with patch.dict(os.environ, {"MLX_METAL_EDGE_PARTITION": "1"}):
            for dtype in (mx.float32, mx.float16, mx.bfloat16):
                for values in (
                    np.tile(bits.view(np.float32), 20),
                    np.full(160, np.nan),
                    np.zeros(160),
                    np.ones(160),
                    np.full(160, -np.inf),
                ):
                    for kth in (0, 7, -1, -8):
                        self.check_partition(mx.array(values, dtype), kth)

    def test_layouts_and_fallbacks(self):
        rng = np.random.default_rng(15)
        x = mx.array(rng.normal(size=(3, 64, 5)), mx.float32)
        with patch.dict(os.environ, {"MLX_METAL_EDGE_PARTITION": "1"}):
            for view, axis in (
                (x, 1),
                (x[::-1, ::-1, ::-1], 1),
                (x.transpose(1, 2, 0), 0),
                (mx.broadcast_to(x[:1], (8, 64, 5)), 1),
                (x[:, ::2, :], 1),
            ):
                for kth in (0, 7, -1, -8):
                    self.check_partition(view, kth, axis)
            for dtype in (mx.int32, mx.int64, mx.uint32, mx.float32):
                for width, kth in ((16, 1), (257, 1), (64, 31)):
                    self.check_partition(
                        mx.array(rng.integers(-5, 5, (8, width)), dtype), kth
                    )

    def test_complex_fallback(self):
        rng = np.random.default_rng(18)
        x = mx.array(rng.normal(size=(8, 64)) + 1j * rng.normal(size=(8, 64)))
        with patch.dict(os.environ, {"MLX_METAL_EDGE_PARTITION": "1"}):
            for kth in (0, 7, -1, -8):
                np.testing.assert_array_equal(
                    np.array(mx.partition(x, kth)), np.array(mx.sort(x))
                )
                np.testing.assert_array_equal(
                    np.array(mx.argpartition(x, kth)), np.array(mx.argsort(x))
                )

    def test_topk_and_transforms(self):
        rng = np.random.default_rng(16)
        a = rng.normal(size=(8, 64)).astype(np.float32)
        x = mx.array(a)
        with patch.dict(os.environ, {"MLX_METAL_EDGE_PARTITION": "1"}):
            for k in (1, 4, 8):
                np.testing.assert_array_equal(
                    np.sort(np.array(mx.topk(x, k)), axis=-1),
                    np.sort(a, axis=-1)[:, -k:],
                )
                actual = mx.vmap(lambda row: mx.partition(row, -k))(x)
                np.testing.assert_array_equal(
                    np.sort(np.array(actual), axis=-1), np.sort(a, axis=-1)
                )
                gradient = mx.grad(lambda z: mx.sum(mx.topk(z, k)))(x)
                expected = np.zeros_like(a)
                np.put_along_axis(expected, np.argsort(a, axis=-1)[:, -k:], 1, -1)
                np.testing.assert_array_equal(np.array(gradient), expected)

    def test_runtime_switch(self):
        x = mx.array(np.random.default_rng(17).normal(size=(8, 64)), mx.float32)
        for enabled in ("0", "1", "0", "1"):
            with patch.dict(os.environ, {"MLX_METAL_EDGE_PARTITION": enabled}):
                self.check_partition(x, -4)
                result = np.array(mx.partition(x, -4))
                sorted_input = np.sort(np.array(x), axis=-1)
                if enabled == "0":
                    np.testing.assert_array_equal(result, sorted_input)
                else:
                    self.assertFalse(np.array_equal(result, sorted_input))


if __name__ == "__main__":
    mlx_tests.MLXTestRunner()
