# Copyright © 2026 Apple Inc.

import io
import unittest

import mlx.core as mx
import mlx.nn as nn
import mlx_tests
import numpy as np


class TestMetalCrossEntropy(mlx_tests.MLXTestCase):
    def setUp(self):
        super().setUp()
        if not mx.metal.is_available() or mx.default_device() != mx.gpu:
            self.skipTest("Metal GPU required")
        self.env = mlx_tests.scoped_env(MLX_METAL_CROSS_ENTROPY="1")
        self.env.__enter__()
        self.addCleanup(self.env.__exit__, None, None, None)

    def check_oracle(self, x, y, g):
        # Convert the represented input, including low-precision rounding.
        xn = np.array(x.astype(mx.float32)).astype(np.float64)
        yn = np.array(y).astype(np.int64)
        gn = np.array(g).astype(np.float64)
        shifted = xn - xn.max(axis=-1, keepdims=True)
        ex = np.exp(shifted)
        expected = (
            np.log(ex.sum(axis=-1))
            - np.take_along_axis(shifted, yn[..., None], axis=-1)[..., 0]
        )
        grad = ex / ex.sum(axis=-1, keepdims=True)
        np.put_along_axis(
            grad,
            (yn % xn.shape[-1])[..., None],
            np.take_along_axis(grad, (yn % xn.shape[-1])[..., None], -1) - 1,
            axis=-1,
        )
        grad *= gn[..., None]
        out, grads = mx.vjp(lambda a: mx.fast.cross_entropy(a, y), [x], [g])
        self.assertEqual(out[0].dtype, mx.float32)
        self.assertEqual(grads[0].dtype, x.dtype)
        np.testing.assert_allclose(np.array(out[0]), expected, rtol=2e-6, atol=2e-6)
        # Compare with the same final dtype rounding as the primitive.
        expected_grad = mx.array(grad.astype(np.float32)).astype(x.dtype)
        np.testing.assert_allclose(
            np.array(grads[0].astype(mx.float32)),
            np.array(expected_grad.astype(mx.float32)),
            rtol={mx.float32: 2e-5, mx.float16: 2e-3, mx.bfloat16: 2e-2}[x.dtype],
            atol={mx.float32: 2e-6, mx.float16: 2e-5, mx.bfloat16: 2e-4}[x.dtype],
        )

    def test_widths_dtypes_and_offsets(self):
        rng = np.random.default_rng(19)
        for dtype in [mx.float32, mx.float16, mx.bfloat16]:
            for width in [1, 7, 31, 32, 33, 255, 256, 257, 4096, 8193, 32769]:
                for offset in [0, 10000]:
                    with self.subTest(dtype=dtype, width=width, offset=offset):
                        x = mx.array(rng.normal(size=(3, width)) * 2 + offset).astype(
                            dtype
                        )
                        y = mx.array([0, width - 1, -1], mx.int64)
                        self.check_oracle(x, y, mx.array([0.25, -2.0, 0.0]))

    def test_strides_and_broadcasts(self):
        x = mx.arange(2 * 3 * 14, dtype=mx.float32).reshape(2, 3, 14)
        x = mx.sin(x).transpose(1, 0, 2)[..., ::2]
        y = mx.array([[0, 1, 2], [3, 4, 5]]).T
        g = mx.array([[0.5, -0.25, 2], [1, 0, -1]]).T
        self.check_oracle(x, y, g)
        self.check_oracle(
            mx.broadcast_to(x[:1], (4, 2, 7)),
            mx.broadcast_to(y[:1], (4, 2)),
            mx.broadcast_to(g[:1], (4, 2)),
        )
        self.check_oracle(x[::-1, :, ::-1], y[::-1], g[::-1])

    def test_scalar_and_empty_batch(self):
        self.check_oracle(mx.array([1.0, -2.0, 3.0]), mx.array(1), mx.array(-0.75))
        x = mx.zeros((0, 7))
        y = mx.zeros((0,), mx.int32)
        out, grad = mx.vjp(lambda a: mx.fast.cross_entropy(a, y), [x], [mx.zeros((0,))])
        mx.eval(out, grad)
        self.assertEqual(out[0].shape, (0,))
        self.assertEqual(grad[0].shape, (0, 7))

    def test_compile_and_loss_route(self):
        x = mx.arange(21, dtype=mx.float32).reshape(3, 7) / 5
        y = mx.array([0, 3, 6])
        w = mx.array([0.25, -1.0, 2.0])
        f = lambda a: nn.losses.cross_entropy(a, y, weights=w, reduction="sum")
        out, grad = mx.value_and_grad(f)(x)
        cout, cgrad = mx.compile(mx.value_and_grad(f))(x)
        np.testing.assert_allclose(np.array(cout), np.array(out), atol=1e-6)
        np.testing.assert_allclose(np.array(cgrad), np.array(grad), atol=1e-6)
        graph = io.StringIO()
        mx.export_to_dot(graph, nn.losses.cross_entropy(x, y))
        self.assertIn("CrossEntropy", graph.getvalue())
        with mlx_tests.scoped_env(MLX_METAL_CROSS_ENTROPY="0"):
            graph = io.StringIO()
            mx.export_to_dot(graph, nn.losses.cross_entropy(x, y))
            self.assertNotIn("CrossEntropy", graph.getvalue())

    def test_transforms(self):
        x = mx.arange(21, dtype=mx.float32).reshape(3, 7) / 7
        y = mx.array([0, 3, 6])
        f = lambda a: mx.fast.cross_entropy(a, y).sum()
        with mlx_tests.scoped_env(MLX_METAL_CROSS_ENTROPY="0"):
            expected = mx.grad(lambda a: mx.grad(f)(a).square().sum())(x)
            _, jexpected = mx.jvp(f, [x], [mx.sin(x)])
        actual = mx.grad(lambda a: mx.grad(f)(a).square().sum())(x)
        _, jactual = mx.jvp(f, [x], [mx.sin(x)])
        np.testing.assert_allclose(np.array(actual), np.array(expected), atol=1e-6)
        np.testing.assert_allclose(np.array(jactual), np.array(jexpected), atol=1e-6)
        mapped = mx.vmap(mx.fast.cross_entropy)(x, y)
        np.testing.assert_allclose(
            np.array(mapped), np.array(mx.fast.cross_entropy(x, y)), atol=1e-6
        )

    def test_stable_transform_fallbacks(self):
        x = mx.array([[10000.0, 10001.0, 9999.0], [10002.0, 10000.0, 10001.0]])
        y = mx.array([-1, -3])
        tangent = mx.array([[0.2, -0.5, 1.0], [-1.0, 0.3, 0.1]])

        def stable(a):
            z = a - mx.stop_gradient(a.max(axis=-1, keepdims=True))
            return mx.logsumexp(z, axis=-1) - mx.take_along_axis(
                z, y[:, None], -1
            ).squeeze(-1)

        f = lambda a: mx.fast.cross_entropy(a, y).sum()
        ref = lambda a: stable(a).sum()
        mapped = mx.vmap(mx.fast.cross_entropy)(x, y)
        np.testing.assert_allclose(np.array(mapped), np.array(stable(x)), atol=1e-6)
        _, actual = mx.jvp(f, [x], [tangent])
        _, expected = mx.jvp(ref, [x], [tangent])
        np.testing.assert_allclose(np.array(actual), np.array(expected), atol=1e-6)
        actual = mx.grad(lambda a: mx.sum(mx.square(mx.grad(f)(a))))(x)
        expected = mx.grad(lambda a: mx.sum(mx.square(mx.grad(ref)(a))))(x)
        np.testing.assert_allclose(np.array(actual), np.array(expected), atol=1e-6)

    def test_loss_fallbacks(self):
        x = mx.arange(21, dtype=mx.float32).reshape(3, 7) / 7
        for targets, kwargs in [
            (mx.full((3, 7), 1 / 7), {}),
            (mx.array([0, 3, 6]), {"label_smoothing": 0.2}),
            (mx.zeros((7,), mx.int32), {"axis": 0}),
        ]:
            graph = io.StringIO()
            out = nn.losses.cross_entropy(x, targets, **kwargs)
            mx.export_to_dot(graph, out)
            self.assertNotIn("CrossEntropy", graph.getvalue())
            with mlx_tests.scoped_env(MLX_METAL_CROSS_ENTROPY="0"):
                ref = nn.losses.cross_entropy(x, targets, **kwargs)
            np.testing.assert_array_equal(np.array(out), np.array(ref))
        graph = io.StringIO()
        out = mx.fast.cross_entropy(x, mx.zeros((3,), mx.int32), stream=mx.cpu)
        mx.export_to_dot(graph, out)
        self.assertNotIn("CrossEntropy", graph.getvalue())
        mx.eval(out)

    def test_nonfinite_and_invalid_targets(self):
        x = mx.array(
            [
                [float("-inf"), float("-inf")],
                [float("inf"), 0],
                [float("inf"), 0],
                [float("nan"), 0],
            ]
        )
        y = mx.array([0, 0, 1, 1])
        actual = mx.fast.cross_entropy(x, y)
        with mlx_tests.scoped_env(MLX_METAL_CROSS_ENTROPY="0"):
            ref = mx.fast.cross_entropy(x, y)
        np.testing.assert_allclose(np.array(actual), np.array(ref), equal_nan=True)
        x = mx.ones((2, 3))
        y = mx.array([-4, 3])
        out, grad = mx.vjp(lambda a: mx.fast.cross_entropy(a, y), [x], [mx.ones((2,))])
        self.assertTrue(mx.all(mx.isnan(out[0])).item())
        self.assertTrue(mx.all(mx.isnan(grad[0])).item())


if __name__ == "__main__":
    mlx_tests.MLXTestRunner()
