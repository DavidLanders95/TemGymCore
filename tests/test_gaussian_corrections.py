import numpy as np
import jax
import jax.numpy as jnp

from temgym_core.components import Detector
from temgym_core.evaluate import evaluate_gaussians_for
from temgym_core.gaussian import make_gaussian
from temgym_core.gaussian_corrections import (
    estimate_width_scale_for_residual,
    fit_quadratic_action_on_beam,
    quadratic_residual_on_beam,
    sampled_quadratic_residual_on_beam,
    split_gaussian_realspace,
)


jax.config.update("jax_enable_x64", True)


def test_quadratic_residual_is_zero_for_quadratic_action():
    ray = make_gaussian(
        x=0.2e-6,
        y=-0.3e-6,
        voltage=200e3,
        waist_x=0.8e-6,
        waist_y=1.1e-6,
    )
    curvature = jnp.array([[2.0e4, -1.0e4], [-1.0e4, 3.0e4]])

    def quadratic(xy):
        return 0.7 + jnp.dot(jnp.array([1.0e-3, -2.0e-3]), xy) + 0.5 * (
            xy @ curvature @ xy
        )

    residual = quadratic_residual_on_beam(quadratic, ray, scale=ray.k)

    np.testing.assert_allclose(np.asarray(residual.max_abs), 0.0, atol=1e-3)


def test_quadratic_residual_detects_cubic_action():
    ray = make_gaussian(
        x=0.0,
        y=0.0,
        voltage=200e3,
        waist_x=1.0e-6,
        waist_y=1.0e-6,
    )

    def cubic(xy):
        return 1.0e12 * xy[0] ** 3

    residual = quadratic_residual_on_beam(cubic, ray, scale=ray.k)
    width_scale = estimate_width_scale_for_residual(residual.max_abs, target=0.25)

    assert float(residual.max_abs) > 1.0
    assert 0.1 <= float(width_scale) < 1.0


def test_sampled_quadratic_residual_matches_quadratic_exactly():
    ray = make_gaussian(
        x=-0.1e-6,
        y=0.15e-6,
        voltage=200e3,
        waist_x=0.8e-6,
        waist_y=1.1e-6,
    )

    def quadratic(xy):
        return 0.2 + 2.0e-3 * xy[0] - 3.0e-3 * xy[1] + 1.0e10 * xy[0] * xy[1]

    residual = sampled_quadratic_residual_on_beam(
        quadratic,
        ray,
        scale=ray.k,
        support_radius=1.0,
        fit_samples_per_axis=5,
    )

    np.testing.assert_allclose(np.asarray(residual.max_abs), 0.0, atol=1e-3)


def test_fit_quadratic_action_recovers_physical_coefficients():
    ray = make_gaussian(
        x=0.2e-6,
        y=-0.15e-6,
        voltage=200e3,
        waist_x=0.9e-6,
        waist_y=1.2e-6,
    )
    d0 = 0.3
    d1 = jnp.array([1.5e-3, -2.5e-3])
    d2 = jnp.array([[2.0e4, -0.7e4], [-0.7e4, 3.0e4]])

    def quadratic(xy):
        offset = xy - ray.r_xy
        return d0 + jnp.dot(d1, offset) + 0.5 * offset @ d2 @ offset

    fit = fit_quadratic_action_on_beam(
        quadratic,
        ray,
        fit_support_radius=1.0,
        fit_samples_per_axis=5,
    )

    np.testing.assert_allclose(np.asarray(fit.dS0), d0, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(np.asarray(fit.dS1), np.asarray(d1), rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(np.asarray(fit.dS2), np.asarray(d2), rtol=1e-10, atol=1e-3)


def test_split_gaussian_realspace_reconstructs_parent_field():
    ray = make_gaussian(
        x=0.2e-6,
        y=-0.1e-6,
        dx=1.0e-4,
        dy=-2.0e-4,
        voltage=200e3,
        waist_x=1.0e-6,
        waist_y=0.7e-6,
        rcurv_x=0.02,
        rcurv_y=-0.03,
    )
    children = split_gaussian_realspace(ray)

    grid = Detector(z=0.0, pixel_size=(0.06e-6, 0.06e-6), shape=(64, 64))
    parent_field = evaluate_gaussians_for(ray, grid)
    child_field = evaluate_gaussians_for(children, grid)

    parent_np = np.asarray(parent_field)
    child_np = np.asarray(child_field)
    support = np.abs(parent_np) > np.max(np.abs(parent_np)) * 1e-3
    rel_l2 = np.linalg.norm((child_np - parent_np)[support]) / np.linalg.norm(
        parent_np[support]
    )

    assert np.asarray(children.x).shape == (25,)
    assert rel_l2 < 0.03
