from __future__ import annotations

import dataclasses
from typing import Callable, NamedTuple, Sequence

import jax
import jax.numpy as jnp
import numpy as np

from .gaussian import GaussianBeam, _inverse_2x2, apply_action_delta, taylor_expand


class QuadraticResidual(NamedTuple):
    """Residual of a local quadratic action model over one beam support."""

    max_abs: jnp.ndarray
    rms: jnp.ndarray
    points: jnp.ndarray
    residuals: jnp.ndarray


class QuadraticActionFit(NamedTuple):
    """Quadratic action coefficients fitted over one beam support."""

    dS0: jnp.ndarray
    dS1: jnp.ndarray
    dS2: jnp.ndarray
    fit_offsets: jnp.ndarray
    fit_values: jnp.ndarray


class PhaseResidualPolynomialFit(NamedTuple):
    """Polynomial fit of the phase residual left after the Taylor quadratic."""

    coeffs: jnp.ndarray
    powers: tuple[tuple[int, int], ...]
    dS0: jnp.ndarray
    dS1: jnp.ndarray
    dS2: jnp.ndarray
    eigvecs: jnp.ndarray
    radii: jnp.ndarray
    fit_offsets: jnp.ndarray
    fit_values: jnp.ndarray


def _as_single_gaussian(ray: GaussianBeam) -> GaussianBeam:
    if jnp.asarray(ray.r_xy).ndim > 1:
        raise ValueError("Expected a single Gaussian beam, got a batched bundle.")
    return ray


def _symmetrize_real(matrix: jnp.ndarray) -> jnp.ndarray:
    matrix = jnp.real(matrix)
    return 0.5 * (matrix + matrix.T)


def _symmetric_eigh_2x2(matrix: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Eigen-decomposition for a real symmetric 2x2 matrix without cuSolver."""

    matrix = _symmetrize_real(matrix)
    a = matrix[0, 0]
    b = matrix[0, 1]
    d = matrix[1, 1]
    delta = jnp.sqrt((a - d) ** 2 + 4.0 * b * b)
    eigvals = jnp.array(
        (0.5 * (a + d - delta), 0.5 * (a + d + delta)),
        dtype=matrix.dtype,
    )

    theta = 0.5 * jnp.atan2(2.0 * b, a - d)
    cos_t = jnp.cos(theta)
    sin_t = jnp.sin(theta)
    eigvec_small = jnp.array((-sin_t, cos_t), dtype=matrix.dtype)
    eigvec_large = jnp.array((cos_t, sin_t), dtype=matrix.dtype)
    eigvecs = jnp.stack((eigvec_small, eigvec_large), axis=1)
    return eigvals, eigvecs


def gaussian_envelope_precision(ray: GaussianBeam) -> jnp.ndarray:
    """Return the real amplitude precision ``k Im(Q_inv)`` for one beam."""

    ray = _as_single_gaussian(ray)
    return ray.k * _symmetrize_real(jnp.imag(ray.Q_inv))


def _support_frame(
    ray: GaussianBeam,
    *,
    floor: float = 1e-300,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    precision = gaussian_envelope_precision(ray)
    eigvals, eigvecs = _symmetric_eigh_2x2(precision)
    eigvals = jnp.maximum(eigvals, floor)
    radii = jnp.sqrt(2.0 / eigvals)
    return precision, eigvals, eigvecs, radii


def gaussian_1e_radii(ray: GaussianBeam, *, floor: float = 1e-300) -> jnp.ndarray:
    """Return principal 1/e amplitude radii of one Gaussian beam."""

    _, _, _, radii = _support_frame(ray, floor=floor)
    return radii


def _default_support_directions(dtype) -> jnp.ndarray:
    dirs = jnp.array(
        [
            [1.0, 0.0],
            [-1.0, 0.0],
            [0.0, 1.0],
            [0.0, -1.0],
            [1.0, 1.0],
            [1.0, -1.0],
            [-1.0, 1.0],
            [-1.0, -1.0],
        ],
        dtype=dtype,
    )
    return dirs / jnp.linalg.norm(dirs, axis=1, keepdims=True)


def gaussian_support_offsets(
    ray: GaussianBeam,
    *,
    support_radius: float = 1.0,
    directions: jnp.ndarray | None = None,
) -> jnp.ndarray:
    """Sample offsets on an elliptical beam support contour.

    ``support_radius=1`` samples the 1/e amplitude contour.  Larger values
    probe farther into the tails and are useful for conservative error tests.
    """

    ray = _as_single_gaussian(ray)
    _, _, eigvecs, radii = _support_frame(ray)
    radii = support_radius * radii

    if directions is None:
        directions = _default_support_directions(radii.dtype)
    directions = jnp.asarray(directions, dtype=radii.dtype)

    local_offsets = directions * radii[None, :]
    return local_offsets @ eigvecs.T


def quadratic_residual_on_beam(
    fn: Callable[..., jnp.ndarray],
    ray: GaussianBeam,
    *args,
    scale: float | jnp.ndarray | None = None,
    support_radius: float = 1.0,
    directions: jnp.ndarray | None = None,
    complex_valued: bool = False,
) -> QuadraticResidual:
    """Measure how non-quadratic ``fn`` is over one Gaussian beam.

    ``fn`` is Taylor-expanded to second order at the beam centre and sampled on
    the beam's natural support contour.  For optical path/action functions,
    pass ``scale=ray.k`` to report the residual in radians.
    """

    ray = _as_single_gaussian(ray)
    xy_ref = ray.r_xy
    offsets = gaussian_support_offsets(
        ray, support_radius=support_radius, directions=directions
    )
    points = xy_ref[None, :] + offsets
    if complex_valued:
        d0, d1, d2 = taylor_expand(fn, xy_ref, *args)
    else:
        def scalar_fn(xy, *fn_args):
            return jnp.real(fn(xy, *fn_args))

        d0 = scalar_fn(xy_ref, *args)
        d1 = jax.grad(scalar_fn)(xy_ref, *args)
        d2 = jax.hessian(scalar_fn)(xy_ref, *args)

    if scale is None:
        scale = 1.0

    def residual_at(offset, point):
        model = d0 + jnp.dot(d1, offset) + 0.5 * jnp.dot(offset, d2 @ offset)
        return scale * (fn(point, *args) - model)

    residuals = jax.vmap(residual_at)(offsets, points)
    abs_residuals = jnp.abs(residuals)
    return QuadraticResidual(
        max_abs=jnp.max(abs_residuals),
        rms=jnp.sqrt(jnp.mean(abs_residuals**2)),
        points=points,
        residuals=residuals,
    )


def estimate_width_scale_for_residual(
    residual: float | jnp.ndarray,
    *,
    target: float = 0.25,
    order: int = 3,
    min_scale: float = 0.1,
    max_scale: float = 1.0,
) -> jnp.ndarray:
    """Estimate beam-width reduction needed for a target residual.

    Cubic residuals scale approximately as width^3, quartic residuals as
    width^4, etc.  This helper is intentionally conservative and only provides
    an order-of-magnitude splitting guide.
    """

    residual = jnp.asarray(residual)
    safe = jnp.maximum(residual, 1e-300)
    scale = (target / safe) ** (1.0 / order)
    scale = jnp.where(residual <= target, max_scale, scale)
    return jnp.clip(scale, min_scale, max_scale)


def _grid_offsets(
    ray: GaussianBeam,
    grid_shape: tuple[int, int],
    support_radius: float,
) -> jnp.ndarray:
    if len(grid_shape) != 2:
        raise ValueError("`grid_shape` must contain two dimensions.")
    if grid_shape[0] <= 0 or grid_shape[1] <= 0:
        raise ValueError("`grid_shape` entries must be positive.")

    _, _, eigvecs, radii = _support_frame(ray)

    u = jnp.linspace(-support_radius, support_radius, grid_shape[0]) * radii[0]
    v = jnp.linspace(-support_radius, support_radius, grid_shape[1]) * radii[1]
    uu, vv = jnp.meshgrid(u, v, indexing="ij")
    local = jnp.stack((uu.ravel(), vv.ravel()), axis=-1)
    return local @ eigvecs.T


def _fit_grid_local(
    samples_per_axis: int,
    support_radius: float,
    dtype,
) -> jnp.ndarray:
    if samples_per_axis < 3:
        raise ValueError("`samples_per_axis` must be at least 3.")

    u = jnp.linspace(-support_radius, support_radius, samples_per_axis, dtype=dtype)
    v = jnp.linspace(-support_radius, support_radius, samples_per_axis, dtype=dtype)
    uu, vv = jnp.meshgrid(u, v, indexing="ij")
    return jnp.stack((uu.ravel(), vv.ravel()), axis=-1)


def _local_to_physical_offsets(
    local_offsets: jnp.ndarray,
    eigvecs: jnp.ndarray,
    radii: jnp.ndarray,
) -> jnp.ndarray:
    return (local_offsets * radii[None, :]) @ eigvecs.T


def _physical_to_local_offsets(
    offsets: jnp.ndarray,
    eigvecs: jnp.ndarray,
    radii: jnp.ndarray,
) -> jnp.ndarray:
    return (offsets @ eigvecs) / radii[None, :]


def _partition_fit_points_and_local(
    ray: GaussianBeam,
    *,
    support_radius: float,
    samples_per_axis: int,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    _, _, eigvecs, radii = _support_frame(ray)
    local = _fit_grid_local(samples_per_axis, support_radius, radii.dtype)
    offsets = _local_to_physical_offsets(local, eigvecs, radii)
    return offsets, local, eigvecs, radii


def _partition_fit_points(
    ray: GaussianBeam,
    *,
    support_radius: float,
    samples_per_axis: int,
) -> jnp.ndarray:
    offsets, _, _, _ = _partition_fit_points_and_local(
        ray,
        support_radius=support_radius,
        samples_per_axis=samples_per_axis,
    )
    return offsets


def _quadratic_design(offsets: jnp.ndarray) -> jnp.ndarray:
    x = offsets[:, 0]
    y = offsets[:, 1]
    return jnp.stack(
        (
            jnp.ones_like(x),
            x,
            y,
            0.5 * x * x,
            x * y,
            0.5 * y * y,
        ),
        axis=-1,
    )


def _polynomial_powers_2d(
    *,
    min_degree: int,
    max_degree: int,
) -> tuple[tuple[int, int], ...]:
    if min_degree < 0:
        raise ValueError("`min_degree` must be non-negative.")
    if max_degree < min_degree:
        raise ValueError("`max_degree` must be at least `min_degree`.")

    powers = []
    for total_degree in range(min_degree, max_degree + 1):
        powers.extend(
            (x_degree, total_degree - x_degree)
            for x_degree in range(total_degree + 1)
        )
    return tuple(powers)


def _polynomial_design_2d(
    local_offsets: jnp.ndarray,
    powers: Sequence[tuple[int, int]],
) -> jnp.ndarray:
    u = local_offsets[:, 0]
    v = local_offsets[:, 1]
    columns = [(u**u_power) * (v**v_power) for u_power, v_power in powers]
    return jnp.stack(columns, axis=-1)


def _fit_quadratic_coefficients_on_local_grid(
    local_offsets: jnp.ndarray,
    values: jnp.ndarray,
    *,
    weight_power: float | None = None,
) -> jnp.ndarray:
    """Fit ``c0 + c1 u + c2 v + .5 h11 u^2 + h12 uv + .5 h22 v^2``.

    The local offsets are a symmetric tensor-product grid, so the polynomial
    columns are orthogonal after centering the even quadratic terms.  This
    avoids dispatching tiny least-squares problems to cuSolver on GPU.

    If ``weight_power`` is provided, points are weighted by
    ``exp(-weight_power * (u**2 + v**2))``.  This biases the fitted action
    toward the high-amplitude beam core while preserving exact recovery of
    quadratic actions on the symmetric grid.
    """

    u = local_offsets[:, 0]
    v = local_offsets[:, 1]
    u2 = u * u
    v2 = v * v
    uv = u * v

    if weight_power is None:
        weights = jnp.ones_like(u)
    else:
        weights = jnp.exp(-jnp.asarray(weight_power, dtype=u.dtype) * (u2 + v2))

    weight_sum = jnp.sum(weights)

    def weighted_mean(x):
        return jnp.sum(weights * x) / weight_sum

    value_mean = weighted_mean(values)
    u2_mean = weighted_mean(u2)
    v2_mean = weighted_mean(v2)
    u2_centered = u2 - u2_mean
    v2_centered = v2 - v2_mean
    values_centered = values - value_mean

    c1 = jnp.sum(weights * values_centered * u) / jnp.sum(weights * u2)
    c2 = jnp.sum(weights * values_centered * v) / jnp.sum(weights * v2)
    h12 = jnp.sum(weights * values_centered * uv) / jnp.sum(weights * uv * uv)
    h11 = (
        2.0
        * jnp.sum(weights * values_centered * u2_centered)
        / jnp.sum(weights * u2_centered * u2_centered)
    )
    h22 = (
        2.0
        * jnp.sum(weights * values_centered * v2_centered)
        / jnp.sum(weights * v2_centered * v2_centered)
    )
    c0 = value_mean - 0.5 * h11 * u2_mean - 0.5 * h22 * v2_mean
    return jnp.stack((c0, c1, c2, h11, h12, h22), axis=0)


def fit_phase_residual_polynomial_on_beam(
    fn: Callable[..., jnp.ndarray],
    ray: GaussianBeam,
    *args,
    taylor_coefficients: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray] | None = None,
    min_degree: int = 3,
    max_degree: int = 4,
    fit_support_radius: float = 1.0,
    fit_samples_per_axis: int = 7,
    rcond: float | None = None,
) -> PhaseResidualPolynomialFit:
    """Fit the cubic-and-higher phase residual over one beam support.

    The returned polynomial is expressed in normalized local beam coordinates
    ``u, v``.  It models ``fn(x0 + dr) - Taylor2(fn, x0)(dr)`` and is intended
    for first-order Hermite/Gaussian corrections, not as a replacement for the
    Taylor Gaussian itself.
    """

    ray = _as_single_gaussian(ray)
    if min_degree < 3:
        raise ValueError("`min_degree` should be at least 3 for a Taylor residual.")

    xy_ref = ray.r_xy
    if taylor_coefficients is None:
        dS0, dS1, dS2 = taylor_expand(fn, xy_ref, *args)
    else:
        dS0, dS1, dS2 = taylor_coefficients

    fit_offsets, fit_offsets_local, eigvecs, radii = _partition_fit_points_and_local(
        ray,
        support_radius=fit_support_radius,
        samples_per_axis=fit_samples_per_axis,
    )
    fit_points = xy_ref[None, :] + fit_offsets
    fit_values = jax.vmap(lambda xy: fn(xy, *args))(fit_points)
    taylor_values = (
        dS0
        + fit_offsets @ dS1
        + 0.5 * jnp.einsum("pi,ij,pj->p", fit_offsets, dS2, fit_offsets)
    )
    residual_values = fit_values - taylor_values

    powers = _polynomial_powers_2d(
        min_degree=min_degree,
        max_degree=max_degree,
    )
    design = _polynomial_design_2d(fit_offsets_local, powers)
    coeffs = np.linalg.lstsq(
        np.asarray(design),
        np.asarray(residual_values),
        rcond=rcond,
    )[0]

    return PhaseResidualPolynomialFit(
        coeffs=jnp.asarray(coeffs, dtype=residual_values.dtype),
        powers=powers,
        dS0=dS0,
        dS1=dS1,
        dS2=dS2,
        eigvecs=eigvecs,
        radii=radii,
        fit_offsets=fit_offsets,
        fit_values=residual_values,
    )


def _local_coefficients_to_physical(
    coeffs: jnp.ndarray,
    eigvecs: jnp.ndarray,
    radii: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    inv_radii = 1.0 / radii
    local_grad = coeffs[1:3]
    local_hess = jnp.array(
        (
            (coeffs[3], coeffs[4]),
            (coeffs[4], coeffs[5]),
        ),
        dtype=coeffs.dtype,
    )
    dS1 = eigvecs @ (local_grad * inv_radii)
    dS2_local_scaled = local_hess * inv_radii[:, None] * inv_radii[None, :]
    dS2 = eigvecs @ dS2_local_scaled @ eigvecs.T
    return coeffs[0], dS1, dS2


def sampled_quadratic_residual_on_beam(
    fn: Callable[..., jnp.ndarray],
    ray: GaussianBeam,
    *args,
    scale: float | jnp.ndarray | None = None,
    support_radius: float = 1.0,
    fit_support_radius: float | None = None,
    fit_samples_per_axis: int = 5,
    fit_weight_power: float | None = None,
    directions: jnp.ndarray | None = None,
    rcond: float | None = None,
) -> QuadraticResidual:
    """Estimate non-quadratic residual by fitting sampled values.

    This avoids autodiff Hessians and is therefore the preferred diagnostic for
    phase functions with polar coordinates, branches, interpolation, or atomic
    potentials.  The returned residual is evaluated on the beam support contour.
    """

    del rcond
    ray = _as_single_gaussian(ray)
    xy_ref = ray.r_xy
    if scale is None:
        scale = 1.0
    if fit_support_radius is None:
        fit_support_radius = support_radius

    fit_offsets, fit_offsets_local, eigvecs, radii = _partition_fit_points_and_local(
        ray,
        support_radius=fit_support_radius,
        samples_per_axis=fit_samples_per_axis,
    )
    fit_points = xy_ref[None, :] + fit_offsets
    fit_values = jax.vmap(lambda xy: fn(xy, *args))(fit_points)
    coeffs = _fit_quadratic_coefficients_on_local_grid(
        fit_offsets_local,
        fit_values,
        weight_power=fit_weight_power,
    )

    support_offsets = gaussian_support_offsets(
        ray, support_radius=support_radius, directions=directions
    )
    eval_offsets = jnp.concatenate((fit_offsets, support_offsets), axis=0)
    eval_offsets_local = jnp.concatenate(
        (
            fit_offsets_local,
            _physical_to_local_offsets(support_offsets, eigvecs, radii),
        ),
        axis=0,
    )
    eval_points = xy_ref[None, :] + eval_offsets
    eval_values = jax.vmap(lambda xy: fn(xy, *args))(eval_points)
    residuals = scale * (
        eval_values - _quadratic_design(eval_offsets_local) @ coeffs
    )
    abs_residuals = jnp.abs(residuals)
    return QuadraticResidual(
        max_abs=jnp.max(abs_residuals),
        rms=jnp.sqrt(jnp.mean(abs_residuals**2)),
        points=eval_points,
        residuals=residuals,
    )


def fit_quadratic_action_on_beam(
    fn: Callable[..., jnp.ndarray],
    ray: GaussianBeam,
    *args,
    fit_support_radius: float = 1.0,
    fit_samples_per_axis: int = 5,
    fit_weight_power: float | None = None,
    rcond: float | None = None,
) -> QuadraticActionFit:
    """Fit a quadratic action over the beam support.

    ``fn`` must return an action/path-length contribution in the same units as
    ``ray.pathlength``.  Unlike a Taylor expansion, this fit uses values across
    the beam support, making it useful for sharp potentials where derivatives at
    the beam centre are a poor local model.  ``fit_weight_power`` applies
    Gaussian weights in normalized beam coordinates to bias the fit toward the
    beam core.
    """

    del rcond
    ray = _as_single_gaussian(ray)
    xy_ref = ray.r_xy
    fit_offsets, fit_offsets_local, eigvecs, radii = _partition_fit_points_and_local(
        ray,
        support_radius=fit_support_radius,
        samples_per_axis=fit_samples_per_axis,
    )
    fit_points = xy_ref[None, :] + fit_offsets
    fit_values = jax.vmap(lambda xy: fn(xy, *args))(fit_points)

    coeffs = _fit_quadratic_coefficients_on_local_grid(
        fit_offsets_local,
        fit_values,
        weight_power=fit_weight_power,
    )
    dS0, dS1, dS2 = _local_coefficients_to_physical(coeffs, eigvecs, radii)
    return QuadraticActionFit(
        dS0=dS0,
        dS1=dS1,
        dS2=dS2,
        fit_offsets=fit_offsets,
        fit_values=fit_values,
    )


def apply_fitted_quadratic_action(
    ray: GaussianBeam,
    fn: Callable[..., jnp.ndarray],
    *args,
    fit_support_radius: float = 1.0,
    fit_samples_per_axis: int = 5,
    fit_weight_power: float | None = None,
    rcond: float | None = None,
) -> GaussianBeam:
    """Apply a fitted local quadratic action to one Gaussian beam."""

    fit = fit_quadratic_action_on_beam(
        fn,
        ray,
        *args,
        fit_support_radius=fit_support_radius,
        fit_samples_per_axis=fit_samples_per_axis,
        fit_weight_power=fit_weight_power,
        rcond=rcond,
    )
    r_xy, d_xy, amplitude, pathlength, q_inv = apply_action_delta(
        ray, dS0=fit.dS0, dS1=fit.dS1, dS2=fit.dS2
    )
    return ray.derive(
        x=r_xy[0],
        y=r_xy[1],
        dx=d_xy[0],
        dy=d_xy[1],
        amplitude=amplitude,
        pathlength=pathlength,
        Q_inv=q_inv,
    )


def apply_taylor_phase_action(
    ray: GaussianBeam,
    fn: Callable[..., jnp.ndarray],
    *args,
) -> GaussianBeam:
    """Apply the Taylor quadratic action of ``fn`` to one Gaussian beam."""

    ray = _as_single_gaussian(ray)
    dS0, dS1, dS2 = taylor_expand(fn, ray.r_xy, *args)
    r_xy, d_xy, amplitude, pathlength, q_inv = apply_action_delta(
        ray,
        dS0=dS0,
        dS1=dS1,
        dS2=dS2,
    )
    return ray.derive(
        x=r_xy[0],
        y=r_xy[1],
        dx=d_xy[0],
        dy=d_xy[1],
        amplitude=amplitude,
        pathlength=pathlength,
        Q_inv=q_inv,
    )


def _raw_gaussian_moments_2d(
    mean: jnp.ndarray,
    covariance: jnp.ndarray,
    max_degree: int,
) -> dict[tuple[int, int], jnp.ndarray]:
    moments: dict[tuple[int, int], jnp.ndarray] = {
        (0, 0): jnp.ones(mean.shape[0], dtype=mean.dtype)
    }
    mu_u = mean[:, 0]
    mu_v = mean[:, 1]
    c_uu = covariance[0, 0]
    c_uv = covariance[0, 1]
    c_vv = covariance[1, 1]

    for total_degree in range(1, max_degree + 1):
        for u_degree in range(total_degree + 1):
            v_degree = total_degree - u_degree
            if u_degree > 0:
                value = mu_u * moments[(u_degree - 1, v_degree)]
                if u_degree >= 2:
                    value = value + (u_degree - 1) * c_uu * moments[
                        (u_degree - 2, v_degree)
                    ]
                if v_degree >= 1:
                    value = value + v_degree * c_uv * moments[
                        (u_degree - 1, v_degree - 1)
                    ]
            else:
                value = mu_v * moments[(0, v_degree - 1)]
                if v_degree >= 2:
                    value = value + (v_degree - 1) * c_vv * moments[
                        (0, v_degree - 2)
                    ]
            moments[(u_degree, v_degree)] = value
    return moments


def propagated_polynomial_moments(
    ray: GaussianBeam,
    coords: jnp.ndarray,
    distance: float,
    powers: Sequence[tuple[int, int]],
    eigvecs: jnp.ndarray,
    radii: jnp.ndarray,
) -> jnp.ndarray:
    """Return propagated input-plane moments for local monomials ``u**a v**b``.

    For a Gaussian propagated by ``distance``, multiplying the input field by a
    polynomial in the input-plane displacement is equivalent to multiplying the
    propagated Gaussian by the corresponding complex Gaussian moment.  The
    monomials are evaluated in normalized local coordinates defined by
    ``eigvecs`` and ``radii``.
    """

    ray = _as_single_gaussian(ray)
    if not powers:
        return jnp.zeros((coords.shape[0], 0), dtype=jnp.complex128)

    coords = jnp.asarray(coords, dtype=jnp.float64)
    distance = jnp.asarray(distance, dtype=jnp.float64)
    identity = jnp.eye(2, dtype=jnp.complex128)
    inv_a = _inverse_2x2(identity + distance * ray.Q_inv)

    output_center = ray.r_xy + distance * ray.d_xy
    output_offsets = coords - output_center[None, :]
    input_mean = output_offsets @ inv_a.T
    input_covariance = 1j * distance * inv_a / ray.k

    local_axes = eigvecs / radii[None, :]
    local_mean = input_mean @ local_axes
    local_covariance = local_axes.T @ input_covariance @ local_axes

    max_degree = max(sum(power) for power in powers)
    moments = _raw_gaussian_moments_2d(local_mean, local_covariance, max_degree)
    return jnp.stack([moments[power] for power in powers], axis=-1)


def first_order_phase_residual_correction_factor(
    ray: GaussianBeam,
    coords: jnp.ndarray,
    distance: float,
    fit: PhaseResidualPolynomialFit,
) -> jnp.ndarray:
    """Return the first-order propagated phase-residual correction factor.

    The corrected propagated field is ``base_field * (1 + factor)``, where
    ``base_field`` is the Taylor-propagated Gaussian field.  This implements the
    first-order term of ``exp(i k residual)`` and is mainly useful as a
    diagnostic for whether Hermite/Gaussian residual modes can improve Taylor.
    """

    moments = propagated_polynomial_moments(
        ray,
        coords,
        distance,
        fit.powers,
        fit.eigvecs,
        fit.radii,
    )
    residual_moment = jnp.sum(moments * fit.coeffs[None, :], axis=-1)
    return 1j * ray.k * residual_moment


def _fit_window_weights(
    window_centers: jnp.ndarray,
    window_precision: jnp.ndarray,
    fit_offsets: jnp.ndarray,
    *,
    ridge: float,
) -> jnp.ndarray:
    deltas = fit_offsets[:, None, :] - window_centers[None, :, :]
    exponents = -0.5 * jnp.einsum("pmi,ij,pmj->pm", deltas, window_precision, deltas)
    windows = jnp.exp(exponents)
    lhs = windows.T @ windows + ridge * jnp.eye(windows.shape[1], dtype=windows.dtype)
    rhs = windows.T @ jnp.ones((windows.shape[0],), dtype=windows.dtype)
    lhs_np = np.asarray(lhs)
    rhs_np = np.asarray(rhs)
    try:
        weights = np.linalg.solve(lhs_np, rhs_np)
    except np.linalg.LinAlgError:
        weights = np.linalg.lstsq(lhs_np, rhs_np, rcond=None)[0]
    return jnp.asarray(weights, dtype=windows.dtype)


def split_gaussian_realspace(
    ray: GaussianBeam,
    *,
    grid_shape: tuple[int, int] = (5, 5),
    support_radius: float = 1.1,
    child_width_scale: float = 0.6,
    fit_partition: bool = True,
    fit_support_radius: float | None = None,
    fit_samples_per_axis: int = 7,
    ridge: float = 1e-8,
) -> GaussianBeam:
    """Split one already-propagated Gaussian into a fitted child bundle.

    The split is implemented by multiplying the incoming packet with a local
    grid of real Gaussian windows.  Each product remains an exact Gaussian; the
    window weights are chosen by a small least-squares partition fit so that the
    sum of children reconstructs the parent over its significant support.
    """

    ray = _as_single_gaussian(ray)
    if not (0.0 < child_width_scale < 1.0):
        raise ValueError("`child_width_scale` must be between 0 and 1.")

    parent_precision = gaussian_envelope_precision(ray)
    precision_factor = child_width_scale ** -2 - 1.0
    window_precision = precision_factor * parent_precision

    child_offsets = _grid_offsets(ray, grid_shape, support_radius)
    center_fraction = 1.0 - child_width_scale**2
    window_centers = child_offsets / center_fraction

    if fit_partition:
        if fit_support_radius is None:
            fit_support_radius = max(1.25, support_radius)
        fit_offsets = _partition_fit_points(
            ray,
            support_radius=fit_support_radius,
            samples_per_axis=fit_samples_per_axis,
        )
        weights = _fit_window_weights(
            window_centers, window_precision, fit_offsets, ridge=ridge
        )
    else:
        weights = jnp.ones((window_centers.shape[0],), dtype=jnp.float64)
        center_windows = jnp.exp(
            -0.5
            * jnp.einsum(
                "mi,ij,mj->m", window_centers, window_precision, window_centers
            )
        )
        weights = weights / jnp.maximum(jnp.sum(center_windows), 1e-300)

    xy_ref = ray.r_xy
    k = ray.k
    hess_log_window = -window_precision
    dS2 = -1j * (hess_log_window / k)

    def child(center_offset, weight):
        center = xy_ref + center_offset
        delta = xy_ref - center
        log_w0 = -0.5 * jnp.dot(delta, window_precision @ delta)
        grad_log_w = -window_precision @ delta
        dS0 = -1j * (log_w0 / k)
        dS1 = -1j * (grad_log_w / k)
        r_xy, d_xy, amplitude, pathlength, q_new = apply_action_delta(
            ray, dS0=dS0, dS1=dS1, dS2=dS2
        )
        return r_xy, d_xy, amplitude * weight, pathlength, q_new

    r_xy, d_xy, amplitude, pathlength, q_inv = jax.vmap(child)(
        window_centers, weights
    )

    return ray.derive(
        x=r_xy[:, 0],
        y=r_xy[:, 1],
        dx=d_xy[:, 0],
        dy=d_xy[:, 1],
        amplitude=amplitude,
        pathlength=pathlength,
        Q_inv=q_inv,
    ).to_vector()


def concatenate_gaussian_beams(beams: Sequence[GaussianBeam]) -> GaussianBeam:
    """Concatenate scalar or batched Gaussian beams along the beam axis."""

    if not beams:
        raise ValueError("Cannot concatenate an empty beam sequence.")

    vectors = [beam.to_vector() for beam in beams]
    first = vectors[0]
    params = {}
    for field in dataclasses.fields(first):
        name = field.name
        value0 = getattr(first, name)
        if isinstance(value0, str) or value0 is None:
            params[name] = value0
            continue
        params[name] = jnp.concatenate(
            [jnp.asarray(getattr(beam, name)) for beam in vectors],
            axis=0,
        )
    return type(first)(**params)


def split_gaussian_bundle(
    ray: GaussianBeam,
    **kwargs,
) -> GaussianBeam:
    """Split every beam in a Gaussian bundle and flatten the children."""

    if jnp.asarray(ray.r_xy).ndim == 1:
        return split_gaussian_realspace(ray, **kwargs)

    vector = ray.to_vector()
    children = [
        split_gaussian_realspace(vector[i], **kwargs)
        for i in range(int(jnp.asarray(vector.x).shape[0]))
    ]
    return concatenate_gaussian_beams(children)


__all__ = [
    "QuadraticResidual",
    "QuadraticActionFit",
    "PhaseResidualPolynomialFit",
    "gaussian_envelope_precision",
    "gaussian_1e_radii",
    "gaussian_support_offsets",
    "quadratic_residual_on_beam",
    "sampled_quadratic_residual_on_beam",
    "fit_quadratic_action_on_beam",
    "apply_fitted_quadratic_action",
    "apply_taylor_phase_action",
    "fit_phase_residual_polynomial_on_beam",
    "propagated_polynomial_moments",
    "first_order_phase_residual_correction_factor",
    "estimate_width_scale_for_residual",
    "split_gaussian_realspace",
    "split_gaussian_bundle",
    "concatenate_gaussian_beams",
]
