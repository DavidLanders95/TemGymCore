from __future__ import annotations

import dataclasses
from typing import Callable, NamedTuple, Sequence

import jax
import jax.numpy as jnp

from .gaussian import GaussianBeam, apply_action_delta, taylor_expand


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


def _as_single_gaussian(ray: GaussianBeam) -> GaussianBeam:
    if jnp.asarray(ray.r_xy).ndim > 1:
        raise ValueError("Expected a single Gaussian beam, got a batched bundle.")
    return ray


def _symmetrize_real(matrix: jnp.ndarray) -> jnp.ndarray:
    matrix = jnp.real(matrix)
    return 0.5 * (matrix + matrix.T)


def gaussian_envelope_precision(ray: GaussianBeam) -> jnp.ndarray:
    """Return the real amplitude precision ``k Im(Q_inv)`` for one beam."""

    ray = _as_single_gaussian(ray)
    return ray.k * _symmetrize_real(jnp.imag(ray.Q_inv))


def gaussian_1e_radii(ray: GaussianBeam, *, floor: float = 1e-300) -> jnp.ndarray:
    """Return principal 1/e amplitude radii of one Gaussian beam."""

    precision = gaussian_envelope_precision(ray)
    eigvals = jnp.linalg.eigvalsh(precision)
    eigvals = jnp.maximum(eigvals, floor)
    return jnp.sqrt(2.0 / eigvals)


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
    precision = gaussian_envelope_precision(ray)
    eigvals, eigvecs = jnp.linalg.eigh(precision)
    radii = support_radius * jnp.sqrt(2.0 / jnp.maximum(eigvals, 1e-300))

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

    precision = gaussian_envelope_precision(ray)
    eigvals, eigvecs = jnp.linalg.eigh(precision)
    radii = support_radius * jnp.sqrt(2.0 / jnp.maximum(eigvals, 1e-300))

    u = jnp.linspace(-1.0, 1.0, grid_shape[0]) * radii[0]
    v = jnp.linspace(-1.0, 1.0, grid_shape[1]) * radii[1]
    uu, vv = jnp.meshgrid(u, v, indexing="ij")
    local = jnp.stack((uu.ravel(), vv.ravel()), axis=-1)
    return local @ eigvecs.T


def _partition_fit_points(
    ray: GaussianBeam,
    *,
    support_radius: float,
    samples_per_axis: int,
) -> jnp.ndarray:
    if samples_per_axis < 2:
        raise ValueError("`samples_per_axis` must be at least 2.")

    precision = gaussian_envelope_precision(ray)
    eigvals, eigvecs = jnp.linalg.eigh(precision)
    radii = support_radius * jnp.sqrt(2.0 / jnp.maximum(eigvals, 1e-300))

    u = jnp.linspace(-1.0, 1.0, samples_per_axis) * radii[0]
    v = jnp.linspace(-1.0, 1.0, samples_per_axis) * radii[1]
    uu, vv = jnp.meshgrid(u, v, indexing="ij")
    local = jnp.stack((uu.ravel(), vv.ravel()), axis=-1)
    return local @ eigvecs.T


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


def sampled_quadratic_residual_on_beam(
    fn: Callable[..., jnp.ndarray],
    ray: GaussianBeam,
    *args,
    scale: float | jnp.ndarray | None = None,
    support_radius: float = 1.0,
    fit_support_radius: float | None = None,
    fit_samples_per_axis: int = 5,
    directions: jnp.ndarray | None = None,
    rcond: float | None = None,
) -> QuadraticResidual:
    """Estimate non-quadratic residual by fitting sampled values.

    This avoids autodiff Hessians and is therefore the preferred diagnostic for
    phase functions with polar coordinates, branches, interpolation, or atomic
    potentials.  The returned residual is evaluated on the beam support contour.
    """

    ray = _as_single_gaussian(ray)
    xy_ref = ray.r_xy
    if scale is None:
        scale = 1.0
    if fit_support_radius is None:
        fit_support_radius = support_radius

    fit_offsets = _partition_fit_points(
        ray,
        support_radius=fit_support_radius,
        samples_per_axis=fit_samples_per_axis,
    )
    coord_scale = jnp.maximum(jnp.max(jnp.abs(fit_offsets), axis=0), 1e-300)
    fit_offsets_scaled = fit_offsets / coord_scale
    fit_points = xy_ref[None, :] + fit_offsets
    fit_values = jax.vmap(lambda xy: fn(xy, *args))(fit_points)
    design = _quadratic_design(fit_offsets_scaled)
    coeffs = jnp.linalg.lstsq(design, fit_values, rcond=rcond)[0]

    support_offsets = gaussian_support_offsets(
        ray, support_radius=support_radius, directions=directions
    )
    eval_offsets = jnp.concatenate((fit_offsets, support_offsets), axis=0)
    eval_offsets_scaled = eval_offsets / coord_scale
    eval_points = xy_ref[None, :] + eval_offsets
    eval_values = jax.vmap(lambda xy: fn(xy, *args))(eval_points)
    residuals = scale * (
        eval_values - _quadratic_design(eval_offsets_scaled) @ coeffs
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
    rcond: float | None = None,
) -> QuadraticActionFit:
    """Fit a quadratic action over the beam support.

    ``fn`` must return an action/path-length contribution in the same units as
    ``ray.pathlength``.  Unlike a Taylor expansion, this fit uses values across
    the beam support, making it useful for sharp potentials where derivatives at
    the beam centre are a poor local model.
    """

    ray = _as_single_gaussian(ray)
    xy_ref = ray.r_xy
    fit_offsets = _partition_fit_points(
        ray,
        support_radius=fit_support_radius,
        samples_per_axis=fit_samples_per_axis,
    )
    coord_scale = jnp.maximum(jnp.max(jnp.abs(fit_offsets), axis=0), 1e-300)
    fit_offsets_scaled = fit_offsets / coord_scale
    fit_points = xy_ref[None, :] + fit_offsets
    fit_values = jax.vmap(lambda xy: fn(xy, *args))(fit_points)

    coeffs = jnp.linalg.lstsq(
        _quadratic_design(fit_offsets_scaled), fit_values, rcond=rcond
    )[0]

    sx, sy = coord_scale[0], coord_scale[1]
    dS0 = coeffs[0]
    dS1 = jnp.array([coeffs[1] / sx, coeffs[2] / sy])
    dS2 = jnp.array(
        [
            [coeffs[3] / (sx * sx), coeffs[4] / (sx * sy)],
            [coeffs[4] / (sx * sy), coeffs[5] / (sy * sy)],
        ]
    )
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
    rcond: float | None = None,
) -> GaussianBeam:
    """Apply a fitted local quadratic action to one Gaussian beam."""

    fit = fit_quadratic_action_on_beam(
        fn,
        ray,
        *args,
        fit_support_radius=fit_support_radius,
        fit_samples_per_axis=fit_samples_per_axis,
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
    return jnp.linalg.solve(lhs, rhs)


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
    "gaussian_envelope_precision",
    "gaussian_1e_radii",
    "gaussian_support_offsets",
    "quadratic_residual_on_beam",
    "sampled_quadratic_residual_on_beam",
    "fit_quadratic_action_on_beam",
    "apply_fitted_quadratic_action",
    "estimate_width_scale_for_residual",
    "split_gaussian_realspace",
    "split_gaussian_bundle",
    "concatenate_gaussian_beams",
]
