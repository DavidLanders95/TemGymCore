from __future__ import annotations

import os
import time
import argparse
from dataclasses import dataclass, replace
from functools import partial
from pathlib import Path
from typing import Any

os.environ.setdefault("JAX_ENABLE_X64", "1")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.85")

import jax
import jax.numpy as jnp
import numpy as np

from temgym_core.components import Detector
from temgym_core.evaluate import evaluate_gaussians_fast
from temgym_core.gaussian import GaussianBeam, propagate_free_space_jit
from temgym_core.gaussian_corrections import (
    apply_fitted_quadratic_action,
    concatenate_gaussian_beams,
)
from temgym_core.potential import potential_smoothed_from_r2
from temgym_core.source import square_input_wave
from temgym_core.utils import FresnelPropagator

jax.config.update("jax_enable_x64", True)


AU_LOBATO = jnp.array(
    [
        [
            1.6759346706487,
            3.00486602969729,
            0.595340013161635,
            0.0117163186623094,
            4.296782976398e-05,
        ],
        [
            5.52231093211402,
            1.38007223007196,
            0.162229237655945,
            0.00901814890416575,
            0.00037927766747767,
        ],
    ],
    dtype=jnp.float64,
)


@dataclass(frozen=True)
class MultisliceBenchmarkConfig:
    voltage: float = 200e3
    lattice_constant: float = 4.078
    n_cells_xy: int = 9
    n_slices: int = 10
    grid_shape: int = 192
    grid_margin: float = 6.0
    target_beams_side: int = 224
    beam_overlap: float = 2.0
    interaction_thickness: float | None = None
    propagation_per_slice: float | None = None
    potential_tail_cutoff: float = 6.0
    projection_samples: int = 9
    fit_support_radius: float = 1.0
    fit_samples_per_axis: int = 3
    beam_chunk_size: int = 1792
    eval_method: str = "auto"
    eval_tile_pixels: int = 64
    eval_tile_beams: int = 16
    reference_jit: bool = True
    record_slices: bool = False


def block_until_ready(value):
    def block_leaf(x):
        return x.block_until_ready() if hasattr(x, "block_until_ready") else x

    return jax.tree_util.tree_map(block_leaf, value)


def timed_call(fn):
    t0 = time.perf_counter()
    value = fn()
    block_until_ready(value)
    return value, time.perf_counter() - t0


def beam_count(beam: GaussianBeam) -> int:
    return int(np.asarray(beam.to_vector().x).size)


def aligned_field_error(test, ref) -> float:
    alpha = jnp.vdot(test, ref) / jnp.vdot(test, test)
    return float(jnp.linalg.norm(alpha * test - ref) / jnp.linalg.norm(ref))


def normalized_intensity_error(test, ref) -> float:
    test_i = jnp.abs(test) ** 2
    ref_i = jnp.abs(ref) ** 2
    test_i = test_i / jnp.sum(test_i)
    ref_i = ref_i / jnp.sum(ref_i)
    return float(jnp.linalg.norm(test_i - ref_i) / jnp.linalg.norm(ref_i))


def make_fcc_au_001_slices(
    *,
    n_cells_xy: int,
    n_slices: int,
    lattice_constant: float,
) -> jnp.ndarray:
    """Return Au [001] atom-plane coordinates with shape (slices, atoms, xy)."""

    a = float(lattice_constant)
    cell_width = n_cells_xy * a
    planes: list[list[tuple[float, float]]] = []
    for slice_index in range(n_slices):
        if slice_index % 2 == 0:
            basis = ((0.0, 0.0), (0.5 * a, 0.5 * a))
        else:
            basis = ((0.0, 0.5 * a), (0.5 * a, 0.0))

        atoms: list[tuple[float, float]] = []
        for ix in range(n_cells_xy):
            for iy in range(n_cells_xy):
                for bx, by in basis:
                    atoms.append(
                        (
                            ix * a + bx - 0.5 * cell_width,
                            iy * a + by - 0.5 * cell_width,
                        )
                    )
        planes.append(atoms)
    return jnp.asarray(planes, dtype=jnp.float64)


def make_grid_and_beam(config: MultisliceBenchmarkConfig):
    sample_width = config.n_cells_xy * config.lattice_constant
    extent = 0.5 * sample_width + config.grid_margin
    pixel = 2.0 * extent / config.grid_shape
    grid = Detector(z=0.0, pixel_size=(pixel, pixel), shape=(config.grid_shape, config.grid_shape))

    aperture_length = 2.0 * extent
    waist = aperture_length * config.beam_overlap / config.target_beams_side
    beam = square_input_wave(
        aperture_length=aperture_length,
        waist=waist,
        voltage=config.voltage,
        overlap_factor=config.beam_overlap,
        wavelength_unit="angstrom",
    )
    return grid, beam, extent, waist


def resolve_interaction_thickness(
    config: MultisliceBenchmarkConfig,
    propagation_distance: float,
) -> float:
    if config.interaction_thickness is not None:
        return float(config.interaction_thickness)
    return float(propagation_distance)


def projection_quadrature(thickness: float, samples: int) -> tuple[jnp.ndarray, jnp.ndarray]:
    if thickness <= 0.0:
        raise ValueError("`interaction_thickness` must be positive.")
    if samples < 1:
        raise ValueError("`projection_samples` must be at least 1.")

    nodes, weights = np.polynomial.legendre.leggauss(samples)
    z_offsets = 0.5 * thickness * nodes
    z_weights = 0.5 * thickness * weights
    return (
        jnp.asarray(z_offsets, dtype=jnp.float64),
        jnp.asarray(z_weights, dtype=jnp.float64),
    )


def _slice_action(
    xy,
    sigma,
    k,
    slice_xy,
    z_offsets,
    z_weights,
    smoothing_radius,
    potential_tail_cutoff,
    element_params,
):
    delta = xy[None, :] - slice_xy
    transverse_r2 = jnp.sum(delta * delta, axis=-1)

    def potential_at_z(z, weight):
        r2 = transverse_r2 + z * z
        values = jax.vmap(
            lambda r2_i: potential_smoothed_from_r2(
                r2_i, element_params, smoothing_radius
            )
        )(r2)
        keep = (potential_tail_cutoff <= 0.0) | (r2 <= potential_tail_cutoff**2)
        return weight * jnp.sum(jnp.where(keep, values, 0.0))

    projected_potential = jnp.sum(jax.vmap(potential_at_z)(z_offsets, z_weights))
    return -(sigma / k) * projected_potential


@partial(jax.jit, static_argnames=("shape",))
def _slice_action_grid(
    coords,
    shape,
    slice_xy,
    sigma,
    k,
    z_offsets,
    z_weights,
    smoothing_radius,
    potential_tail_cutoff,
    element_params,
):
    action = jax.vmap(
        lambda xy: _slice_action(
            xy,
            sigma,
            k,
            slice_xy,
            z_offsets,
            z_weights,
            smoothing_radius,
            potential_tail_cutoff,
            element_params,
        )
    )(coords)
    return action.reshape(shape)


def build_action_grids(
    grid,
    beam: GaussianBeam,
    slice_xy_stack,
    config: MultisliceBenchmarkConfig,
    *,
    interaction_thickness: float,
    smoothing_radius: float,
    element_params=AU_LOBATO,
):
    vector = beam.to_vector()
    sigma = vector.sigma[0]
    k = vector.k[0]
    z_offsets, z_weights = projection_quadrature(
        interaction_thickness, config.projection_samples
    )
    action_grids = []
    times = []
    for slice_index in range(slice_xy_stack.shape[0]):
        action_grid, elapsed = timed_call(
            lambda slice_xy=slice_xy_stack[slice_index]: _slice_action_grid(
                grid.coords,
                grid.shape,
                slice_xy,
                sigma,
                k,
                z_offsets,
                z_weights,
                smoothing_radius,
                config.potential_tail_cutoff,
                element_params,
            )
        )
        action_grids.append(action_grid)
        times.append(elapsed)
    return jnp.stack(action_grids, axis=0), times


def _bilinear_sample(image, xy, x0, y0, pixel_x, pixel_y):
    h, w = image.shape
    px = (xy[0] - x0) / pixel_x
    py = (xy[1] - y0) / pixel_y
    px = jnp.clip(px, 0.0, w - 1.0)
    py = jnp.clip(py, 0.0, h - 1.0)

    ix0 = jnp.floor(px).astype(jnp.int32)
    iy0 = jnp.floor(py).astype(jnp.int32)
    ix1 = jnp.minimum(ix0 + 1, w - 1)
    iy1 = jnp.minimum(iy0 + 1, h - 1)
    wx = px - ix0
    wy = py - iy0

    v00 = image[iy0, ix0]
    v10 = image[iy0, ix1]
    v01 = image[iy1, ix0]
    v11 = image[iy1, ix1]
    return (
        (1.0 - wx) * (1.0 - wy) * v00
        + wx * (1.0 - wy) * v10
        + (1.0 - wx) * wy * v01
        + wx * wy * v11
    )


@partial(
    jax.jit,
    static_argnames=("fit_support_radius", "fit_samples_per_axis"),
)
def _apply_fitted_slice_to_vector(
    vector,
    slice_xy,
    z_offsets,
    z_weights,
    smoothing_radius,
    potential_tail_cutoff,
    element_params,
    fit_support_radius,
    fit_samples_per_axis,
):
    def one_ray(ray):
        return apply_fitted_quadratic_action(
            ray,
            lambda xy: _slice_action(
                xy,
                ray.sigma,
                ray.k,
                slice_xy,
                z_offsets,
                z_weights,
                smoothing_radius,
                potential_tail_cutoff,
                element_params,
            ),
            fit_support_radius=fit_support_radius,
            fit_samples_per_axis=fit_samples_per_axis,
        )

    return jax.vmap(one_ray)(vector)


@partial(
    jax.jit,
    static_argnames=("fit_support_radius", "fit_samples_per_axis"),
)
def _apply_fitted_action_grid_to_vector(
    vector,
    action_grid,
    x0,
    y0,
    pixel_x,
    pixel_y,
    fit_support_radius,
    fit_samples_per_axis,
):
    def one_ray(ray):
        return apply_fitted_quadratic_action(
            ray,
            lambda xy: _bilinear_sample(action_grid, xy, x0, y0, pixel_x, pixel_y),
            fit_support_radius=fit_support_radius,
            fit_samples_per_axis=fit_samples_per_axis,
        )

    return jax.vmap(one_ray)(vector)


def _beam_chunks(beam: GaussianBeam, chunk_size: int):
    vector = beam.to_vector()
    n = beam_count(vector)
    for start in range(0, n, chunk_size):
        yield vector[start : min(start + chunk_size, n)]


def _concat_chunks(chunks: list[GaussianBeam]) -> GaussianBeam:
    if len(chunks) == 1:
        return chunks[0].to_vector()
    return concatenate_gaussian_beams(chunks).to_vector()


def apply_fitted_action_grid_chunked(
    beam: GaussianBeam,
    action_grid,
    *,
    x0: float,
    y0: float,
    pixel_x: float,
    pixel_y: float,
    fit_support_radius: float = 1.0,
    fit_samples_per_axis: int = 3,
    chunk_size: int = 2048,
) -> GaussianBeam:
    chunks = []
    for chunk in _beam_chunks(beam, chunk_size):
        out = _apply_fitted_action_grid_to_vector(
            chunk,
            action_grid,
            x0,
            y0,
            pixel_x,
            pixel_y,
            fit_support_radius,
            fit_samples_per_axis,
        )
        chunks.append(block_until_ready(out.to_vector()))
    return _concat_chunks(chunks)


def propagate_chunked(beam: GaussianBeam, distance: float, *, chunk_size: int) -> GaussianBeam:
    chunks = []
    for chunk in _beam_chunks(beam, chunk_size):
        chunks.append(block_until_ready(propagate_free_space_jit(chunk, distance).to_vector()))
    return _concat_chunks(chunks)


def _reference_step_eager(
    field,
    action_grid,
    pixel_size,
    wavelength,
    k,
    propagation_distance,
):
    transmitted = field * jnp.exp(1j * k * action_grid)
    return FresnelPropagator(
        transmitted,
        L=pixel_size * field.shape[0],
        wavelength=wavelength,
        z=propagation_distance,
    )


_reference_step_jit = jax.jit(_reference_step_eager)


def reference_multislice(
    input_field,
    grid,
    beam: GaussianBeam,
    action_grid_stack,
    config: MultisliceBenchmarkConfig,
    *,
    propagation_distance: float,
):
    vector = beam.to_vector()
    wavelength = vector.wavelength[0]
    k = vector.k[0]
    step = _reference_step_jit if config.reference_jit else _reference_step_eager

    field = input_field
    per_slice = []
    field_history = [] if config.record_slices else None
    for slice_index in range(action_grid_stack.shape[0]):
        field, t_step = timed_call(
            lambda action_grid=action_grid_stack[slice_index], field=field: step(
                field,
                action_grid,
                grid.pixel_size[0],
                wavelength,
                k,
                propagation_distance,
            )
        )
        per_slice.append(t_step)
        if field_history is not None:
            field_history.append(field)
    return field, per_slice, field_history


def gaussian_multislice(
    beam: GaussianBeam,
    action_grid_stack,
    config: MultisliceBenchmarkConfig,
    *,
    grid,
    propagation_distance: float,
):
    current = beam.to_vector()
    action_times = []
    propagation_times = []
    field_history = [] if config.record_slices else None
    field_eval_times = []
    x_coords, y_coords = grid.coords_1d
    x0 = x_coords[0]
    y0 = y_coords[0]
    pixel_x = grid.pixel_size[1]
    pixel_y = grid.pixel_size[0]

    for slice_index in range(action_grid_stack.shape[0]):
        current, t_action = timed_call(
            lambda action_grid=action_grid_stack[slice_index], current=current: apply_fitted_action_grid_chunked(
                current,
                action_grid,
                x0=x0,
                y0=y0,
                pixel_x=pixel_x,
                pixel_y=pixel_y,
                fit_support_radius=config.fit_support_radius,
                fit_samples_per_axis=config.fit_samples_per_axis,
                chunk_size=config.beam_chunk_size,
            )
        )
        current, t_prop = timed_call(
            lambda current=current: propagate_chunked(
                current,
                propagation_distance,
                chunk_size=config.beam_chunk_size,
            )
        )
        action_times.append(t_action)
        propagation_times.append(t_prop)
        if field_history is not None:
            field_snapshot, t_eval = timed_call(lambda current=current: evaluate_field(current, grid, config))
            field_history.append(field_snapshot)
            field_eval_times.append(t_eval)
        else:
            t_eval = None
        print(
            f"slice {slice_index + 1:02d}/{action_grid_stack.shape[0]}: "
            f"fit={t_action:.3f}s, propagate={t_prop:.3f}s"
            + (f", eval={t_eval:.3f}s" if t_eval is not None else "")
        )
    return current.to_vector(), action_times, propagation_times, field_history, field_eval_times


def evaluate_field(beam: GaussianBeam, grid, config: MultisliceBenchmarkConfig):
    return evaluate_gaussians_fast(
        beam.to_vector(),
        grid,
        method=config.eval_method,
        tile_pixels=config.eval_tile_pixels,
        tile_beams=config.eval_tile_beams,
    )


def action_to_potential_grid_stack(
    action_grid_stack,
    beam: GaussianBeam,
):
    vector = beam.to_vector()
    return -(vector.k[0] / vector.sigma[0]) * action_grid_stack


def run_benchmark(config: MultisliceBenchmarkConfig) -> dict[str, Any]:
    propagation_distance = (
        config.propagation_per_slice
        if config.propagation_per_slice is not None
        else 0.5 * config.lattice_constant
    )
    interaction_thickness = resolve_interaction_thickness(config, propagation_distance)
    grid, beam, extent, waist = make_grid_and_beam(config)
    slice_xy_stack = make_fcc_au_001_slices(
        n_cells_xy=config.n_cells_xy,
        n_slices=config.n_slices,
        lattice_constant=config.lattice_constant,
    )
    smoothing_radius = grid.pixel_size[0] / 3.0

    print("JAX backend:", jax.default_backend())
    print(f"Grid: {grid.shape}, pixel={grid.pixel_size[0]:.4f} A")
    print(f"Au [001] slices: {config.n_slices}, atoms/slice={slice_xy_stack.shape[1]}")
    print(f"TEM Gaussian beam count: {beam_count(beam)}; waist={waist:.4f} A")
    print(f"Propagation per slice: {propagation_distance:.4f} A")
    print(
        f"Projected-potential thickness: {interaction_thickness:.4f} A "
        f"({config.projection_samples} z samples)"
    )

    input_field, t_input = timed_call(lambda: evaluate_field(beam, grid, config))
    action_grid_stack, action_grid_times = build_action_grids(
        grid,
        beam,
        slice_xy_stack,
        config,
        interaction_thickness=interaction_thickness,
        smoothing_radius=smoothing_radius,
    )
    potential_grid_stack = action_to_potential_grid_stack(
        action_grid_stack,
        beam,
    )
    reference, reference_slice_times, reference_history = reference_multislice(
        input_field,
        grid,
        beam,
        action_grid_stack,
        config,
        propagation_distance=propagation_distance,
    )
    (
        propagated_beam,
        action_times,
        propagation_times,
        gaussian_history,
        gaussian_history_eval_times,
    ) = gaussian_multislice(
        beam,
        action_grid_stack,
        config,
        grid=grid,
        propagation_distance=propagation_distance,
    )
    gaussian_field, t_output = timed_call(lambda: evaluate_field(propagated_beam, grid, config))

    timings = {
        "input_eval": t_input,
        "action_grid_total": float(np.sum(action_grid_times)),
        "action_grid_per_slice": action_grid_times,
        "reference_total": float(np.sum(reference_slice_times)),
        "reference_per_slice": reference_slice_times,
        "gaussian_fit_total": float(np.sum(action_times)),
        "gaussian_fit_per_slice": action_times,
        "gaussian_propagate_total": float(np.sum(propagation_times)),
        "gaussian_propagate_per_slice": propagation_times,
        "gaussian_history_eval_total": float(np.sum(gaussian_history_eval_times)),
        "gaussian_history_eval_per_slice": gaussian_history_eval_times,
        "output_eval": t_output,
    }
    timings["gaussian_path_total"] = (
        timings["gaussian_fit_total"]
        + timings["gaussian_propagate_total"]
        + timings["output_eval"]
    )

    result = {
        "config": config,
        "interaction_thickness": interaction_thickness,
        "grid": grid,
        "extent": extent,
        "slice_xy_stack": slice_xy_stack,
        "action_grid_stack": action_grid_stack,
        "potential_grid_stack": potential_grid_stack,
        "input_beam": beam,
        "output_beam": propagated_beam,
        "input_field": input_field,
        "reference": reference,
        "gaussian": gaussian_field,
        "timings": timings,
        "field_error": aligned_field_error(gaussian_field, reference),
        "intensity_error": normalized_intensity_error(gaussian_field, reference),
    }
    if reference_history is not None and gaussian_history is not None:
        result["reference_history"] = jnp.stack(reference_history, axis=0)
        result["gaussian_history"] = jnp.stack(gaussian_history, axis=0)
        result["slice_field_errors"] = [
            aligned_field_error(gaussian_history[i], reference_history[i])
            for i in range(len(reference_history))
        ]
        result["slice_intensity_errors"] = [
            normalized_intensity_error(gaussian_history[i], reference_history[i])
            for i in range(len(reference_history))
        ]
    print(f"Field error: {result['field_error']:.6g}")
    print(f"Intensity error: {result['intensity_error']:.6g}")
    print(f"Action-grid build total: {timings['action_grid_total']:.3f}s")
    print(f"Reference total: {timings['reference_total']:.3f}s")
    print(f"Gaussian path total: {timings['gaussian_path_total']:.3f}s")
    return result


def save_outputs(result: dict[str, Any], output_dir: str | Path) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    import matplotlib.pyplot as plt

    reference = np.asarray(result["reference"])
    gaussian = np.asarray(result["gaussian"])
    intensity_diff = np.abs(gaussian) ** 2 - np.abs(reference) ** 2
    relative_phase = np.angle(gaussian * np.conj(reference))
    extent = float(result["extent"])
    plot_extent = (-extent, extent, -extent, extent)
    config = result["config"]

    fig, axes = plt.subplots(2, 3, figsize=(11, 6.5), sharex=True, sharey=True)
    images = [
        (np.abs(reference), "Fresnel amplitude", "inferno"),
        (np.abs(gaussian), "Gaussian amplitude", "inferno"),
        (intensity_diff, "Intensity difference", "coolwarm"),
        (np.angle(reference), "Fresnel phase", "twilight"),
        (np.angle(gaussian), "Gaussian phase", "twilight"),
        (relative_phase, "Relative phase", "twilight"),
    ]
    for ax, (data, title, cmap) in zip(axes.ravel(), images):
        im = ax.imshow(data, extent=plot_extent, origin="lower", cmap=cmap)
        ax.set_title(title)
        ax.set_xlabel("x (A)")
        ax.set_ylabel("y (A)")
        fig.colorbar(im, ax=ax, shrink=0.75)
    fig.suptitle(
        f"field error={result['field_error']:.3g}, "
        f"intensity error={result['intensity_error']:.3g}"
    )
    fig.tight_layout()
    fig.savefig(output_dir / "field_comparison.png", dpi=180)
    plt.close(fig)

    potential = np.asarray(result["potential_grid_stack"])
    n_slices = potential.shape[0]
    selected = np.linspace(0, n_slices - 1, min(n_slices, 10), dtype=int)
    reference_diffracted_power = None
    gaussian_diffracted_power = None

    fig, axes = plt.subplots(1, len(selected) + 1, figsize=(2.4 * (len(selected) + 1), 2.7))
    if len(selected) == 0:
        axes = np.asarray([axes])
    vmax = float(np.max(potential)) if potential.size else 1.0
    for ax, slice_index in zip(np.ravel(axes)[:-1], selected):
        im = ax.imshow(
            potential[slice_index],
            extent=plot_extent,
            origin="lower",
            cmap="magma",
            vmin=0.0,
            vmax=vmax,
        )
        ax.set_title(f"Slice {slice_index + 1}")
        ax.set_xlabel("x (A)")
        ax.set_ylabel("y (A)")
    im = np.ravel(axes)[-1].imshow(
        np.sum(potential, axis=0),
        extent=plot_extent,
        origin="lower",
        cmap="magma",
    )
    np.ravel(axes)[-1].set_title("Projected")
    np.ravel(axes)[-1].set_xlabel("x (A)")
    np.ravel(axes)[-1].set_ylabel("y (A)")
    fig.colorbar(
        im,
        ax=np.ravel(axes).tolist(),
        shrink=0.75,
        label="projected potential",
    )
    fig.suptitle("Projected atomic potential by slice")
    fig.savefig(output_dir / "atomic_potential_slices.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    reference_history = result.get("reference_history")
    gaussian_history = result.get("gaussian_history")
    if reference_history is not None and gaussian_history is not None:
        reference_history = np.asarray(reference_history)
        gaussian_history = np.asarray(gaussian_history)
        ref_intensity = np.abs(reference_history) ** 2
        gauss_intensity = np.abs(gaussian_history) ** 2
        intensity_history_diff = gauss_intensity - ref_intensity
        selected = np.linspace(0, reference_history.shape[0] - 1, min(reference_history.shape[0], 10), dtype=int)

        fig, axes = plt.subplots(
            4,
            len(selected),
            figsize=(2.35 * len(selected), 8.2),
            sharex=True,
            sharey=True,
        )
        if len(selected) == 1:
            axes = axes[:, None]
        intensity_vmax = float(max(np.max(ref_intensity), np.max(gauss_intensity)))
        diff_vmax = float(np.max(np.abs(intensity_history_diff)))
        for col, slice_index in enumerate(selected):
            panels = [
                (potential[slice_index], "Projected potential", "magma", 0.0, vmax),
                (ref_intensity[slice_index], "Fresnel |psi|^2", "inferno", 0.0, intensity_vmax),
                (gauss_intensity[slice_index], "Gaussian |psi|^2", "inferno", 0.0, intensity_vmax),
                (
                    intensity_history_diff[slice_index],
                    "Intensity diff",
                    "coolwarm",
                    -diff_vmax,
                    diff_vmax,
                ),
            ]
            for row, (data, label, cmap, vmin, vmax_i) in enumerate(panels):
                im = axes[row, col].imshow(
                    data,
                    extent=plot_extent,
                    origin="lower",
                    cmap=cmap,
                    vmin=vmin,
                    vmax=vmax_i,
                )
                if row == 0:
                    err = result.get("slice_intensity_errors", [None] * reference_history.shape[0])[slice_index]
                    axes[row, col].set_title(f"Slice {slice_index + 1}\nI err={err:.2e}")
                if col == 0:
                    axes[row, col].set_ylabel(label)
                axes[row, col].set_xlabel("x (A)")
        fig.suptitle("Slice-by-slice projected potential and intensity evolution")
        fig.tight_layout()
        fig.savefig(output_dir / "slice_intensity_history.png", dpi=180)
        plt.close(fig)

        def symmetric_limit(values, percentile=99.0):
            limit = float(np.percentile(np.abs(values), percentile))
            return max(limit, 1e-12)

        def diffraction_log_intensity(fields):
            diffraction = np.fft.fftshift(
                np.fft.fft2(fields, axes=(-2, -1)),
                axes=(-2, -1),
            )
            intensity = np.abs(diffraction) ** 2
            scale = np.max(intensity, axis=(-2, -1), keepdims=True)
            scale = np.maximum(scale, 1e-300)
            return np.log10(intensity / scale + 1e-12)

        def noncentral_diffraction_fraction(fields, center_radius_pixels=1):
            diffraction = np.fft.fftshift(
                np.fft.fft2(fields, axes=(-2, -1)),
                axes=(-2, -1),
            )
            intensity = np.abs(diffraction) ** 2
            cy = intensity.shape[-2] // 2
            cx = intensity.shape[-1] // 2
            central = intensity[
                ...,
                cy - center_radius_pixels : cy + center_radius_pixels + 1,
                cx - center_radius_pixels : cx + center_radius_pixels + 1,
            ]
            total_power = np.sum(intensity, axis=(-2, -1))
            central_power = np.sum(central, axis=(-2, -1))
            return 1.0 - central_power / np.maximum(total_power, 1e-300)

        wavelength = float(result["input_beam"].to_vector().wavelength[0])
        pixel = 2.0 * extent / reference_history.shape[-1]
        freq = np.fft.fftshift(np.fft.fftfreq(reference_history.shape[-1], d=pixel))
        angle_mrad = wavelength * freq * 1e3
        angle_limit = min(float(np.max(np.abs(angle_mrad))), 30.0)
        angle_mask = np.where(np.abs(angle_mrad) <= angle_limit)[0]
        k0, k1 = int(angle_mask[0]), int(angle_mask[-1] + 1)
        diffraction_extent = (
            angle_mrad[k0],
            angle_mrad[k1 - 1],
            angle_mrad[k0],
            angle_mrad[k1 - 1],
        )

        ref_diffraction = diffraction_log_intensity(reference_history)[:, k0:k1, k0:k1]
        gauss_diffraction = diffraction_log_intensity(gaussian_history)[:, k0:k1, k0:k1]
        reference_diffracted_power = noncentral_diffraction_fraction(reference_history)
        gaussian_diffracted_power = noncentral_diffraction_fraction(gaussian_history)
        diffraction_diff = gauss_diffraction - ref_diffraction
        diffraction_vmin = -8.0
        diffraction_vmax = 0.0
        diffraction_diff_vmax = symmetric_limit(diffraction_diff, 99.0)

        fig, axes = plt.subplots(
            3,
            len(selected),
            figsize=(2.35 * len(selected), 6.4),
            sharex=True,
            sharey=True,
        )
        if len(selected) == 1:
            axes = axes[:, None]
        for col, slice_index in enumerate(selected):
            panels = [
                (
                    ref_diffraction[slice_index],
                    "Fresnel log diffraction",
                    "inferno",
                    diffraction_vmin,
                    diffraction_vmax,
                ),
                (
                    gauss_diffraction[slice_index],
                    "Gaussian log diffraction",
                    "inferno",
                    diffraction_vmin,
                    diffraction_vmax,
                ),
                (
                    diffraction_diff[slice_index],
                    "Log diffraction diff",
                    "coolwarm",
                    -diffraction_diff_vmax,
                    diffraction_diff_vmax,
                ),
            ]
            for row, (data, label, cmap, vmin, vmax_i) in enumerate(panels):
                axes[row, col].imshow(
                    data,
                    extent=diffraction_extent,
                    origin="lower",
                    cmap=cmap,
                    vmin=vmin,
                    vmax=vmax_i,
                )
                if row == 0:
                    axes[row, col].set_title(f"Slice {slice_index + 1}")
                if col == 0:
                    axes[row, col].set_ylabel(label)
                axes[row, col].set_xlabel("angle x/y (mrad)")
        fig.suptitle("Slice-by-slice diffraction pattern evolution")
        fig.tight_layout()
        fig.savefig(output_dir / "slice_diffraction_history.png", dpi=180)
        plt.close(fig)

        relative_phase_history = np.angle(gaussian_history * np.conj(reference_history))
        fig, axes = plt.subplots(
            3,
            len(selected),
            figsize=(2.35 * len(selected), 6.4),
            sharex=True,
            sharey=True,
        )
        if len(selected) == 1:
            axes = axes[:, None]
        for col, slice_index in enumerate(selected):
            panels = [
                (np.angle(reference_history[slice_index]), "Fresnel phase", "twilight"),
                (np.angle(gaussian_history[slice_index]), "Gaussian phase", "twilight"),
                (relative_phase_history[slice_index], "Relative phase", "twilight"),
            ]
            for row, (data, label, cmap) in enumerate(panels):
                axes[row, col].imshow(data, extent=plot_extent, origin="lower", cmap=cmap)
                if row == 0:
                    axes[row, col].set_title(f"Slice {slice_index + 1}")
                if col == 0:
                    axes[row, col].set_ylabel(label)
                axes[row, col].set_xlabel("x (A)")
        fig.suptitle("Slice-by-slice phase evolution")
        fig.tight_layout()
        fig.savefig(output_dir / "slice_phase_history.png", dpi=180)
        plt.close(fig)

        height, width = potential.shape[-2:]
        x_coords = np.linspace(-extent, extent, width)
        y_coords = np.linspace(-extent, extent, height)
        crop_half_width = min(
            extent,
            0.5 * config.n_cells_xy * config.lattice_constant + 1.0,
        )
        x_crop = np.where(np.abs(x_coords) <= crop_half_width)[0]
        y_crop = np.where(np.abs(y_coords) <= crop_half_width)[0]
        x0, x1 = int(x_crop[0]), int(x_crop[-1] + 1)
        y0, y1 = int(y_crop[0]), int(y_crop[-1] + 1)
        crop_extent = (
            x_coords[x0],
            x_coords[x1 - 1],
            y_coords[y0],
            y_coords[y1 - 1],
        )

        def intensity_modulation(intensity):
            mean = np.mean(intensity, axis=(-2, -1), keepdims=True)
            return intensity / mean - 1.0

        def phase_modulation(fields):
            mean_phase = np.angle(np.mean(fields, axis=(-2, -1), keepdims=True))
            return np.angle(fields * np.exp(-1j * mean_phase))

        potential_crop = potential[:, y0:y1, x0:x1]
        ref_intensity_crop = ref_intensity[:, y0:y1, x0:x1]
        gauss_intensity_crop = gauss_intensity[:, y0:y1, x0:x1]
        reference_crop = reference_history[:, y0:y1, x0:x1]
        gaussian_crop = gaussian_history[:, y0:y1, x0:x1]

        ref_i_mod = intensity_modulation(ref_intensity_crop)
        gauss_i_mod = intensity_modulation(gauss_intensity_crop)
        i_mod_diff = gauss_i_mod - ref_i_mod
        ref_phase_mod = phase_modulation(reference_crop)
        gauss_phase_mod = phase_modulation(gaussian_crop)
        phase_mod_diff = np.angle(np.exp(1j * (gauss_phase_mod - ref_phase_mod)))

        i_mod_vmax = symmetric_limit(np.concatenate((ref_i_mod.ravel(), gauss_i_mod.ravel())), 98.0)
        i_diff_vmax = symmetric_limit(i_mod_diff, 98.0)
        phase_vmax = symmetric_limit(np.concatenate((ref_phase_mod.ravel(), gauss_phase_mod.ravel())), 98.0)
        phase_diff_vmax = symmetric_limit(phase_mod_diff, 98.0)
        potential_crop_vmax = float(np.max(potential_crop))

        fig, axes = plt.subplots(
            7,
            len(selected),
            figsize=(2.35 * len(selected), 13.2),
            sharex=True,
            sharey=True,
        )
        if len(selected) == 1:
            axes = axes[:, None]
        for col, slice_index in enumerate(selected):
            panels = [
                (
                    potential_crop[slice_index],
                    "Projected potential",
                    "magma",
                    0.0,
                    potential_crop_vmax,
                ),
                (ref_i_mod[slice_index], "Fresnel I mod", "coolwarm", -i_mod_vmax, i_mod_vmax),
                (gauss_i_mod[slice_index], "Gaussian I mod", "coolwarm", -i_mod_vmax, i_mod_vmax),
                (i_mod_diff[slice_index], "I mod diff", "coolwarm", -i_diff_vmax, i_diff_vmax),
                (ref_phase_mod[slice_index], "Fresnel phase mod", "coolwarm", -phase_vmax, phase_vmax),
                (gauss_phase_mod[slice_index], "Gaussian phase mod", "coolwarm", -phase_vmax, phase_vmax),
                (phase_mod_diff[slice_index], "Phase mod diff", "coolwarm", -phase_diff_vmax, phase_diff_vmax),
            ]
            for row, (data, label, cmap, vmin, vmax_i) in enumerate(panels):
                axes[row, col].imshow(
                    data,
                    extent=crop_extent,
                    origin="lower",
                    cmap=cmap,
                    vmin=vmin,
                    vmax=vmax_i,
                )
                if row == 0:
                    axes[row, col].set_title(f"Slice {slice_index + 1}")
                if col == 0:
                    axes[row, col].set_ylabel(label)
                axes[row, col].set_xlabel("x (A)")
        fig.suptitle("Cropped atom-induced intensity and phase modulation")
        fig.tight_layout()
        fig.savefig(output_dir / "slice_modulation_history.png", dpi=180)
        plt.close(fig)

    timings = result["timings"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
    axes[0].bar(
        ["Reference", "Gaussian"],
        [timings["reference_total"], timings["gaussian_path_total"]],
        color=["#4C78A8", "#F58518"],
    )
    axes[0].set_ylabel("seconds")
    axes[0].set_title("End-to-end path timing")

    x = np.arange(config.n_slices) + 1
    axes[1].plot(x, timings["action_grid_per_slice"], marker="o", label="Build action grid")
    axes[1].plot(x, timings["reference_per_slice"], marker="o", label="Fresnel slice")
    axes[1].plot(x, timings["gaussian_fit_per_slice"], marker="o", label="Gaussian fit")
    axes[1].plot(
        x,
        timings["gaussian_propagate_per_slice"],
        marker="o",
        label="Gaussian propagate",
    )
    axes[1].set_xlabel("slice")
    axes[1].set_ylabel("seconds")
    axes[1].set_title("Per-slice timing")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "timing_breakdown.png", dpi=180)
    plt.close(fig)

    np.savez_compressed(
        output_dir / "fields.npz",
        reference=reference,
        gaussian=gaussian,
        intensity_diff=intensity_diff,
        relative_phase=relative_phase,
        potential_grid_stack=potential,
        reference_history=np.asarray(reference_history)
        if reference_history is not None
        else np.asarray([]),
        gaussian_history=np.asarray(gaussian_history)
        if gaussian_history is not None
        else np.asarray([]),
        reference_diffracted_power=reference_diffracted_power
        if reference_diffracted_power is not None
        else np.asarray([]),
        gaussian_diffracted_power=gaussian_diffracted_power
        if gaussian_diffracted_power is not None
        else np.asarray([]),
    )
    with (output_dir / "summary.txt").open("w") as f:
        f.write(f"JAX backend: {jax.default_backend()}\n")
        f.write(
            f"projected_potential_thickness: "
            f"{float(result['interaction_thickness']):.12g}\n"
        )
        f.write(f"projection_samples: {int(config.projection_samples)}\n")
        f.write(f"projected_potential_max: {float(np.max(potential)):.12g}\n")
        f.write(f"field_error: {result['field_error']:.12g}\n")
        f.write(f"intensity_error: {result['intensity_error']:.12g}\n")
        if "slice_field_errors" in result:
            f.write(
                "slice_field_errors: "
                + ", ".join(f"{float(v):.6g}" for v in result["slice_field_errors"])
                + "\n"
            )
            f.write(
                "slice_intensity_errors: "
                + ", ".join(f"{float(v):.6g}" for v in result["slice_intensity_errors"])
                + "\n"
            )
        if reference_diffracted_power is not None and gaussian_diffracted_power is not None:
            f.write(
                "reference_diffracted_power: "
                + ", ".join(f"{float(v):.6g}" for v in reference_diffracted_power)
                + "\n"
            )
            f.write(
                "gaussian_diffracted_power: "
                + ", ".join(f"{float(v):.6g}" for v in gaussian_diffracted_power)
                + "\n"
            )
        for key, value in timings.items():
            if isinstance(value, list):
                values = ", ".join(f"{float(item):.6g}" for item in value)
                f.write(f"{key}: {values}\n")
            else:
                f.write(f"{key}: {float(value):.12g}\n")
    print(f"Saved plots and fields to {output_dir}")


def quick_config() -> MultisliceBenchmarkConfig:
    return MultisliceBenchmarkConfig(
        n_cells_xy=3,
        n_slices=2,
        grid_shape=64,
        target_beams_side=24,
        beam_chunk_size=288,
        reference_jit=False,
    )


def full_config() -> MultisliceBenchmarkConfig:
    return MultisliceBenchmarkConfig(
        n_cells_xy=9,
        n_slices=10,
        grid_shape=192,
        target_beams_side=224,
        beam_chunk_size=1792,
        fit_samples_per_axis=3,
        fit_support_radius=1.0,
        reference_jit=os.environ.get("TEMGYM_REFERENCE_JIT", "1") != "0",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full", action="store_true", help="run the 50,176-beam / 10-slice case")
    parser.add_argument("--quick", action="store_true", help="run the small smoke-test case")
    parser.add_argument("--require-gpu", action="store_true", help="fail if JAX is not using CUDA")
    parser.add_argument(
        "--record-slices",
        action="store_true",
        help="evaluate and store Fresnel/Gaussian fields after every slice",
    )
    parser.add_argument(
        "--interaction-thickness",
        type=float,
        help=(
            "projected-potential thickness in angstrom; defaults to the "
            "propagation distance per slice"
        ),
    )
    parser.add_argument(
        "--projection-samples",
        type=int,
        help="number of Gauss-Legendre z samples used for slice projection",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="directory for comparison plots, timing plots, and output fields",
    )
    args = parser.parse_args()
    if args.require_gpu and jax.default_backend() != "gpu":
        raise RuntimeError(f"Expected JAX GPU backend, got {jax.default_backend()!r}.")

    config = full_config() if args.full else quick_config()
    if args.record_slices:
        config = replace(config, record_slices=True)
    if args.interaction_thickness is not None:
        config = replace(config, interaction_thickness=args.interaction_thickness)
    if args.projection_samples is not None:
        config = replace(config, projection_samples=args.projection_samples)
    result = run_benchmark(config)
    if args.output_dir is not None:
        save_outputs(result, args.output_dir)
    print("benchmark complete", result["field_error"], result["intensity_error"])
