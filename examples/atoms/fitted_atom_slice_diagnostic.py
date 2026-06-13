"""Small CPU diagnostic for fitted Gaussian atom-slice propagation.

The reference is a one-slice multislice-style grid calculation:

    input field -> exact transmission exp(i sigma V dz) -> Fresnel propagation

The Gaussian path applies a sampled fitted quadratic action to each Gaussian
beam, propagates the resulting bundle, and compares the detector field.
"""

from __future__ import annotations

import os
import time

os.environ.setdefault("JAX_ENABLE_X64", "1")
os.environ["JAX_PLATFORM_NAME"] = "cpu"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import jax
import jax.numpy as jnp
import numpy as np

from temgym_core.components import Detector
from temgym_core.evaluate import evaluate_gaussians_for
from temgym_core.gaussian import FreeSpacePropagator, GaussianBeam, make_gaussian
from temgym_core.gaussian_corrections import (
    apply_fitted_quadratic_action,
    concatenate_gaussian_beams,
)
from temgym_core.potential import Si, potential_smoothed_from_r2
from temgym_core.source import square_input_wave
from temgym_core.utils import FresnelPropagator

jax.config.update("jax_enable_x64", True)


def _multi_atom_action(xy, z, sigma, k, atom_xyz, dz, cutoff_radius):
    def one_atom(atom):
        r2 = (
            (xy[0] - atom[0]) ** 2
            + (xy[1] - atom[1]) ** 2
            + (z - atom[2]) ** 2
        )
        return potential_smoothed_from_r2(r2, Si, cutoff_radius)

    potential = jnp.sum(jax.vmap(one_atom)(atom_xyz))
    return -(sigma / k) * potential * dz


def _apply_fitted_slice_to_bundle(
    beam: GaussianBeam,
    atom_xyz,
    dz,
    cutoff_radius,
    *,
    fit_support_radius=1.0,
    fit_samples_per_axis=3,
):
    vector = beam.to_vector()
    out = []
    for i in range(int(np.asarray(vector.x).shape[0])):
        ray = vector[i]
        out.append(
            apply_fitted_quadratic_action(
                ray,
                lambda xy: _multi_atom_action(
                    xy, ray.z, ray.sigma, ray.k, atom_xyz, dz, cutoff_radius
                ),
                fit_support_radius=fit_support_radius,
                fit_samples_per_axis=fit_samples_per_axis,
            )
        )
    return concatenate_gaussian_beams(out)


def _reference_field(
    input_field, grid, wavelength, atom_xyz, sigma, k, dz, cutoff_radius, propagation
):
    action = jax.vmap(
        lambda xy: _multi_atom_action(xy, 0.0, sigma, k, atom_xyz, dz, cutoff_radius)
    )(grid.coords).reshape(grid.shape)
    transmitted = input_field * jnp.exp(1j * k * action)
    return FresnelPropagator(
        transmitted,
        L=grid.pixel_size[0] * grid.shape[0],
        wavelength=wavelength,
        z=propagation,
    )


def _aligned_field_error(test, ref):
    alpha = jnp.vdot(test, ref) / jnp.vdot(test, test)
    return float(jnp.linalg.norm(alpha * test - ref) / jnp.linalg.norm(ref))


def _case(name, beam, grid, atom_xyz, dz, cutoff_radius, propagation):
    t0 = time.time()
    wavelength = beam.to_vector().wavelength[0]
    sigma = beam.to_vector().sigma[0]
    k = beam.to_vector().k[0]

    input_field = evaluate_gaussians_for(beam, grid)
    ref = _reference_field(
        input_field, grid, wavelength, atom_xyz, sigma, k, dz, cutoff_radius, propagation
    )

    fitted = _apply_fitted_slice_to_bundle(beam, atom_xyz, dz, cutoff_radius)
    propagated = FreeSpacePropagator()(fitted, propagation)
    gaussian_field = evaluate_gaussians_for(propagated, grid)

    print(
        f"{name}: beams={np.asarray(beam.to_vector().x).size}, "
        f"field_error={_aligned_field_error(gaussian_field, ref):.4g}, "
        f"time={time.time() - t0:.1f}s"
    )


def main():
    voltage = 100e3
    extent = 4.0  # angstrom full field width is 2*extent
    n_px = 48
    dz = 0.05  # angstrom
    propagation = 2.0  # angstrom
    pixel = 2 * extent / n_px
    cutoff_radius = pixel / 3.0

    grid = Detector(z=0.0, pixel_size=(pixel, pixel), shape=(n_px, n_px))

    # A small square plane of atoms.  Si is used because the repository already
    # contains Si potential parameters.
    spacing = 2.35
    atom_xy = jnp.array(
        [
            [-0.5 * spacing, -0.5 * spacing, 0.0],
            [0.5 * spacing, -0.5 * spacing, 0.0],
            [-0.5 * spacing, 0.5 * spacing, 0.0],
            [0.5 * spacing, 0.5 * spacing, 0.0],
        ]
    )

    tem = square_input_wave(
        aperture_length=2 * extent,
        waist=2.0,
        voltage=voltage,
        overlap_factor=1.0,
        wavelength_unit="angstrom",
    )
    stem = make_gaussian(
        x=0.0,
        y=0.0,
        z=0.0,
        voltage=voltage,
        waist_x=0.5,
        waist_y=0.5,
        wavelength_unit="angstrom",
    )

    _case("TEM-like broad beam", tem, grid, atom_xy, dz, cutoff_radius, propagation)
    _case("STEM-like focused probe", stem, grid, atom_xy, dz, cutoff_radius, propagation)


if __name__ == "__main__":
    main()
