import numpy as np
import pytest
import jax
import jax.numpy as jnp

from temgym_core.components import (
    ABCDTransfer,
    Detector,
    ElectromagneticLens,
    Lens,
    LinearPhaseShift,
    Rotator,
)
from temgym_core.constants import energy2wavelength
from temgym_core.evaluate import evaluate_gaussians_for, evaluate_gaussians_jax_scan
from temgym_core.gaussian import FreeSpacePropagator, make_gaussian


jax.config.update("jax_enable_x64", True)


def _batched_beam():
    return make_gaussian(
        x=jnp.array([-2.0e-6, 0.0, 1.5e-6]),
        y=jnp.array([1.0e-6, -1.0e-6, 0.5e-6]),
        dx=jnp.array([0.0, 1.0e-4, -2.0e-4]),
        dy=jnp.array([2.0e-4, 0.0, -1.0e-4]),
        z=0.0,
        voltage=jnp.array([100e3, 200e3, 300e3]),
        amp=jnp.array([1.0, 0.5, 0.25]),
        phase=jnp.array([0.0, 0.2, -0.1]),
        waist_x=jnp.array([1.0e-6, 1.5e-6, 2.0e-6]),
        waist_y=1.25e-6,
        rcurv_x=jnp.inf,
        rcurv_y=jnp.array([jnp.inf, 0.2, -0.15]),
    )


def _assert_component_matches_per_beam(component, beam):
    out_batch = component(beam)
    out_items = [component(beam[i]) for i in range(np.asarray(beam.x).shape[0])]

    for field in ("x", "y", "dx", "dy", "z", "pathlength", "amplitude", "Q_inv"):
        actual = np.asarray(getattr(out_batch, field))
        expected = np.stack(
            [np.asarray(getattr(out_item, field)) for out_item in out_items],
            axis=0,
        )
        np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)


def test_make_gaussian_broadcasts_constructor_fields_from_common_length():
    voltages = jnp.array([100e3, 200e3])
    waist = 1.0e-6

    beam = make_gaussian(
        x=0.0,
        y=jnp.array([0.0, 1.0e-6]),
        voltage=voltages,
        waist_x=waist,
        waist_y=waist,
    )

    assert np.asarray(beam.x).shape == (2,)
    assert np.asarray(beam.y).shape == (2,)
    assert np.asarray(beam.z).shape == (2,)
    assert np.asarray(beam.voltage).shape == (2,)
    assert np.asarray(beam.Q_inv).shape == (2, 2, 2)

    expected_q_imag = energy2wavelength(voltages) / (jnp.pi * waist**2)
    np.testing.assert_allclose(
        np.asarray(jnp.imag(beam.Q_inv[:, 0, 0])),
        np.asarray(expected_q_imag),
        rtol=1e-12,
        atol=1e-12,
    )


def test_make_gaussian_rejects_incompatible_constructor_lengths():
    with pytest.raises(ValueError, match="share one leading length"):
        make_gaussian(x=jnp.array([0.0, 1.0]), y=jnp.array([0.0, 1.0, 2.0]))


def test_gaussian_to_vector_promotes_q_inv_with_leading_axis():
    beam = make_gaussian(x=0.0, y=0.0, waist_x=1e-6, waist_y=2e-6)
    vector = beam.to_vector()

    assert np.asarray(vector.x).shape == (1,)
    assert np.asarray(vector.z).shape == (1,)
    assert np.asarray(vector.amplitude).shape == (1,)
    assert np.asarray(vector.Q_inv).shape == (1, 2, 2)


@pytest.mark.parametrize(
    "component",
    [
        FreeSpacePropagator().with_distance(0.015),
        Lens(z=0.0, focal_length=0.25),
        LinearPhaseShift(z=0.0, linear_phase_shift=jnp.array([1.0e-4, -2.0e-4])),
        Rotator(z=0.0, angle=27.0),
        ABCDTransfer(A=jnp.eye(2), B=0.015 * jnp.eye(2)),
        ElectromagneticLens(z=0.0, turns=10.0, current=2.0, Gc=1.0e-3),
    ],
)
def test_gaussian_components_match_per_beam_application(component):
    _assert_component_matches_per_beam(component, _batched_beam())


def test_gaussian_evaluators_agree_for_batched_beam():
    beam = _batched_beam()
    grid = Detector(z=0.0, pixel_size=(1e-6, 1e-6), shape=(5, 4))

    field_loop = evaluate_gaussians_for(beam, grid)
    field_scan = evaluate_gaussians_jax_scan(beam, grid, batch_size=2)

    np.testing.assert_allclose(
        np.asarray(field_scan),
        np.asarray(field_loop),
        rtol=1e-12,
        atol=1e-12,
    )
