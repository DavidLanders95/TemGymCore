import dataclasses
from typing import Any, Callable, NamedTuple, Sequence, Tuple

import jax
import jax.numpy as jnp
import jax_dataclasses as jdc
from ase import units
from jax import lax

from .constants import energy2wavelength, relativistic_mass_correction
from .ray import Ray

LENGTH = {
    "m": 1.0,
    "A": 1e-10,
    "angstrom": 1e-10,
}


def _as_constructor_vector(name: str, value: Any) -> jnp.ndarray:
    arr = jnp.asarray(value)
    if arr.ndim == 0:
        return arr[None]
    if arr.ndim == 1:
        return arr
    raise ValueError(
        f"`{name}` must be scalar or one-dimensional, got shape {arr.shape}."
    )


def _broadcast_constructor_fields(**fields: Any) -> dict[str, jnp.ndarray]:
    arrays = {
        name: _as_constructor_vector(name, value)
        for name, value in fields.items()
    }
    try:
        shape = jnp.broadcast_shapes(*(arr.shape for arr in arrays.values()))
    except ValueError as exc:
        sizes = ", ".join(f"{name}={arr.shape}" for name, arr in arrays.items())
        raise ValueError(
            "Gaussian constructor fields must be scalar or share one leading "
            f"length; got {sizes}."
        ) from exc

    return {
        name: jnp.broadcast_to(arr, shape)
        for name, arr in arrays.items()
    }


def _validate_positive(name: str, value: jnp.ndarray) -> None:
    try:
        invalid = bool(jnp.any(value <= 0))
    except Exception:
        return
    if invalid:
        raise ValueError(f"`{name}` must be positive.")


def _field_leading_size(name: str, value: Any) -> int:
    if isinstance(value, str) or value is None:
        return 1

    arr = jnp.asarray(value)
    if name == "Q_inv":
        if arr.shape == (2, 2):
            return 1
        if arr.ndim >= 3 and arr.shape[-2:] == (2, 2):
            return int(arr.shape[0])
        raise ValueError(
            f"`Q_inv` must have shape (2, 2) or (N, 2, 2), got {arr.shape}."
        )

    if arr.ndim == 0:
        return 1
    return int(arr.shape[0])


def _common_leading_size(params: dict[str, Any]) -> int:
    sizes = {
        _field_leading_size(name, value)
        for name, value in params.items()
        if not isinstance(value, str) and value is not None
    }
    non_scalar_sizes = {size for size in sizes if size != 1}
    if len(non_scalar_sizes) > 1:
        details = ", ".join(
            f"{name}={_field_leading_size(name, value)}"
            for name, value in params.items()
            if not isinstance(value, str) and value is not None
        )
        raise ValueError(f"GaussianBeam fields have incompatible leading sizes: {details}.")
    return next(iter(non_scalar_sizes), 1)


def _promote_gaussian_field(name: str, value: Any, size: int) -> Any:
    if isinstance(value, str) or value is None:
        return value

    arr = jnp.asarray(value)
    if name == "Q_inv":
        if arr.shape == (2, 2):
            arr = arr[None, ...]
        elif arr.ndim == 3 and arr.shape[-2:] == (2, 2):
            pass
        else:
            raise ValueError(
                f"`Q_inv` must have shape (2, 2) or (N, 2, 2), got {arr.shape}."
            )
    elif arr.ndim == 0:
        arr = arr[None]

    if arr.shape[0] == size:
        return arr
    if arr.shape[0] == 1:
        return jnp.broadcast_to(arr, (size,) + arr.shape[1:])
    raise ValueError(
        f"`{name}` has leading size {arr.shape[0]}, expected {size} or 1."
    )


def _matmul(left: jnp.ndarray, right: jnp.ndarray) -> jnp.ndarray:
    return jnp.einsum("...ij,...jk->...ik", left, right)


@jdc.pytree_dataclass(kw_only=True)
class GaussianBeam(Ray):
    amplitude: jnp.ndarray | complex
    Q_inv: jnp.ndarray | complex
    voltage: jnp.ndarray | float | None = None
    wavelength_unit: jdc.Static[str] = "m"

    @property
    def ray_family(self) -> str:
        return "gaussian"

    def derive(
        self,
        x: float | jnp.ndarray | None = None,
        y: float | jnp.ndarray | None = None,
        dx: float | jnp.ndarray | None = None,
        dy: float | jnp.ndarray | None = None,
        z: float | jnp.ndarray | None = None,
        amplitude: jnp.ndarray | complex | None = None,
        pathlength: float | jnp.ndarray | None = None,
        Q_inv: jnp.ndarray | complex | None = None,
        voltage: float | jnp.ndarray | None = None,
        _one: float | jnp.ndarray | None = None,
        wavelength_unit: str | None = None,
    ) -> "GaussianBeam":
        return GaussianBeam(
            x=self.x if x is None else x,
            y=self.y if y is None else y,
            dx=self.dx if dx is None else dx,
            dy=self.dy if dy is None else dy,
            z=self.z if z is None else z,
            amplitude=self.amplitude if amplitude is None else amplitude,
            pathlength=self.pathlength if pathlength is None else pathlength,
            Q_inv=self.Q_inv if Q_inv is None else Q_inv,
            voltage=self.voltage if voltage is None else voltage,
            _one=self._one if _one is None else _one,
            wavelength_unit=self.wavelength_unit if wavelength_unit is None else wavelength_unit,
        )

    def to_vector(self) -> jnp.ndarray:
        params = dataclasses.asdict(self)
        size = _common_leading_size(params)
        params = {
            k: _promote_gaussian_field(k, v, size)
            for k, v in params.items()
        }
        return type(self)(**params)

    @property
    def wavelength(self):
        return energy2wavelength(self.voltage) / LENGTH[self.wavelength_unit]

    @property
    def mass(self):
        return relativistic_mass_correction(self.voltage) * units._me

    @property
    def sigma(self):
        lam = self.wavelength
        return (
            2 * jnp.pi * self.mass * units.kg * units._e * units.C * lam
            / (units._hplanck * units.s * units.J) ** 2
        )

    @property
    def k(self):
        return 2 * jnp.pi / self.wavelength


def make_gaussian(
    x=0.0,
    y=0.0,
    dx=0.0,
    dy=0.0,
    z=0.0,
    voltage: float | jnp.ndarray = 1e5,
    amp=1.0,
    phase=0.0,
    waist_x=1.0,
    waist_y=1.0,
    rcurv_x=jnp.inf,
    rcurv_y=jnp.inf,
    wavelength_unit: str = "m",
) -> GaussianBeam:
    if wavelength_unit not in LENGTH:
        raise ValueError(
            f"Unknown wavelength unit {wavelength_unit!r}; expected one of "
            f"{tuple(LENGTH)}."
        )

    fields = _broadcast_constructor_fields(
        x=x,
        y=y,
        dx=dx,
        dy=dy,
        z=z,
        voltage=voltage,
        amp=amp,
        phase=phase,
        waist_x=waist_x,
        waist_y=waist_y,
        rcurv_x=rcurv_x,
        rcurv_y=rcurv_y,
    )
    x = fields["x"]
    y = fields["y"]
    dx = fields["dx"]
    dy = fields["dy"]
    z = fields["z"]
    voltage = fields["voltage"]
    amp = fields["amp"]
    phase = fields["phase"]
    waist_x = fields["waist_x"]
    waist_y = fields["waist_y"]
    rcurv_x = fields["rcurv_x"]
    rcurv_y = fields["rcurv_y"]

    _validate_positive("voltage", voltage)
    _validate_positive("waist_x", waist_x)
    _validate_positive("waist_y", waist_y)

    wavelength = energy2wavelength(voltage) / LENGTH[wavelength_unit]
    n_rays = x.shape[0]

    curv_x = 1.0 / rcurv_x
    curv_y = 1.0 / rcurv_y

    Q_inv_re = jnp.zeros((n_rays, 2, 2), dtype=jnp.float64)
    Q_inv_re = Q_inv_re.at[:, 0, 0].set(curv_x)
    Q_inv_re = Q_inv_re.at[:, 1, 1].set(curv_y)

    Q_inv_im = jnp.zeros((n_rays, 2, 2), dtype=jnp.float64)
    Q_inv_im = Q_inv_im.at[:, 0, 0].set(wavelength / (jnp.pi * waist_x**2))
    Q_inv_im = Q_inv_im.at[:, 1, 1].set(wavelength / (jnp.pi * waist_y**2))

    Q_inv = (Q_inv_re + 1j * Q_inv_im).astype(jnp.complex128)

    amplitude = jnp.asarray(amp) * jnp.exp(1j * phase)
    pathlength = jnp.zeros_like(phase)

    ray = GaussianBeam(
        x=x,
        y=y,
        dx=dx,
        dy=dy,
        z=z,
        amplitude=amplitude,
        Q_inv=Q_inv,
        voltage=voltage,
        pathlength=pathlength,
        _one=jnp.ones_like(x),
        wavelength_unit=wavelength_unit,
    ).to_vector()

    if n_rays == 1:
        def squeeze0(a):
            if a is None:
                return None
            a = jnp.asarray(a)
            return jnp.squeeze(a, axis=0) if (a.ndim > 0 and a.shape[0] == 1) else a

        ray = jax.tree.map(squeeze0, ray)

    return ray


def scalar_grad_hess_complex(
    fn: Callable[..., complex],
    x: jnp.ndarray,
    *args: Any,
    diff_argnums: int | Sequence[int] = 0,
) -> Tuple[complex, jnp.ndarray, jnp.ndarray]:
    dS0, grad, hess = taylor_expand(fn, x, *args, diff_argnums=diff_argnums)
    return dS0, grad, hess


def taylor_expand(
    fn: Callable[..., complex],
    x: jnp.ndarray,
    *args: Any,
    diff_argnums: int | Sequence[int] = 0,
) -> Tuple[complex, jnp.ndarray, jnp.ndarray]:
    full_args = (x, *args)

    def re_fn(*fn_args):
        return jnp.real(fn(*fn_args))

    def im_fn(*fn_args):
        return jnp.imag(fn(*fn_args))

    dS0 = fn(*full_args)

    grad_re = jax.grad(re_fn, argnums=diff_argnums)(*full_args)
    grad_im = jax.grad(im_fn, argnums=diff_argnums)(*full_args)

    hess_re = jax.hessian(re_fn, argnums=diff_argnums)(*full_args)
    hess_im = jax.hessian(im_fn, argnums=diff_argnums)(*full_args)

    grad = grad_re + 1j * grad_im
    hess = hess_re + 1j * hess_im
    return dS0, grad, hess


def apply_action_delta(
    ray: GaussianBeam,
    dS0: complex,
    dS1: jnp.ndarray,
    dS2: jnp.ndarray,
    tiny: float = 1e-30,
):
    if jnp.asarray(ray.r_xy).ndim > 1:
        ray = ray.to_vector()
        n = jnp.asarray(ray.x).shape[0]

        def in_axes(value, trailing_ndim: int):
            arr = jnp.asarray(value)
            if trailing_ndim == 0:
                return 0 if arr.ndim == 1 and arr.shape[0] == n else None
            return (
                0
                if arr.ndim == trailing_ndim + 1 and arr.shape[0] == n
                else None
            )

        return jax.vmap(
            lambda ray_i, s0, s1, s2: _apply_action_delta_single(
                ray_i, s0, s1, s2, tiny=tiny
            ),
            in_axes=(
                0,
                in_axes(dS0, 0),
                in_axes(dS1, 1),
                in_axes(dS2, 2),
            ),
        )(ray, dS0, dS1, dS2)

    return _apply_action_delta_single(ray, dS0, dS1, dS2, tiny=tiny)


def _apply_action_delta_single(
    ray: GaussianBeam,
    dS0: complex,
    dS1: jnp.ndarray,
    dS2: jnp.ndarray,
    tiny: float = 1e-30,
):
    k = ray.k
    r0 = ray.r_xy

    S0_old = ray.pathlength
    d_xy_old = ray.d_xy
    Q_old = ray.Q_inv

    S0_prime = S0_old + dS0
    S1_prime = d_xy_old + dS1
    Q_prime = Q_old + dS2

    ImQ = jnp.imag(Q_prime)
    ImS1 = jnp.imag(S1_prime)

    def solve_dx(args):
        ImQ_, ImS1_ = args
        return jnp.linalg.solve(ImQ_, -ImS1_)

    def zero_dx(args):
        _, ImS1_ = args
        return jnp.zeros_like(ImS1_)

    det_ImQ = jnp.linalg.det(ImQ)
    dx = lax.cond(
        jnp.abs(det_ImQ) < tiny,
        zero_dx,
        solve_dx,
        (ImQ, ImS1),
    )

    r_xy_new = r0 + dx
    S0_new = S0_prime + jnp.dot(S1_prime, dx) + 0.5 * jnp.dot(
        dx, Q_prime @ dx
    )
    S1_new = S1_prime + Q_prime @ dx
    Q_new = Q_prime

    d_xy_new = jnp.real(S1_new)
    S0_new_re = jnp.real(S0_new)
    S0_new_im = jnp.imag(S0_new)

    pathlength_new = S0_new_re
    amp_factor = jnp.exp(-k * S0_new_im)
    amplitude_new = ray.amplitude * amp_factor

    return r_xy_new, d_xy_new, amplitude_new, pathlength_new, Q_new


class Propagator(NamedTuple):
    distance: float
    propagator: "BaseGaussianPropagator"

    def __call__(self, ray: GaussianBeam) -> GaussianBeam:
        return self.propagator(ray, self.distance)


class BaseGaussianPropagator:
    def __call__(self, ray: GaussianBeam, distance: float) -> GaussianBeam:
        raise NotImplementedError

    def with_distance(self, distance: float) -> Propagator:
        return Propagator(distance, self)


class FreeSpacePropagator(BaseGaussianPropagator):
    def __call__(self, ray: GaussianBeam, distance: float) -> GaussianBeam:
        theta = ray.d_xy
        Q = ray.Q_inv

        identity = jnp.eye(2, dtype=jnp.complex128)
        A = identity + distance * Q
        invA = jnp.linalg.solve(A, jnp.broadcast_to(identity, A.shape))
        detA = jnp.linalg.det(A)

        Q_new = _matmul(Q, invA)
        r_xy_new = ray.r_xy + distance * theta

        theta_sq = jnp.sum(theta * theta, axis=-1)
        pathlength_new = ray.pathlength + distance + 0.5 * distance * theta_sq
        amplitude_new = ray.amplitude * detA**(-0.5)

        return ray.derive(
            x=r_xy_new[..., 0],
            y=r_xy_new[..., 1],
            dx=theta[..., 0],
            dy=theta[..., 1],
            z=ray.z + distance,
            amplitude=amplitude_new,
            pathlength=pathlength_new,
            Q_inv=Q_new,
        )


_REMOVED_SYMBOL_TO_PATH = {
    "Component": "temgym_core.components.Component",
    "Lens": "temgym_core.components.Lens",
    "KrivanekLens": "temgym_core.components.KrivanekLens",
    "SeidelLens": "temgym_core.components.SeidelLens",
    "DistortedLens": "temgym_core.components.DistortedLens",
    "ElectromagneticLens": "temgym_core.components.ElectromagneticLens",
    "ABCDTransfer": "temgym_core.components.ABCDTransfer",
    "SigmoidAperture": "temgym_core.components.SigmoidAperture",
    "Biprism": "temgym_core.components.PhaseBiprism",
    "PhaseBiprism": "temgym_core.components.PhaseBiprism",
    "DeflectionBiprism": "temgym_core.components.DeflectionBiprism",
    "ConstantPhaseShift": "temgym_core.components.ConstantPhaseShift",
    "LinearPhaseShift": "temgym_core.components.LinearPhaseShift",
    "QuadraticPhaseShift": "temgym_core.components.QuadraticPhaseShift",
    "ConstantAmplitudeShift": "temgym_core.components.ConstantAmplitudeShift",
    "LinearAmplitudeShift": "temgym_core.components.LinearAmplitudeShift",
    "QuadraticAmplitudeShift": "temgym_core.components.QuadraticAmplitudeShift",
    "MagneticPhaseSample": "temgym_core.components.MagneticPhaseSample",
    "RandomPhaseSample": "temgym_core.components.RandomPhaseSample",
    "InterpolatedSample2D": "temgym_core.components.InterpolatedSample2D",
    "InterpolatedFields3D": "temgym_core.components.InterpolatedFields3D",
    "AtomicPotential": "temgym_core.components.AtomicPotential",
    "FourierTransform": "temgym_core.components.FourierTransform",
    "Detector": "temgym_core.components.Detector",
    "run_iter": "temgym_core.run.run_iter",
    "run_to_end": "temgym_core.run.run_to_end",
    "run_iter_vmapped": "temgym_core.run.run_iter_vmapped",
    "run_to_end_vmapped": "temgym_core.run.run_to_end_vmapped",
    "circular_input_wave": "temgym_core.source.circular_input_wave",
    "square_input_wave": "temgym_core.source.square_input_wave",
    "rectangular_input_wave": "temgym_core.source.rectangular_input_wave",
    "sample_input_wave": "temgym_core.source.sample_input_wave",
}


def __getattr__(name: str):
    if name in _REMOVED_SYMBOL_TO_PATH:
        target = _REMOVED_SYMBOL_TO_PATH[name]
        raise AttributeError(
            f"`temgym_core.gaussian.{name}` has moved. Import `{target}` instead."
        )
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "GaussianBeam",
    "make_gaussian",
    "scalar_grad_hess_complex",
    "taylor_expand",
    "apply_action_delta",
    "Propagator",
    "BaseGaussianPropagator",
    "FreeSpacePropagator",
]
