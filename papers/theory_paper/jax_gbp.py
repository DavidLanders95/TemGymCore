import jax.numpy as jnp
from dataclasses import dataclass
from jax import tree_util


@dataclass
class Ray:
    matrix: jnp.ndarray  # Shape (5,) vector [x, dx, y, dy, 1]
    z: float
    amplitude: float
    pathlength: float
    wavelength: float
    w0x: float = 1.0
    Rx: float = 0.0
    w0y: float = 1.0
    Ry: float = 0.0


class RaysWrapper:
    def __init__(self, rays):
        self.rays = rays

    @property
    def x(self):
        return self.matrix[0]

    @property
    def y(self):
        return self.matrix[1]

    @property
    def dx(self):
        return self.matrix[2]

    @property
    def dy(self):
        return self.matrix[3]

    @property
    def qx_inv(self):
        if self.Rx == 0.:
            return - 1j * jnp.pi * self.w0x ** 2 / self.wavelength
        else:
            return 1 / self.Rx - 1j * jnp.pi * self.w0x ** 2 / self.wavelength

    @property
    def qy_inv(self):
        if self.Ry == 0.:
            return 1j * jnp.pi * self.w0y ** 2 / self.wavelength
        else:
            return 1 / self.Ry - 1j * jnp.pi * self.w0y ** 2 / self.wavelength

    @property
    def Q1_inv(self):
        return jnp.diag([self.qx_inv, self.qy_inv])


def point_source(ps_params, ray: Ray):
    centre_yx = ps_params['centre_yx']
    slope_yx = ps_params['slope_yx']
    z = ps_params['z']
    amplitude = ps_params['amplitude']
    wavelength = ps_params['wavelength']

    matrix = jnp.array([centre_yx[1], centre_yx[0], slope_yx[1], slope_yx[0], 1.])

    return Ray(
        z=z,
        matrix=matrix,
        amplitude=amplitude,
        pathlength=0.0,
        wavelength=wavelength
    )


def propagate(distance, ray: Ray):

    x = ray.x + ray.dx * distance
    dx = ray.dx
    y = ray.y + ray.dy * distance
    dy = ray.dy

    pathlength = ray.pathlength + distance * jnp.sqrt(1 + dx ** 2 + dy ** 2)
    new_matrix = jnp.array([x, y, dx, dy, 1.])

    return Ray(
        z=ray.z + distance,
        matrix=new_matrix,
        amplitude=ray.amplitude,
        pathlength=pathlength,
        wavelength=ray.wavelength
    )


def propagate_non_paraxial(distance, ray: Ray):
    L, M, N = convert_slope_to_direction_cosines(ray.dx, ray.dy)

    dx = (L / N)
    dy = (M / N)
    x = ray.x + dx * distance
    y = ray.y + dy * distance

    pathlength = ray.pathlength + distance * jnp.sqrt(1 + (dx / distance) ** 2 + (dy / distance) ** 2)
    new_matrix = jnp.array([x, y, ray.dx, ray.dy, 1.])

    return Ray(
        z=ray.z + distance,
        matrix=new_matrix,
        amplitude=ray.amplitude,
        pathlength=pathlength,
        wavelength=ray.wavelength,
        w0x=ray.w0x,
        Rx=ray.Rx,
        w0y=ray.w0y,
        Ry=ray.Ry
    )


def lens_step(lens_params, ray: Ray):

    f = lens_params['focal_length']
    x = ray.x
    dx = -x / f + ray.dx
    y = ray.y
    dy = -y / f + ray.dy

    pathlength = ray.pathlength - (ray.x ** 2 + ray.y ** 2) / (2 * jnp.float64(f))
    new_matrix = jnp.array([x, y, dx, dy, 1.])

    return Ray(
        z=ray.z,
        matrix=new_matrix,
        amplitude=ray.amplitude,
        pathlength=pathlength,
        wavelength=ray.wavelength
    )


@dataclass
class Detector:
    z: float
    shape: tuple
    px_size: float
    centre_yx: tuple

    def step(ray: Ray):
        return ray

    def xy_grid(self):
        x = jnp.linspace(-self.shape[0] / 2, self.shape[0] / 2, self.shape[0]) + centre_yx[1]
        y = jnp.linspace(-self.shape[1] / 2, self.shape[1] / 2, self.shape[1]) + centre_yx[0]
        return jnp.meshgrid(y, x)

    def get_image(self, ray: Ray):
        x, y = ray.x, ray.y
        x_idx = jnp.round((x - self.px_size / 2) / self.px_size).astype(jnp.int32)
        y_idx = jnp.round((y - self.px_size / 2) / self.px_size).astype(jnp.int32)

        image = jnp.zeros(self.shape, dtype=jnp.complex64)

        # Add the amplitude and phase of each ray to the pixel it lands on
        if (0 <= x_idx < self.shape[1]) and (0 <= y_idx < self.shape[0]):
            image[y_idx, x_idx] += ray.amplitude * jnp.exp(1j * ray.pathlength)
        return image


@dataclass
class Model:
    components: list

    def run_to_end(self, ray: Ray):
        for component in self.components:
            distance = component.z - ray.z
            ray = ray.propagate(distance)
            ray = component.step(ray)
        return ray

# Register the Ray dataclass with JAX
tree_util.register_pytree_node(
    Ray,
    lambda x: ((x.matrix, x.z, x.amplitude, x.pathlength, x.wavelength), None),
    lambda _, xs: Ray(*xs)
)


def convert_slope_to_direction_cosines(dx, dy):
    l_dir_cosine = dx / jnp.sqrt(1 + dx ** 2 + dy ** 2)
    m_dir_cosine = dy / jnp.sqrt(1 + dx ** 2 + dy ** 2)
    n_dir_cosine = 1 / jnp.sqrt(1 + dx ** 2 + dy ** 2)
    return l_dir_cosine, m_dir_cosine, n_dir_cosine

def calculate_direction_cosines(x0, y0, z0, x1, y1, z1):

    vx = x1 - x0
    vy = y1 - y0
    vz = z1 - z0
    v_mag = jnp.sqrt(vx**2 + vy**2 + vz**2)

    # And it's direction cosines
    M = vy / v_mag
    L = vx / v_mag
    N = vz / v_mag

    return L, M, N


def Q2_inv(Q1_inv, A, B, C, D):
    return C + D @ Q1_inv @ jnp.linalg.inv(A + B @ Q1_inv)
