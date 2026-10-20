import numpy as np
from atoms import k, polarizability

def scalar_field(positions, amplitudes, grid_x=None, grid_y=None):
    if grid_x is None or grid_y is None:
        x = np.linspace(-2, 2, 401)
        y = np.linspace(-1.5, 1.5, 301)
        grid_x, grid_y = np.meshgrid(x, y)
    result = np.zeros_like(grid_x, dtype=complex)
    for (sx, sy), amplitude in zip(positions, amplitudes):
        distance = np.hypot(grid_x - sx, grid_y - sy)
        valid = distance > 0.08
        result[valid] += amplitude * np.exp(1j * k * distance[valid]) / (4 * np.pi * distance[valid])
    return result


def scalar_far_field(positions, amplitudes, angles):
    directions = np.column_stack([np.cos(angles), np.sin(angles)])
    phase = np.exp(-1j * k * directions @ positions.T)
    return phase @ np.asarray(amplitudes)


def scalar_coupling_matrix(positions):
    displacement = positions[:, None, :] - positions[None, :, :]
    distance = np.linalg.norm(displacement, axis=-1)
    matrix = np.zeros(distance.shape, dtype=complex)
    mask = ~np.eye(len(positions), dtype=bool)
    matrix[mask] = k ** 2 * np.exp(1j * k * distance[mask]) / (4 * np.pi * distance[mask])
    return matrix


def solve_scalar(positions, detuning):
    alpha = np.broadcast_to(polarizability(detuning), len(positions))
    incident = np.exp(1j * k * positions[:, 0])
    matrix = np.eye(len(positions)) - alpha[:, None] * scalar_coupling_matrix(positions)
    dipoles = np.linalg.solve(matrix, alpha * incident)
    return dipoles, matrix


def green_tensor(displacement, wave_number=k):
    displacement = np.asarray(displacement, dtype=float)
    radius = np.linalg.norm(displacement, axis=-1)
    if np.any(radius == 0):
        raise ValueError('Green tensor is undefined at the source')
    direction = displacement / radius[..., None]
    kr = wave_number * radius
    identity = np.eye(3)
    radial = direction[..., :, None] * direction[..., None, :]
    transverse_coefficient = np.asarray(1 + 1j / kr - 1 / kr ** 2)
    radial_coefficient = np.asarray(-1 - 3j / kr + 3 / kr ** 2)
    return np.asarray(np.exp(1j * kr) / (4 * np.pi * radius))[..., None, None] * (
        transverse_coefficient[..., None, None] * identity
        + radial_coefficient[..., None, None] * radial)


def dipole_field(displacement, dipole):
    return k ** 2 * np.einsum('...ab,b->...a', green_tensor(displacement), dipole)


def solve_vector(positions, detuning, coupling=True, wave_number=k):
    # Fixed z-oriented transition: solve only the projected z component.
    positions = np.asarray(positions, dtype=float)
    alpha = np.broadcast_to(polarizability(detuning), len(positions))
    incoming = np.exp(1j * wave_number * positions[:, 0])
    matrix = np.eye(len(positions), dtype=complex)
    if coupling:
        for i in range(len(positions)):
            for j in range(len(positions)):
                if i != j:
                    matrix[i, j] -= alpha[i] * wave_number ** 2 * green_tensor(
                        positions[i] - positions[j], wave_number)[2, 2]
    dipoles = np.linalg.solve(matrix, alpha * incoming)
    residual = np.linalg.norm(matrix @ dipoles - alpha * incoming) / np.linalg.norm(alpha * incoming)
    return dipoles, residual, np.linalg.cond(matrix)


def vector_scattered_field(points, positions, dipoles, exclusion=0.08,
                           wave_number=k):
    result = np.zeros(points.shape, dtype=complex)
    valid = np.ones(points.shape[:-1], dtype=bool)
    for atom, amplitude in zip(positions, dipoles):
        displacement = points - atom
        radius = np.linalg.norm(displacement, axis=-1)
        valid &= radius > exclusion
        safe_displacement = displacement.copy()
        safe_displacement[radius <= exclusion] = [exclusion, 0, 0]
        result += wave_number ** 2 * np.einsum('...ab,b->...a',
                                               green_tensor(safe_displacement, wave_number),
                                     np.array([0, 0, amplitude]))
    result[~valid] = np.nan
    return result


def vector_far_field(positions, dipoles, directions):
    moments = np.column_stack([np.zeros(len(dipoles)), np.zeros(len(dipoles)), dipoles])
    phases = np.exp(-1j * k * directions @ positions.T)
    summed = phases @ moments
    return summed - directions * np.sum(directions * summed, axis=-1)[:, None]

