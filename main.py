import matplotlib.pyplot as plt
import numpy as np
import os
from matplotlib.animation import FuncAnimation

wavelength = 1.0
k = 2 * np.pi / wavelength
x = np.linspace(-2, 2, 401)
y = np.linspace(-1.5, 1.5, 301)
X, Y = np.meshgrid(x, y)
phase = k * X
field = np.exp(1j * phase)

fig, ax = plt.subplots(figsize=(8, 5))
im = ax.imshow(field.real, extent=[x.min(), x.max(), y.min(), y.max()],
               origin='lower', cmap='RdBu_r', vmin=-1, vmax=1)
fig.colorbar(im, ax=ax, label=r'$\mathrm{Re}(E)$')
ax.set(xlabel=r'$x/\lambda$', ylabel=r'$y/\lambda$',
       title=r'Plane wave, $+x$ propagation, $\lambda=1$')
fig.tight_layout()
fig.savefig('/tmp/qlf_plane_wave.png', dpi=140)

animation_figure, animation_ax = plt.subplots(figsize=(8, 5))
animated_image = animation_ax.imshow(field.real, extent=[-2, 2, -1.5, 1.5],
                                     origin='lower', cmap='RdBu_r', vmin=-1, vmax=1)
animation_ax.set(xlabel=r'$x/\lambda$', ylabel=r'$y/\lambda$')

def advance(frame):
    animated_image.set_data(np.real(field * np.exp(-1j * 2 * np.pi * frame / 32)))
    return (animated_image,)

wave_animation = FuncAnimation(animation_figure, advance, frames=32,
                               interval=40, blit=True)
if os.environ.get('QLF_EXPORT') == '1':
    wave_animation.save('/tmp/qlf_plane_wave.gif', writer='pillow', fps=25)

# The plot is a 2D slice of a 3D outgoing spherical wave.
source = np.array([-0.4, 0.0])
radius = np.hypot(X - source[0], Y - source[1])
point_wave = np.zeros_like(field)
safe = radius > 0.08
point_wave[safe] = np.exp(1j * k * radius[safe]) / (4 * np.pi * radius[safe])
source_fig, source_axes = plt.subplots(1, 2, figsize=(10, 4))
for ax, values, title in zip(source_axes, (field.real, point_wave.real),
                             ('Incident plane wave', 'Outgoing point wave')):
    ax.imshow(values, extent=[-2, 2, -1.5, 1.5], origin='lower',
              cmap='RdBu_r', vmin=-1, vmax=1)
    ax.set_title(title)
source_axes[1].plot(*source, 'ko', markersize=4)
source_fig.tight_layout()
source_fig.savefig('/tmp/qlf_point_wave.png', dpi=140)

def two_sources(separation, phase_difference):
    waves = []
    for sx, phase_offset in [(-separation / 2, 0), (separation / 2, phase_difference)]:
        distance = np.hypot(X - sx, Y)
        wave = np.zeros_like(field)
        valid = distance > 0.08
        wave[valid] = np.exp(1j * (k * distance[valid] + phase_offset)) / (4 * np.pi * distance[valid])
        waves.append(wave)
    return waves[0] + waves[1]

interference_fig, interference_axes = plt.subplots(2, 2, figsize=(9, 7))
for ax, separation, offset in zip(interference_axes.flat,
                                  (0.5, 0.5, 1.0, 1.0),
                                  (0, np.pi, 0, np.pi)):
    wave = two_sources(separation, offset)
    ax.imshow(wave.real, extent=[-2, 2, -1.5, 1.5], origin='lower',
              cmap='RdBu_r', vmin=-0.4, vmax=0.4)
    ax.plot([-separation / 2, separation / 2], [0, 0], 'ko', markersize=3)
    ax.set_title(rf'$d={separation}\lambda$, $\Delta\phi={offset/np.pi:g}\pi$')
interference_fig.tight_layout()
interference_fig.savefig('/tmp/qlf_two_sources.png', dpi=140)

scattered = 2.0 * point_wave
total = field + scattered
fields_fig, fields_axes = plt.subplots(1, 3, figsize=(12, 3.8))
for ax, values, title in zip(fields_axes,
                             (abs(field) ** 2, abs(scattered) ** 2, abs(total) ** 2),
                             ('Incident intensity', 'Scattered intensity', 'Total intensity')):
    image = ax.imshow(values, extent=[-2, 2, -1.5, 1.5], origin='lower', cmap='magma')
    fields_fig.colorbar(image, ax=ax, shrink=0.8)
    ax.set_title(title)
fields_fig.tight_layout()
fields_fig.savefig('/tmp/qlf_total_intensity.png', dpi=140)
probe = (200, 270)
cross_term = 2 * np.real(field[probe].conjugate() * scattered[probe])
assert np.isclose(abs(total[probe]) ** 2,
                  abs(field[probe]) ** 2 + abs(scattered[probe]) ** 2 + cross_term)
print(f'Interference term at probe: {cross_term:.4f}')

plt.style.use('dark_background')
showcase_fig, showcase_ax = plt.subplots(figsize=(9, 5), facecolor='#081322')
showcase_ax.set_facecolor('#081322')
display = showcase_ax.imshow(total.real, extent=[-2, 2, -1.5, 1.5],
                             origin='lower', cmap='coolwarm', vmin=-1.8, vmax=1.8,
                             interpolation='bilinear')
showcase_ax.plot(*source, 'o', color='#ffc87a', markersize=5)
showcase_ax.set(xlabel=r'$x/\lambda$', ylabel=r'$y/\lambda$',
                title='Plane wave and outgoing scattered wave')
showcase_fig.tight_layout()

def animate_total(frame):
    display.set_data(np.real(total * np.exp(-1j * 2 * np.pi * frame / 48)))
    return (display,)

showcase_animation = FuncAnimation(showcase_fig, animate_total, frames=48,
                                   interval=35, blit=True)
if os.environ.get('QLF_EXPORT') == '1':
    showcase_animation.save('/tmp/qlf_showcase.gif', writer='pillow', fps=28, dpi=140)

def scalar_field(positions, amplitudes, grid_x=X, grid_y=Y):
    result = np.zeros_like(grid_x, dtype=complex)
    for (sx, sy), amplitude in zip(positions, amplitudes):
        distance = np.hypot(grid_x - sx, grid_y - sy)
        valid = distance > 0.08
        result[valid] += amplitude * np.exp(1j * k * distance[valid]) / (4 * np.pi * distance[valid])
    return result

line_fig, line_axes = plt.subplots(1, 2, figsize=(10, 4))
for ax, spacing in zip(line_axes, (0.4, 0.8)):
    line_atoms = np.column_stack([np.zeros(5), (np.arange(5) - 2) * spacing])
    emission = scalar_field(line_atoms, np.ones(5))
    ax.imshow(emission.real, extent=[-2, 2, -1.5, 1.5], origin='lower',
              cmap='coolwarm', vmin=-0.5, vmax=0.5)
    ax.scatter(line_atoms[:, 0], line_atoms[:, 1], c='#ffc87a', s=16)
    ax.set_title(rf'5 emitters, $d={spacing}\lambda$')
line_fig.tight_layout()
line_fig.savefig('/tmp/qlf_line_array.png', dpi=140)

def array_positions(kind, spacing=0.5):
    if kind == 'line':
        return np.column_stack([np.zeros(5), (np.arange(5) - 2) * spacing])
    if kind == 'square':
        xx, yy = np.meshgrid((np.arange(3) - 1) * spacing,
                             (np.arange(3) - 1) * spacing)
        return np.column_stack([xx.ravel(), yy.ravel()])
    if kind == 'circle':
        angle = np.arange(8) * 2 * np.pi / 8
        return np.column_stack([spacing * np.cos(angle), spacing * np.sin(angle)])
    if kind == 'rows':
        xx, yy = np.meshgrid([-spacing / 2, spacing / 2],
                             (np.arange(5) - 2) * spacing)
        return np.column_stack([xx.ravel(), yy.ravel()])
    raise ValueError(kind)

geometry_fig, geometry_axes = plt.subplots(2, 2, figsize=(9, 7))
for ax, kind in zip(geometry_axes.flat, ('line', 'square', 'circle', 'rows')):
    atoms = array_positions(kind)
    wave = scalar_field(atoms, np.ones(len(atoms)))
    ax.imshow(wave.real, extent=[-2, 2, -1.5, 1.5], origin='lower',
              cmap='coolwarm', vmin=-0.6, vmax=0.6)
    ax.scatter(atoms[:, 0], atoms[:, 1], c='#ffc87a', s=14)
    ax.set_title(kind)
geometry_fig.tight_layout()
geometry_fig.savefig('/tmp/qlf_geometries.png', dpi=140)

def scalar_far_field(positions, amplitudes, angles):
    directions = np.column_stack([np.cos(angles), np.sin(angles)])
    phase = np.exp(-1j * k * directions @ positions.T)
    return phase @ np.asarray(amplitudes)

angles = np.linspace(-np.pi, np.pi, 721)
far_fig = plt.figure(figsize=(7, 6))
far_ax = far_fig.add_subplot(projection='polar')
for kind in ('line', 'square', 'circle'):
    atoms = array_positions(kind)
    amplitude = scalar_far_field(atoms, np.ones(len(atoms)), angles)
    far_ax.plot(angles, abs(amplitude) ** 2 / len(atoms) ** 2, label=kind)
far_ax.legend(loc='lower left')
far_fig.tight_layout()
far_fig.savefig('/tmp/qlf_scalar_far_field.png', dpi=150)
assert np.allclose(abs(scalar_far_field(np.zeros((1, 2)), [1], angles)) ** 2, 1)

steering_atoms = array_positions('line', 0.4)
steering_angles = np.linspace(-np.pi / 2, np.pi / 2, 721)
steering_fig, steering_ax = plt.subplots(figsize=(7, 4))
for target_deg in (-30, 0, 30):
    target = np.deg2rad(target_deg)
    prescribed = np.exp(1j * k * steering_atoms[:, 1] * np.sin(target))
    pattern = abs(scalar_far_field(steering_atoms, prescribed, steering_angles)) ** 2
    steering_ax.plot(np.rad2deg(steering_angles), pattern / pattern.max(),
                     label=rf'target ${target_deg}^\circ$')
    observed = np.rad2deg(steering_angles[np.argmax(pattern)])
    assert abs(observed - target_deg) < 0.5
steering_ax.set(xlabel='Forward observation angle (degrees)',
                ylabel='Intensity / peak', title='Prescribed emitter phases')
steering_ax.legend()
steering_fig.tight_layout()
steering_fig.savefig('/tmp/qlf_prescribed_steering.png', dpi=150)

moving_fig, moving_ax = plt.subplots(figsize=(7, 4))
moving_line, = moving_ax.plot([], [], color='#5eead4')
moving_ax.set(xlim=(-90, 90), ylim=(0, 1.05),
              xlabel='Observation angle (degrees)', ylabel='Intensity / peak')
def move_beam(frame):
    target = np.deg2rad(-35 + 70 * frame / 39)
    prescribed = np.exp(1j * k * steering_atoms[:, 1] * np.sin(target))
    pattern = abs(scalar_far_field(steering_atoms, prescribed, steering_angles)) ** 2
    moving_line.set_data(np.rad2deg(steering_angles), pattern / pattern.max())
    return (moving_line,)
moving_animation = FuncAnimation(moving_fig, move_beam, frames=40, blit=True)
if os.environ.get('QLF_EXPORT') == '1':
    moving_animation.save('/tmp/qlf_prescribed_steering.gif', writer='pillow', fps=20)

def polarizability(detuning, gamma_rad=1.0, gamma_total=1.0):
    # epsilon_0 = 1, length in wavelengths, rates in radiative linewidths.
    return -(6 * np.pi / k ** 3) * (gamma_rad / 2) / (detuning + 1j * gamma_total / 2)

detunings = np.linspace(-3, 3, 601)
alpha_values = polarizability(detunings)
response_fig, response_axes = plt.subplots(1, 2, figsize=(9, 3.5))
response_axes[0].plot(detunings, abs(alpha_values) ** 2)
response_axes[0].set(xlabel=r'$\Delta/\Gamma$', ylabel=r'$|\alpha|^2$')
response_axes[1].plot(detunings, np.angle(alpha_values))
response_axes[1].set(xlabel=r'$\Delta/\Gamma$', ylabel=r'$\arg\alpha$ (rad)')
response_fig.tight_layout()
response_fig.savefig('/tmp/qlf_polarizability.png', dpi=150)
assert np.isclose(polarizability(0).imag, 6 * np.pi / k ** 3)
assert np.isclose(abs(polarizability(0.5)) ** 2,
                  abs(polarizability(0)) ** 2 / 2)
assert np.isclose((1 / polarizability(0)).imag, -k ** 3 / (6 * np.pi))

atomic_line = array_positions('line', 0.55)
incident_at_atoms = np.exp(1j * k * atomic_line[:, 0])
independent_dipoles = polarizability(0.4) * incident_at_atoms
independent_scattered = scalar_field(atomic_line, k ** 2 * independent_dipoles)
independent_far = scalar_far_field(atomic_line, k ** 2 * independent_dipoles,
                                   steering_angles)
independent_fig, independent_axes = plt.subplots(1, 2, figsize=(10, 4))
independent_axes[0].imshow(abs(field + independent_scattered) ** 2,
                           extent=[-2, 2, -1.5, 1.5], origin='lower', cmap='magma',
                           vmin=0, vmax=6)
independent_axes[0].scatter(atomic_line[:, 0], atomic_line[:, 1],
                            c='#5eead4', s=15)
independent_axes[0].set_title('Independent atomic response')
independent_axes[1].plot(np.rad2deg(steering_angles), abs(independent_far) ** 2)
independent_axes[1].set(xlabel='Observation angle (degrees)',
                         ylabel='Scalar far-field intensity')
independent_fig.tight_layout()
independent_fig.savefig('/tmp/qlf_independent_atoms.png', dpi=150)

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

cooperative_dipoles, scalar_matrix = solve_scalar(atomic_line, 0.4)
assert np.linalg.norm(scalar_matrix @ cooperative_dipoles
                      - polarizability(0.4) * incident_at_atoms) < 1e-12
cooperative_far = scalar_far_field(atomic_line, k ** 2 * cooperative_dipoles,
                                   steering_angles)
coupling_fig, coupling_ax = plt.subplots(figsize=(7, 4))
coupling_ax.plot(np.rad2deg(steering_angles), abs(independent_far) ** 2,
                 label='Independent')
coupling_ax.plot(np.rad2deg(steering_angles), abs(cooperative_far) ** 2,
                 label='Coupled')
coupling_ax.set(xlabel='Observation angle (degrees)', ylabel='Scalar intensity')
coupling_ax.legend()
coupling_fig.tight_layout()
coupling_fig.savefig('/tmp/qlf_scalar_coupling.png', dpi=150)

spectrum_detunings = np.linspace(-2, 2, 161)
independent_spectrum = []
cooperative_spectrum = []
for detuning in spectrum_detunings:
    independent = polarizability(detuning) * incident_at_atoms
    cooperative, _ = solve_scalar(atomic_line, detuning)
    independent_spectrum.append(np.trapezoid(
        abs(scalar_far_field(atomic_line, independent, steering_angles)) ** 2,
        steering_angles))
    cooperative_spectrum.append(np.trapezoid(
        abs(scalar_far_field(atomic_line, cooperative, steering_angles)) ** 2,
        steering_angles))
spectrum_fig, spectrum_ax = plt.subplots(figsize=(7, 4))
spectrum_ax.plot(spectrum_detunings, independent_spectrum, label='Independent')
spectrum_ax.plot(spectrum_detunings, cooperative_spectrum, label='Cooperative')
spectrum_ax.set(xlabel=r'$\Delta/\Gamma$', ylabel='Angle-integrated scalar response')
spectrum_ax.legend()
spectrum_fig.tight_layout()
spectrum_fig.savefig('/tmp/qlf_scalar_spectrum.png', dpi=150)
print('Scalar spectral peaks:',
      spectrum_detunings[np.argmax(independent_spectrum)],
      spectrum_detunings[np.argmax(cooperative_spectrum)])

spacing_fig, spacing_axes = plt.subplots(1, 2, figsize=(10, 4))
for spacing in (0.35, 0.55, 0.8):
    positions = array_positions('line', spacing)
    dipoles, _ = solve_scalar(positions, 0.4)
    pattern = abs(scalar_far_field(positions, dipoles, steering_angles)) ** 2
    spacing_axes[0].plot(np.rad2deg(steering_angles), pattern,
                         label=rf'$d={spacing}\lambda$')
    scan = []
    for detuning in spectrum_detunings:
        trial, _ = solve_scalar(positions, detuning)
        scan.append(np.trapezoid(abs(scalar_far_field(positions, trial,
                                steering_angles)) ** 2, steering_angles))
    spacing_axes[1].plot(spectrum_detunings, scan, label=rf'$d={spacing}\lambda$')
spacing_axes[0].set(xlabel='Observation angle (degrees)', ylabel='Intensity')
spacing_axes[1].set(xlabel=r'$\Delta/\Gamma$', ylabel='Integrated response')
for ax in spacing_axes:
    ax.legend()
spacing_fig.tight_layout()
spacing_fig.savefig('/tmp/qlf_spacing.png', dpi=150)

from scipy.linalg import lstsq

check_positions = array_positions('square', 0.55)
check_dipoles, check_matrix = solve_scalar(check_positions, 0.25)
check_rhs = polarizability(0.25) * np.exp(1j * k * check_positions[:, 0])
least_squares, _, _, _ = lstsq(check_matrix, check_rhs)
relative_residual = np.linalg.norm(check_matrix @ check_dipoles - check_rhs) / np.linalg.norm(check_rhs)
condition_number = np.linalg.cond(check_matrix)
assert relative_residual < 1e-12
assert np.allclose(check_dipoles, least_squares, rtol=1e-11, atol=1e-12)
single_dipole, single_matrix = solve_scalar(np.array([[0.0, 0.0]]), 0.25)
assert np.isclose(single_dipole[0], polarizability(0.25))
assert np.array_equal(single_matrix, np.eye(1))
assert np.allclose(np.linalg.solve(np.eye(len(check_rhs)), check_rhs), check_rhs)
print(f'Scalar solver residual={relative_residual:.2e}, condition={condition_number:.2f}')

def green_tensor(displacement):
    displacement = np.asarray(displacement, dtype=float)
    radius = np.linalg.norm(displacement, axis=-1)
    if np.any(radius == 0):
        raise ValueError('Green tensor is undefined at the source')
    direction = displacement / radius[..., None]
    kr = k * radius
    identity = np.eye(3)
    radial = direction[..., :, None] * direction[..., None, :]
    transverse_coefficient = np.asarray(1 + 1j / kr - 1 / kr ** 2)
    radial_coefficient = np.asarray(-1 - 3j / kr + 3 / kr ** 2)
    return np.asarray(np.exp(1j * kr) / (4 * np.pi * radius))[..., None, None] * (
        transverse_coefficient[..., None, None] * identity
        + radial_coefficient[..., None, None] * radial)

def dipole_field(displacement, dipole):
    return k ** 2 * np.einsum('...ab,b->...a', green_tensor(displacement), dipole)

polar_angles = np.linspace(0, np.pi, 361)
directions_3d = np.column_stack([np.sin(polar_angles),
                                  np.zeros_like(polar_angles), np.cos(polar_angles)])
z_polarized = np.array([0.0, 0.0, 1.0])
transverse = z_polarized - directions_3d * (directions_3d @ z_polarized)[:, None]
vector_pattern = np.sum(abs(transverse) ** 2, axis=1)
assert np.allclose(vector_pattern, np.sin(polar_angles) ** 2)
assert vector_pattern[0] < 1e-12 and np.isclose(vector_pattern[180], 1)
vector_fig, vector_ax = plt.subplots(figsize=(6, 4))
vector_ax.plot(np.rad2deg(polar_angles), vector_pattern, label='Vector dipole')
vector_ax.plot(np.rad2deg(polar_angles), np.ones_like(polar_angles), '--',
               label='Scalar emitter')
vector_ax.set(xlabel='Angle from dipole axis (degrees)', ylabel='Far-field intensity / peak')
vector_ax.legend()
vector_fig.tight_layout()
vector_fig.savefig('/tmp/qlf_vector_dipole.png', dpi=150)

def to_3d(positions):
    return np.column_stack([positions, np.zeros(len(positions))])

def solve_vector(positions, detuning, coupling=True):
    # Fixed z-oriented transition: solve only the projected z component.
    positions = np.asarray(positions, dtype=float)
    alpha = np.broadcast_to(polarizability(detuning), len(positions))
    incoming = np.exp(1j * k * positions[:, 0])
    matrix = np.eye(len(positions), dtype=complex)
    if coupling:
        for i in range(len(positions)):
            for j in range(len(positions)):
                if i != j:
                    matrix[i, j] -= alpha[i] * k ** 2 * green_tensor(positions[i] - positions[j])[2, 2]
    dipoles = np.linalg.solve(matrix, alpha * incoming)
    residual = np.linalg.norm(matrix @ dipoles - alpha * incoming) / np.linalg.norm(alpha * incoming)
    return dipoles, residual, np.linalg.cond(matrix)

def vector_scattered_field(points, positions, dipoles, exclusion=0.08):
    result = np.zeros(points.shape, dtype=complex)
    valid = np.ones(points.shape[:-1], dtype=bool)
    for atom, amplitude in zip(positions, dipoles):
        displacement = points - atom
        radius = np.linalg.norm(displacement, axis=-1)
        valid &= radius > exclusion
        safe_displacement = displacement.copy()
        safe_displacement[radius <= exclusion] = [exclusion, 0, 0]
        result += k ** 2 * np.einsum('...ab,b->...a', green_tensor(safe_displacement),
                                     np.array([0, 0, amplitude]))
    result[~valid] = np.nan
    return result

def vector_far_field(positions, dipoles, directions):
    moments = np.column_stack([np.zeros(len(dipoles)), np.zeros(len(dipoles)), dipoles])
    phases = np.exp(-1j * k * directions @ positions.T)
    summed = phases @ moments
    return summed - directions * np.sum(directions * summed, axis=-1)[:, None]

vector_atoms = to_3d(array_positions('square', 0.55))
vector_dipoles, vector_residual, vector_condition = solve_vector(vector_atoms, 0.2)
assert vector_residual < 1e-12
view_x, view_y = np.meshgrid(np.linspace(-2, 2, 241), np.linspace(-1.5, 1.5, 181))
view_points = np.stack([view_x, view_y, np.zeros_like(view_x)], axis=-1)
vector_scattered = vector_scattered_field(view_points, vector_atoms, vector_dipoles)
vector_total_z = np.exp(1j * k * view_x) + vector_scattered[..., 2]
forward_directions = np.column_stack([np.cos(steering_angles),
                                      np.sin(steering_angles), np.zeros_like(steering_angles)])
vector_angular = np.sum(abs(vector_far_field(vector_atoms, vector_dipoles,
                                             forward_directions)) ** 2, axis=1)
vector_gate_fig, vector_gate_axes = plt.subplots(1, 2, figsize=(11, 4))
vector_gate_axes[0].imshow(np.real(vector_total_z), extent=[-2, 2, -1.5, 1.5],
                           origin='lower', cmap='coolwarm', vmin=-2, vmax=2)
vector_gate_axes[0].scatter(vector_atoms[:, 0], vector_atoms[:, 1], c='#ffc87a', s=15)
vector_gate_axes[0].set_title(r'$\mathrm{Re}(E_z)$, coupled vector dipoles')
vector_gate_axes[1].plot(np.rad2deg(steering_angles), vector_angular)
vector_gate_axes[1].set(xlabel='Forward angle (degrees)', ylabel='Transverse far-field intensity')
vector_gate_fig.tight_layout()
vector_gate_fig.savefig('/tmp/qlf_vector_coupled.png', dpi=150)
print(f'Vector solver residual={vector_residual:.2e}, condition={vector_condition:.2f}')

def plot_dipoles(ax, positions, dipoles, mode='phase'):
    values = np.angle(dipoles) if mode == 'phase' else abs(dipoles)
    colors = 'twilight' if mode == 'phase' else 'viridis'
    limits = (-np.pi, np.pi) if mode == 'phase' else (0, max(values) * 1.05)
    return ax.scatter(positions[:, 0], positions[:, 1], c=values, s=170,
                      cmap=colors, vmin=limits[0], vmax=limits[1], edgecolors='white')

phase_fig, phase_axes = plt.subplots(2, 2, figsize=(8, 7))
for row, detuning in enumerate((-0.6, 0.6)):
    moments, _, _ = solve_vector(vector_atoms, detuning)
    for col, mode in enumerate(('amplitude', 'phase')):
        image = plot_dipoles(phase_axes[row, col], vector_atoms, moments, mode)
        phase_axes[row, col].set(xlim=(-1, 1), ylim=(-1, 1), aspect='equal',
                                 title=rf'$\Delta={detuning}\Gamma$, {mode}')
        phase_fig.colorbar(image, ax=phase_axes[row, col], shrink=0.75)
phase_fig.tight_layout()
phase_fig.savefig('/tmp/qlf_dipole_maps.png', dpi=150)

steer_atoms = to_3d(np.column_stack([np.zeros(9), (np.arange(9) - 4) * 0.35]))
gradient_fig, gradient_ax = plt.subplots(figsize=(7, 4))
gradient_peaks = {}
for gradient in (-0.7, 0.0, 0.7):
    local_detuning = 0.2 + gradient * steer_atoms[:, 1]
    moments, residual, condition = solve_vector(steer_atoms, local_detuning)
    radiation = vector_far_field(steer_atoms, moments, forward_directions)
    intensity = np.sum(abs(radiation) ** 2, axis=1)
    peak_angle = np.rad2deg(steering_angles[np.argmax(intensity)])
    gradient_peaks[gradient] = peak_angle
    gradient_ax.plot(np.rad2deg(steering_angles), intensity,
                     label=rf'gradient ${gradient:+.1f}\Gamma/\lambda$')
    assert residual < 1e-12 and condition < 1e5
gradient_ax.set(xlabel='Forward angle (degrees)', ylabel='Vector far-field intensity')
gradient_ax.legend()
gradient_fig.tight_layout()
gradient_fig.savefig('/tmp/qlf_detuning_steering.png', dpi=150)
print('Detuning-gradient peak angles:', gradient_peaks)

from scipy.optimize import differential_evolution

focus_atoms = to_3d(array_positions('line', 0.45))
focus_target = np.array([1.6, 0.25, 0.0])
focus_incoming = np.exp(1j * k * focus_atoms[:, 0])
focus_coupling = np.zeros((len(focus_atoms), len(focus_atoms)), dtype=complex)
for i in range(len(focus_atoms)):
    for j in range(len(focus_atoms)):
        if i != j:
            focus_coupling[i, j] = k ** 2 * green_tensor(focus_atoms[i] - focus_atoms[j])[2, 2]
target_green = np.array([k ** 2 * green_tensor(focus_target - atom)[2, 2]
                         for atom in focus_atoms])

def focus_dipoles(local_detunings):
    alpha = polarizability(np.asarray(local_detunings))
    matrix = np.eye(len(alpha)) - alpha[:, None] * focus_coupling
    return np.linalg.solve(matrix, alpha * focus_incoming)

def focus_intensity(local_detunings):
    moments = focus_dipoles(local_detunings)
    field_at_target = np.exp(1j * k * focus_target[0]) + target_green @ moments
    return abs(field_at_target) ** 2

baseline_detunings = np.full(len(focus_atoms), 0.2)
baseline_focus = focus_intensity(baseline_detunings)
optimization = differential_evolution(lambda values: -focus_intensity(values),
                                      [(-2, 2)] * len(focus_atoms),
                                      seed=42, maxiter=25, popsize=6, tol=1e-5)
optimized_focus = focus_intensity(optimization.x)
assert optimized_focus > baseline_focus
focus_profile_y = np.linspace(-1.2, 1.2, 241)
focus_profile_points = np.column_stack([np.full_like(focus_profile_y, focus_target[0]),
                                        focus_profile_y, np.zeros_like(focus_profile_y)])
focus_fig, focus_ax = plt.subplots(figsize=(7, 4))
for label, detuning_choice in [('Uniform', baseline_detunings),
                                ('Optimized', optimization.x)]:
    moments = focus_dipoles(detuning_choice)
    scattered_profile = vector_scattered_field(focus_profile_points,
                                                focus_atoms, moments)[:, 2]
    intensity = abs(np.exp(1j * k * focus_target[0]) + scattered_profile) ** 2
    focus_ax.plot(focus_profile_y, intensity, label=label)
focus_ax.axvline(focus_target[1], color='white', linestyle=':')
focus_ax.set(xlabel=r'$y/\lambda$ at $x=1.6\lambda$', ylabel=r'$|E_z|^2$')
focus_ax.legend()
focus_fig.tight_layout()
focus_fig.savefig('/tmp/qlf_focus.png', dpi=150)
print(f'Focus intensity: {baseline_focus:.4f} -> {optimized_focus:.4f}, '
      f'factor {optimized_focus/baseline_focus:.3f}')

from matplotlib.widgets import Slider

plt.close('all')
steering_demo, steering_panels = plt.subplots(1, 3, figsize=(12, 4))
steering_demo.subplots_adjust(bottom=0.22)
slider_axis = steering_demo.add_axes([0.22, 0.06, 0.56, 0.04])
gradient_slider = Slider(slider_axis, 'Detuning gradient', -1.0, 1.0, valinit=0.0)
demo_field = steering_panels[0].imshow(np.zeros_like(view_x),
                                       extent=[-2, 2, -1.5, 1.5], origin='lower',
                                       cmap='coolwarm', vmin=-2, vmax=2)
steering_panels[0].scatter(steer_atoms[:, 0], steer_atoms[:, 1], c='#ffc87a', s=12)
demo_phases = plot_dipoles(steering_panels[1], steer_atoms, np.ones(len(steer_atoms)))
demo_far, = steering_panels[2].plot([], [], color='#5eead4')
steering_panels[0].set_title(r'$\mathrm{Re}(E_z)$')
steering_panels[1].set_title('Calculated dipole phase')
steering_panels[2].set(xlim=(-90, 90), xlabel='Far-field angle (degrees)',
                       ylabel='Intensity')

def update_steering(value):
    moments, _, _ = solve_vector(steer_atoms, 0.2 + value * steer_atoms[:, 1])
    scattered_now = vector_scattered_field(view_points, steer_atoms, moments)[..., 2]
    demo_field.set_data(np.real(np.exp(1j * k * view_x) + scattered_now))
    demo_phases.set_array(np.angle(moments))
    intensity = np.sum(abs(vector_far_field(steer_atoms, moments,
                                             forward_directions)) ** 2, axis=1)
    demo_far.set_data(np.rad2deg(steering_angles), intensity)
    steering_panels[2].set_ylim(0, intensity.max() * 1.05)
    steering_demo.canvas.draw_idle()

gradient_slider.on_changed(update_steering)
update_steering(0.0)
steering_demo.savefig('/tmp/qlf_steering_demo.png', dpi=150)

rng = np.random.default_rng(23)
defect_cases = {
    'Perfect': (steer_atoms, np.full(len(steer_atoms), 0.2)),
    'Missing atom': (np.delete(steer_atoms, 2, axis=0), np.full(len(steer_atoms)-1, 0.2)),
    'Position disorder': (steer_atoms + np.column_stack([
        rng.normal(0, 0.03, len(steer_atoms)),
        rng.normal(0, 0.03, len(steer_atoms)),
        np.zeros(len(steer_atoms))]), np.full(len(steer_atoms), 0.2)),
    'Detuning defect': (steer_atoms, 0.2 + np.where(np.arange(len(steer_atoms)) == 2, 0.8, 0.0)),
}
defect_fig, defect_axes = plt.subplots(1, 2, figsize=(10, 4))
for label, (positions, local_detuning) in defect_cases.items():
    moments, _, _ = solve_vector(positions, local_detuning)
    angular = np.sum(abs(vector_far_field(positions, moments,
                                           forward_directions)) ** 2, axis=1)
    defect_axes[0].plot(np.rad2deg(steering_angles), angular, label=label)
    spectrum = []
    for frequency in np.linspace(-1, 1, 61):
        trial, _, _ = solve_vector(positions, local_detuning + frequency)
        pattern = vector_far_field(positions, trial, forward_directions)
        spectrum.append(np.trapezoid(np.sum(abs(pattern) ** 2, axis=1), steering_angles))
    defect_axes[1].plot(np.linspace(-1, 1, 61), spectrum, label=label)
    at_focus = (np.exp(1j*k*focus_target[0])
                + vector_scattered_field(focus_target[None, :], positions,
                                         moments)[0, 2])
    print(f'{label}: peak angle {np.rad2deg(steering_angles[np.argmax(angular)]):.2f}, '
          f'focus intensity {abs(at_focus)**2:.3f}')
defect_axes[0].set(xlabel='Forward angle (degrees)', ylabel='Intensity')
defect_axes[1].set(xlabel=r'Extra $\Delta/\Gamma$', ylabel='Integrated response')
for ax in defect_axes:
    ax.legend(fontsize=8)
defect_fig.tight_layout()
defect_fig.savefig('/tmp/qlf_defects.png', dpi=150)

ancilla_location = np.array([0.0, 0.35, 0.8])
distance_to_ancilla = np.linalg.norm(steer_atoms - ancilla_location, axis=1)
conditional_shift = 1.5 / (1 + (distance_to_ancilla / 0.8) ** 6)
conditional_fig, conditional_axes = plt.subplots(1, 2, figsize=(10, 4))
for label, shift in [('Ancilla off', np.zeros(len(steer_atoms))),
                     ('Effective ancilla on', conditional_shift)]:
    moments, _, _ = solve_vector(steer_atoms, 0.2 + shift)
    pattern = np.sum(abs(vector_far_field(steer_atoms, moments,
                                           forward_directions)) ** 2, axis=1)
    conditional_axes[0].plot(np.rad2deg(steering_angles), pattern, label=label)
    conditional_axes[1].plot(steer_atoms[:, 1], abs(moments), 'o-', label=label)
conditional_axes[0].set(xlabel='Forward angle (degrees)', ylabel='Intensity')
conditional_axes[1].set(xlabel=r'Atom $y/\lambda$', ylabel='Dipole amplitude')
for ax in conditional_axes:
    ax.legend()
conditional_fig.tight_layout()
conditional_fig.savefig('/tmp/qlf_conditional.png', dpi=150)
print('Conditional detuning shifts:', np.round(conditional_shift, 3))

from matplotlib.widgets import Button, CheckButtons, RadioButtons

plt.close('all')
dashboard = plt.figure(figsize=(14, 8), facecolor='#081322')
layout = dashboard.add_gridspec(3, 4, left=0.05, right=0.76,
                                bottom=0.16, top=0.93, wspace=0.48, hspace=0.55)
field_ax = dashboard.add_subplot(layout[:, :2])
atom_ax = dashboard.add_subplot(layout[0, 2:])
spectrum_ax = dashboard.add_subplot(layout[1, 2:])
radiation_ax = dashboard.add_subplot(layout[2, 2:])
field_image = field_ax.imshow(np.zeros_like(view_x), extent=[-2, 2, -1.5, 1.5],
                              origin='lower', cmap='coolwarm', vmin=-2, vmax=2)
atom_image = atom_ax.scatter([], [], c=[], cmap='twilight', vmin=-np.pi,
                              vmax=np.pi, s=75)
spectrum_line, = spectrum_ax.plot([], [], color='#c18cff')
spectrum_marker = spectrum_ax.axvline(0, color='#ffc87a', linestyle=':')
radiation_line, = radiation_ax.plot([], [], color='#5eead4')
dashboard_cache = {}
dashboard_state = {'phase': 0.0, 'playing': True, 'fields': None, 'positions': None,
                   'dipoles': None, 'spectrum': None, 'radiation': None}

def dashboard_positions(kind, count, spacing):
    if kind == 'Line':
        return to_3d(np.column_stack([np.zeros(count),
                                      (np.arange(count) - (count - 1) / 2) * spacing]))
    if kind == 'Square':
        side = max(2, int(np.ceil(np.sqrt(count))))
        xx, yy = np.meshgrid((np.arange(side) - (side - 1) / 2) * spacing,
                             (np.arange(side) - (side - 1) / 2) * spacing)
        return to_3d(np.column_stack([xx.ravel(), yy.ravel()])[:count])
    if kind == 'Circle':
        azimuth = np.arange(count) * 2 * np.pi / count
        return to_3d(np.column_stack([spacing * np.cos(azimuth),
                                      spacing * np.sin(azimuth)]))
    xx, yy = np.meshgrid([-spacing / 2, spacing / 2],
                         (np.arange(int(np.ceil(count / 2)))
                          - (int(np.ceil(count / 2)) - 1) / 2) * spacing)
    return to_3d(np.column_stack([xx.ravel(), yy.ravel()])[:count])

def dashboard_calculate(kind, count, spacing, detuning, gradient, defect):
    key = (kind, count, spacing, detuning, gradient, defect)
    if key in dashboard_cache:
        return dashboard_cache[key]
    positions = dashboard_positions(kind, count, spacing)
    if defect and len(positions) > 2:
        positions = np.delete(positions, 1, axis=0)
    local = detuning + gradient * positions[:, 1]
    moments, residual, condition = solve_vector(positions, local)
    if residual > 1e-9 or condition > 1e8:
        raise RuntimeError('Coupled-dipole solve is poorly conditioned')
    scattered_z = vector_scattered_field(view_points, positions, moments)[..., 2]
    incident_z = np.exp(1j * k * view_x)
    spec = []
    for frequency in np.linspace(-2, 2, 81):
        test, _, _ = solve_vector(positions, local + frequency)
        pattern = vector_far_field(positions, test, forward_directions)
        spec.append(np.trapezoid(np.sum(abs(pattern) ** 2, axis=1), steering_angles))
    pattern = vector_far_field(positions, moments, forward_directions)
    angular = np.sum(abs(pattern) ** 2, axis=1)
    result = (positions, moments, incident_z, scattered_z, np.asarray(spec), angular)
    dashboard_cache[key] = result
    return result

space_slider = Slider(dashboard.add_axes([0.08, 0.105, 0.18, 0.025]),
                      'Spacing', 0.25, 0.8, valinit=0.45)
number_slider = Slider(dashboard.add_axes([0.32, 0.105, 0.17, 0.025]),
                       'Atoms', 3, 16, valinit=9, valstep=1)
detuning_slider = Slider(dashboard.add_axes([0.08, 0.055, 0.18, 0.025]),
                         'Detuning', -2, 2, valinit=0.2)
gradient_control = Slider(dashboard.add_axes([0.32, 0.055, 0.17, 0.025]),
                          'Gradient', -1, 1, valinit=0)
speed_slider = Slider(dashboard.add_axes([0.55, 0.105, 0.16, 0.025]),
                      'Speed', 0.2, 3, valinit=1)
geometry_control = RadioButtons(dashboard.add_axes([0.79, 0.72, 0.19, 0.21]),
                                ('Line', 'Square', 'Circle', 'Rows'))
field_control = RadioButtons(dashboard.add_axes([0.79, 0.49, 0.19, 0.19]),
                             ('Total', 'Scattered', 'Incident'))
mode_control = RadioButtons(dashboard.add_axes([0.79, 0.29, 0.19, 0.17]),
                            ('Real', 'Intensity', 'Phase'))
defect_control = CheckButtons(dashboard.add_axes([0.79, 0.22, 0.19, 0.05]),
                              ['Missing atom'], [False])
run_button = Button(dashboard.add_axes([0.53, 0.04, 0.08, 0.04]), 'Update')
play_button = Button(dashboard.add_axes([0.62, 0.04, 0.08, 0.04]), 'Pause')
restart_button = Button(dashboard.add_axes([0.71, 0.04, 0.08, 0.04]), 'Restart')

def dashboard_draw():
    if dashboard_state['fields'] is None:
        return
    incident_z, scattered_z = dashboard_state['fields']
    selected = {'Incident': incident_z, 'Scattered': scattered_z,
                'Total': incident_z + scattered_z}[field_control.value_selected]
    mode = mode_control.value_selected
    if mode == 'Real':
        values = np.real(selected * np.exp(-1j * dashboard_state['phase']))
        limit = max(1, np.nanpercentile(abs(values), 99))
        field_image.set(cmap='coolwarm', clim=(-limit, limit))
    elif mode == 'Intensity':
        values = abs(selected) ** 2
        field_image.set(cmap='magma', clim=(0, max(1, np.nanpercentile(values, 99))))
    else:
        values = np.angle(selected)
        field_image.set(cmap='twilight', clim=(-np.pi, np.pi))
    field_image.set_data(values)
    field_ax.set_title(f'{field_control.value_selected} {mode.lower()} ' +
                       (r'$E_z$' if mode == 'Real' else 'field'))
    dashboard.canvas.draw_idle()

def dashboard_run(_=None):
    state = dashboard_calculate(geometry_control.value_selected,
                                int(number_slider.val), round(space_slider.val, 3),
                                round(detuning_slider.val, 3),
                                round(gradient_control.val, 3),
                                defect_control.get_status()[0])
    positions, moments, incident_z, scattered_z, spec, angular = state
    dashboard_state.update(positions=positions, dipoles=moments,
                           fields=(incident_z, scattered_z), spectrum=spec,
                           radiation=angular)
    atom_image.set_offsets(positions[:, :2])
    atom_image.set_array(np.angle(moments))
    atom_image.set_sizes(50 + 170 * abs(moments) / max(abs(moments)))
    atom_ax.set(xlim=(-1.8, 1.8), ylim=(-1.4, 1.4),
                title='Atomic dipole phase and amplitude')
    spectrum_line.set_data(np.linspace(-2, 2, len(spec)), spec)
    spectrum_ax.relim(); spectrum_ax.autoscale_view()
    spectrum_ax.set(xlabel=r'Extra $\Delta/\Gamma$', ylabel='Response')
    spectrum_marker.set_xdata([0, 0])
    radiation_line.set_data(np.rad2deg(steering_angles), angular)
    radiation_ax.set(xlim=(-90, 90), ylim=(0, 1.05 * max(angular)),
                      xlabel='Far-field angle', ylabel='Intensity')
    dashboard_draw()

def dashboard_tick(frame):
    if dashboard_state['playing']:
        dashboard_state['phase'] += 2 * np.pi * 0.03 * speed_slider.val
        if mode_control.value_selected == 'Real':
            dashboard_draw()
    return (field_image,)

def dashboard_play(_):
    dashboard_state['playing'] = not dashboard_state['playing']
    play_button.label.set_text('Pause' if dashboard_state['playing'] else 'Play')

def dashboard_restart(_):
    dashboard_state['phase'] = 0.0
    dashboard_draw()

run_button.on_clicked(dashboard_run)
play_button.on_clicked(dashboard_play)
restart_button.on_clicked(dashboard_restart)
field_control.on_clicked(lambda _: dashboard_draw())
mode_control.on_clicked(lambda _: dashboard_draw())
dashboard_run()
dashboard_animation = FuncAnimation(dashboard, dashboard_tick, frames=None,
                                    interval=35, blit=False, cache_frame_data=False)
dashboard.savefig('/tmp/qlf_dashboard.png', dpi=150)
