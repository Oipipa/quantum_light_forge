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
