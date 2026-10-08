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
