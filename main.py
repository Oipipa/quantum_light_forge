import matplotlib.pyplot as plt
import numpy as np
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
