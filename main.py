import matplotlib.pyplot as plt
import numpy as np

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
