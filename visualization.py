import numpy as np
from scattering import vector_far_field

def plot_dipoles(ax, positions, dipoles, mode='phase'):
    values = np.angle(dipoles) if mode == 'phase' else abs(dipoles)
    colors = 'twilight' if mode == 'phase' else 'viridis'
    limits = (-np.pi, np.pi) if mode == 'phase' else (0, max(values) * 1.05)
    return ax.scatter(positions[:, 0], positions[:, 1], c=values, s=170,
                      cmap=colors, vmin=limits[0], vmax=limits[1], edgecolors='white')


def radiation_surface(positions, dipoles):
    theta, azimuth = np.meshgrid(np.linspace(0, np.pi, 61),
                                 np.linspace(0, 2*np.pi, 121), indexing='ij')
    directions = np.column_stack([(np.sin(theta)*np.cos(azimuth)).ravel(),
                                  (np.sin(theta)*np.sin(azimuth)).ravel(),
                                  np.cos(theta).ravel()])
    amplitude = vector_far_field(positions, dipoles, directions)
    intensity = np.sum(abs(amplitude)**2, axis=1).reshape(theta.shape)
    normalized = intensity / intensity.max()
    x_surface = normalized * np.sin(theta) * np.cos(azimuth)
    y_surface = normalized * np.sin(theta) * np.sin(azimuth)
    z_surface = normalized * np.cos(theta)
    return normalized, x_surface, y_surface, z_surface

