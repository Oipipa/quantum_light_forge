import numpy as np

k = 2 * np.pi

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


def polarizability(detuning, gamma_rad=1.0, gamma_total=1.0):
    # epsilon_0 = 1, length in wavelengths, rates in radiative linewidths.
    return -(6 * np.pi / k ** 3) * (gamma_rad / 2) / (detuning + 1j * gamma_total / 2)


def to_3d(positions):
    return np.column_stack([positions, np.zeros(len(positions))])


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

