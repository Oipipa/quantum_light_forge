import numpy as np
import pytest
from scipy.optimize import differential_evolution

from atoms import array_positions, k, polarizability, to_3d
from scattering import (green_tensor, scalar_far_field, solve_scalar,
                        solve_vector, vector_far_field, vector_scattered_field)
from pulse import spectrum_from_time, synthesize_at_time, time_from_spectrum


def test_wave_convention_and_scalar_interference():
    assert np.isclose(np.cos(k * 0.25 - 2 * np.pi * 0.25), 1)
    angles = np.linspace(-np.pi / 2, np.pi / 2, 101)
    one = scalar_far_field(np.array([[0.0, 0.0]]), [1], angles)
    assert np.allclose(abs(one)**2, 1)
    pair = scalar_far_field(np.array([[0.0, 0.0], [0.0, 0.0]]),
                            [1, -1], angles)
    assert np.max(abs(pair)) < 1e-14


def test_polarizability_linewidth_and_passivity():
    alpha = polarizability(np.array([-0.5, 0, 0.5]))
    assert np.isclose(abs(alpha[0])**2, abs(alpha[1])**2 / 2)
    assert np.isclose(abs(alpha[2])**2, abs(alpha[1])**2 / 2)
    assert np.allclose(alpha.imag, k**3 * abs(alpha)**2 / (6*np.pi))
    lossy = polarizability(0.2, gamma_rad=1, gamma_total=1.4)
    assert lossy.imag > k**3 * abs(lossy)**2 / (6*np.pi)


def test_scalar_solver_limits_and_residual():
    atom = np.array([[0.0, 0.0]])
    dipole, matrix = solve_scalar(atom, 0.3)
    assert np.isclose(dipole[0], polarizability(0.3))
    assert np.allclose(matrix, np.eye(1))
    positions = array_positions('square', 0.55)
    dipoles, matrix = solve_scalar(positions, 0.3)
    rhs = polarizability(0.3) * np.exp(1j*k*positions[:, 0])
    assert np.linalg.norm(matrix@dipoles-rhs)/np.linalg.norm(rhs) < 1e-12
    assert np.allclose(dipoles, np.linalg.lstsq(matrix, rhs, rcond=None)[0])


def test_green_tensor_reciprocity_and_transverse_limit():
    displacement = np.array([0.4, -0.7, 0.2])
    assert np.allclose(green_tensor(displacement), green_tensor(-displacement))
    assert np.allclose(green_tensor(displacement), green_tensor(displacement).T)
    with pytest.raises(ValueError):
        green_tensor(np.zeros(3))
    radius = 1000.0
    outgoing = k**2 * green_tensor(np.array([radius, 0, 0]))
    expected = k**2 * np.exp(1j*k*radius)/(4*np.pi*radius)
    assert abs(outgoing[2, 2]/expected-1) < 1e-3
    assert abs(outgoing[0, 0]/expected) < 1e-3


def test_vector_solver_and_far_field_consistency():
    positions = to_3d(array_positions('line', 0.5))
    dipoles, residual, condition = solve_vector(positions, 0.2)
    assert residual < 1e-12 and condition < 1e5
    independent, _, _ = solve_vector(positions, 0.2, coupling=False)
    assert np.allclose(independent, polarizability(0.2))
    direction = np.array([[1.0, 0.0, 0.0]])
    far = vector_far_field(positions, dipoles, direction)[0]
    assert abs(far[0]) < 1e-12
    radius = 10000.0
    near = vector_scattered_field(np.array([[radius, 0, 0]]),
                                  positions, dipoles)[0]
    asymptotic = k**2*np.exp(1j*k*radius)/(4*np.pi*radius)*far
    assert np.linalg.norm(near-asymptotic)/np.linalg.norm(asymptotic) < 1e-3


def test_pulse_fourier_reconstruction_and_delay():
    count, dt = 512, 0.08
    times = (np.arange(count)-count//2)*dt
    frequencies = 2*np.pi*np.fft.fftfreq(count, d=dt)
    envelope = np.exp(-0.5*(times/0.8)**2)
    spectrum = spectrum_from_time(envelope)
    assert np.max(abs(time_from_spectrum(spectrum)-envelope)) < 1e-12
    assert np.isclose(np.sum(abs(envelope)**2),
                      np.sum(abs(spectrum)**2)/count)
    for instant in (-0.4, 0.4, 1.2):
        calculated = synthesize_at_time(spectrum, frequencies, instant, times[0])
        assert abs(calculated-np.exp(-0.5*(instant/0.8)**2)) < 1e-8
    delay = 0.5
    delayed = spectrum*np.exp(1j*frequencies*delay)
    assert abs(synthesize_at_time(delayed, frequencies, delay, times[0])-1) < 1e-8


def test_fourier_resolution_and_seeded_optimization():
    estimates = []
    for count in (256, 512):
        dt = 0.08
        times = (np.arange(count)-count//2)*dt
        frequencies = 2*np.pi*np.fft.fftfreq(count, d=dt)
        pulse = np.exp(-0.5*(times/0.8)**2)
        estimates.append(synthesize_at_time(spectrum_from_time(pulse),
                                             frequencies, 0.35, times[0]))
    assert abs(estimates[0]-estimates[1]) < 1e-8
    positions = to_3d(array_positions('line', 0.45))
    target = np.array([1.6, 0.25, 0.0])
    def objective(values):
        moments, _, _ = solve_vector(positions, values)
        scattered = vector_scattered_field(target[None, :], positions, moments)[0, 2]
        return -abs(np.exp(1j*k*target[0])+scattered)**2
    bounds = [(-2, 2)]*len(positions)
    first = differential_evolution(objective, bounds, seed=42,
                                   maxiter=4, popsize=4, polish=False)
    second = differential_evolution(objective, bounds, seed=42,
                                    maxiter=4, popsize=4, polish=False)
    assert np.array_equal(first.x, second.x)
    assert first.fun < objective(np.full(len(positions), 0.2))
