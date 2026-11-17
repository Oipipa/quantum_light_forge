# Numerical notes

We begin with scalar interference as a simple starting point, then move to a projected vector coupled-dipole model. The atoms lie in a plane, with transition dipoles oriented along $z$. The external drive propagates along $+x$ and is $z$ polarized. The vector Green tensor describes both the field exchanged between atoms and the field at the observation plane. For the far field, we use the transverse projector $I-\hat n\hat n$. All quantities are in relative units: $\lambda=1$, $\epsilon_0=1$, $\Gamma_{\rm rad}=1$, and the incident amplitude is one.

In the weak-driving regime, the polarizability is $\alpha_i=-(6\pi/k^3)(\Gamma_{\rm rad}/2)/(\Delta_i+i\Gamma_{\rm tot}/2)$, using the $e^{-i\omega t}$ time convention. For an isolated atom with radiative decay only, $\mathrm{Im}(1/\alpha)=-k^3/(6\pi)$, consistent with the optical theorem. The outgoing dyadic Green function and coupled linear system follow the conventions of [coupled-dipole theory](https://nano-optics.ac.nz/CoupledDipole.jl/dev/theory/). We omit the self term because radiative linewidth is already included in the polarizability. The scalar model uses $e^{ikr}/(4\pi r)$ as a first approximation.

For a scalar five-atom line with $d=0.55\lambda$, coupling shifts the angle-integrated response peak from $\Delta=0$ to $0.275\Gamma$. In one nine-atom vector-array configuration, a local detuning gradient from $-0.7$ to $+0.7\,\Gamma/\lambda$ moves the forward maximum from $-7.5^\circ$ to $+7.5^\circ$. These angles come from the sampled forward plane, rather than the full three-dimensional pattern, which is available in `figures/radiation3d.html`.

| Calculation | Baseline | Modified | Change |
| --- | ---: | ---: | ---: |
| Scalar spectral peak | $0\Gamma$ | $0.275\Gamma$ | $+0.275\Gamma$ |
| Forward lobe, detuning gradient | $-7.5^\circ$ | $+7.5^\circ$ | $15^\circ$ |
| Target total-field intensity | 0.8079 | 1.4426 | $1.786\times$ |
| Probe intensity, seeded position disorder | 0.839 | 0.789 | $-0.050$ |

For focusing, we use a five-atom line and optimize the detunings within $[-2,2]\Gamma$ to maximize $|E_z|^2$ at $(1.6,0.25)\lambda$. The incident field stays the same before and after optimization. The defect test uses a separate nine-atom array, which explains why its baseline probe intensity is 0.839 rather than 0.8079. Removing one atom increases the probe intensity to 0.935 and shifts the far-field maximum to $-1.5^\circ$. These results describe the response at a particular probe point, not the total efficiency. Position disorder is seeded for reproducibility, and its effect depends on the configuration.

We also consider an optional ancilla-induced shift, $1.5\Gamma/[1+(r/0.8\lambda)^6]$, from a point outside the atomic plane. The largest sampled shift is $0.75\Gamma$. This modifies the local detunings and, in turn, the self-consistent atomic dipoles. It is a simplified shift model: EIT, blockade dynamics, and quantum switching are not included.

Finally, the pulse calculation uses a Gaussian rotating envelope, a frequency-dependent polarizability, $k(\delta)=k_0+\delta/c$, and $c=10\lambda\Gamma$. At each sampled frequency, we recalculate the coupled dipoles and observation field, then reconstruct the time-domain response from the complex spectra using $e^{-i\delta t}$. This captures effective narrowband dispersion without simulating the microscopic optical carrier.

With 512 time samples spaced by $0.08\Gamma^{-1}$, the input waveform is reconstructed to machine precision. At the target, the incident pulse peaks at $0.48\Gamma^{-1}$, close to the geometric delay of $0.5\Gamma^{-1}$. Truncating frequencies with $|\delta|>6\Gamma$ gives a waveform norm error of $1.09\times10^{-5}$ at the target. A small early-time amplitude floor of $6.67\times10^{-4}$ remains in the finite-window calculation; its origin has not yet been isolated.
