# Numerical notes

The calculation starts with scalar interference and ends with a projected vector coupled-dipole model. Atoms occupy a plane; their transition dipoles point along $z$. The external drive propagates along $+x$ and is $z$ polarized. The vector Green tensor gives the interatomic field and the field on the observation plane. The far-field amplitude uses the transverse projector $I-\hat n\hat n$. The normalization is relative: $\lambda=1$, $\epsilon_0=1$, $\Gamma_{\rm rad}=1$, and incident amplitude is one. This is a fixed-transition-polarization calculation, not an isotropic three-component atomic polarizability.

The weak-driving response uses $\alpha_i=-(6\pi/k^3)(\Gamma_{\rm rad}/2)/(\Delta_i+i\Gamma_{\rm tot}/2)$ with $e^{-i\omega t}$ time convention. For a radiative-only isolated atom, $\mathrm{Im}(1/\alpha)=-k^3/(6\pi)$, as required by the optical theorem. The outgoing dyadic Green function and coupled linear system follow the same conventions used in [coupled-dipole theory](https://nano-optics.ac.nz/CoupledDipole.jl/dev/theory/). No self term is included because the polarizability already contains radiative linewidth. The scalar model uses $e^{ikr}/(4\pi r)$ as an introductory approximation and is not presented as two-dimensional electrodynamics.

The scalar five-atom line with $d=0.55\lambda$ moves its angle-integrated response peak from $\Delta=0$ for independent scatterers to $0.275\Gamma$ with coupling. At one nine-atom vector-array configuration, a local detuning gradient $-0.7$ to $+0.7\,\Gamma/\lambda$ moves the forward maximum from $-7.5^\circ$ to $+7.5^\circ$. The measured direction is restricted to the sampled forward plane; a full three-dimensional pattern is in `figures/radiation3d.html`.

| Calculation | Baseline | Modified | Change |
| --- | ---: | ---: | ---: |
| Scalar spectral peak | $0\Gamma$ | $0.275\Gamma$ | $+0.275\Gamma$ |
| Forward lobe, detuning gradient | $-7.5^\circ$ | $+7.5^\circ$ | $15^\circ$ |
| Target total-field intensity | 0.8079 | 1.4426 | $1.786\times$ |
| Probe intensity, seeded position disorder | 0.839 | 0.789 | $-0.050$ |

The focusing calculation uses a five-atom line and maximizes $|E_z|^2$ at $(1.6,0.25)\lambda$ with bounded detunings in $[-2,2]\Gamma$. It uses the same incident field before and after optimization. The defect comparison uses a separate nine-atom array, so its baseline probe intensity is 0.839 rather than the focusing baseline 0.8079. A missing atom raises that one probe intensity to 0.935 while shifting the far-field maximum to $-1.5^\circ$. One-point intensity is not a measure of total efficiency. Position disorder is seeded; the effect is configuration dependent.

The optional ancilla shift is $1.5\Gamma/[1+(r/0.8\lambda)^6]$ from an off-plane point. Its largest sampled shift is $0.75\Gamma$. It changes the atomic detunings and therefore their self-consistent dipoles. It does not include EIT, blockade dynamics, or quantum switching.

The pulse calculation uses a Gaussian rotating envelope, a frequency-dependent polarizability, $k(\delta)=k_0+\delta/c$, and $c=10\lambda\Gamma$. At each sampled frequency the coupled dipoles and observation field are recomputed; complex spectra are then synthesized with $e^{-i\delta t}$. This is an effective narrowband dispersion model, not a microscopic optical carrier simulation. A $512$-sample, $0.08\Gamma^{-1}$ time grid reconstructs the input to machine precision. The incident target peak appears at $0.48\Gamma^{-1}$ for a $0.5\Gamma^{-1}$ geometric delay. Truncating $|\delta|>6\Gamma$ gives waveform norm error $1.09\times10^{-5}$ at the target. A $6.67\times10^{-4}$ early-time amplitude floor remains in the finite-window calculation; its source was not isolated.
