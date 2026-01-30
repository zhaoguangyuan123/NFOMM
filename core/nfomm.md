
# NFOMM (Nonlinear Focal Optical Modulation Microscopy): End-to-End Process (with Equations)

This note summarizes the **NFOMM forward model** (illumination → saturation → detection) and the main **reconstruction algorithms** (single-view / multiview RL and a least-squares gradient method). It also clarifies when the model is **shift-invariant**.

---

## 0) Notation Table

| Symbol | Description |
|--------|-------------|
| $\mathbf{r}=(x,y,z)$ | Spatial coordinate |
| $\mathbf{x}=(x,y)$ | Scan coordinate |
| $\rho(\mathbf{r}) \ge 0$ | Object / fluorophore density (unknown) |
| $I(\mathbf{r})$ | Excitation intensity distribution (illumination PSF) |
| $g(I)$ | Nonlinear excitation-to-emission mapping (per-molecule) |
| $h_{\text{det}}(\mathbf{r})$ | Detection PSF |
| $p(x,y)$ | Pinhole transmission |
| $h_{\text{eff}}(\mathbf{r})$ | Effective confocal PSF |
| $\mathrm{OTF}_{\text{eff}}(\mathbf{k}) = \mathcal{F}\{h_{\text{eff}}\}(\mathbf{k})$ | Effective OTF |

---

## 1) Forward model overview

NFOMM can be viewed as three cascaded blocks:

1. **Subsystem I: Focal-field formation** (phase/polarization modulation + high-NA focusing)
2. **Subsystem II: Nonlinear excitation (saturation) → emission distribution**
3. **Subsystem III: Detection + pinhole → effective PSF/OTF**

Ultimately, the measurement at scan position $\mathbf{x}$ is modeled as a convolution:

$$y(\mathbf{x}) = (\rho * h_{\text{eff}})(\mathbf{x})$$

under the assumptions in Section 5.

---

## 2) Subsystem I: Focal field → excitation intensity $I(\mathbf{r})$

### 2.1 Vectorial high-NA focusing (Richards–Wolf form)

The complex vector field near focus:

$$\mathbf{E}(x,y,z) = iC \int_{0}^{2\pi}\int_{0}^{\theta_{\max}} \mathbf{A}_p(\theta,\phi) A_{\text{amp}}(\theta,\phi) e^{iA_{\text{phase}}(\theta,\phi)} e^{ikn z\cos\theta} e^{ikn (x\cos\phi+y\sin\phi)\sin\theta} \sin\theta \, d\theta \, d\phi$$

where $\theta_{\max}=\arcsin(\mathrm{NA}/n)$.

The excitation intensity (illumination PSF) is the sum of component intensities:

$$I(x,y,z) \equiv h_{\text{ill}}(x,y,z) = |E_x|^2 + |E_y|^2 + |E_z|^2$$

### 2.2 Power scaling

If the optical power changes from $P_0$ to $P$, then typically

$$I(\mathbf{r};P) = \frac{P}{P_0} I(\mathbf{r};P_0)$$

assuming the field shape is unchanged and only amplitude scales.

---

## 3) Subsystem II: Saturation nonlinearity (g(I)) → emission density

### 3.1 Per-molecule saturated excitation (example form)

NFOMM uses a **fluorescence saturation model** that maps local excitation intensity $I$ to an excited-state population (or emission rate proxy). A representative expression is:

$$S_1(I)=\frac{k_{01}(I)}{a k_{\text{exc}}(I)+k_0}, \qquad a \approx 3.245$$

and commonly

$$k_{01}(I) = \sigma_{01} \frac{I}{h c/\lambda} = \sigma_{01} I \frac{\lambda}{hc}$$

A simple (often used) choice is $k_{\text{exc}}(I)=k_{01}(I)$, but alternative parameterizations exist depending on the photophysics approximation.

Define the **nonlinear per-molecule response**:

$$g(I) \propto S_1(I)$$

or, if you want an emission rate,

$$g(I) = k_f S_1(I)$$

with $k_f$ a fluorescence decay rate.

### 3.2 Emitted fluorescence density

Given fluorophore density $\rho(\mathbf{r})$, the emitted fluorescence density under scan $\mathbf{x}$ is:

$$e(\mathbf{r};\mathbf{x}) = \rho(\mathbf{r}) g\big(I(\mathbf{r}-\mathbf{x})\big)$$

**Key assumption:** saturation depends on **local intensity**, not on the image after convolution. The object density multiplies the per-molecule response.

---

## 4) Subsystem III: Detection + pinhole → effective PSF/OTF

Detection and pinhole filtering are linear in emitted photons:

$$y(\mathbf{x}) = \int e(\mathbf{r};\mathbf{x}) h_{\text{det}}(\mathbf{r}-\mathbf{x}) \, d\mathbf{r}$$

and with a pinhole, the detection term is often represented as:

$$h_{\text{det,pinhole}}(\mathbf{r}) = h_{\text{det}}(\mathbf{r}) \otimes p(x,y)$$

### 4.1 Effective PSF

Combine Subsystem II and III into one **effective PSF**:

$$h_{\text{eff}}(\mathbf{r}) = g(I(\mathbf{r})) \big(h_{\text{det}}(\mathbf{r}) \otimes p(x,y)\big)$$

Then the forward model becomes:

$$y(\mathbf{x}) = \int \rho(\mathbf{r}) h_{\text{eff}}(\mathbf{r}-\mathbf{x}) \, d\mathbf{r} = (\rho * h_{\text{eff}})(\mathbf{x})$$

### 4.2 Effective OTF

$$\mathrm{OTF}_{\text{eff}}(\mathbf{k}) = \mathcal{F}\{h_{\text{eff}}\}(\mathbf{k})$$

Different phase modulations (vortex/step/line, etc.) lead to different $I(\mathbf{r})$, hence different $h_{\text{eff}}$ and different $\mathrm{OTF}_{\text{eff}}$.

---

## 5) Shift-invariance: when is “saturate PSF then convolve” valid?

The model remains **shift-invariant** if:

1. **Illumination pattern shifts rigidly** with the scan:
   $$I(\mathbf{r};\mathbf{x}) = I(\mathbf{r}-\mathbf{x})$$
2. **Saturation parameters are spatially uniform** (same $k_0,\sigma_{01},\lambda,\dots$ everywhere).
3. **Object does not reshape the excitation field** (weak absorption/scattering; no strong depletion).
4. No strong **history dependence** (bleaching/triplet buildup negligible or corrected).

If the sample strongly absorbs/scatters or the wavefront varies across the field, then $I(\mathbf{r};\mathbf{x})$ is not a pure shift, and a single global $h_{\text{eff}}$ is no longer accurate.

---

## 6) Reconstruction algorithms

Let $\rho$ be the unknown object, $y$ the measurement (2D scan image), and $h_{\text{eff}}$ known (or estimated).

### 6.1 Single-view Richardson–Lucy (RL)

Poisson likelihood leads to RL updates:

Spatial form:

$$\rho^{(k+1)} = \rho^{(k)} \cdot \left[\left(\frac{y}{\rho^{(k)} * h_{\text{eff}}}\right) * \tilde{h}_{\text{eff}}\right]$$

where $\tilde{h}_{\text{eff}}(\mathbf{r})=h_{\text{eff}}(-\mathbf{r})$, and division is pointwise.

Fourier-accelerated (conceptually):

* forward: $\rho*h = \mathcal{F}^{-1}(\mathcal{F}(\rho) \cdot \mathrm{OTF}_{\text{eff}})$
* backproject: convolution with $\tilde{h}$ corresponds to multiply by $\mathrm{OTF}^*_{\text{eff}}$.

### 6.2 Multiview RL (joint deconvolution)

With multiple measurements $\{y_\ell\}_{\ell=1}^N$ and corresponding effective PSFs $\{h_{\text{eff},\ell}\}$:

**Standard (equal weights):**
$$\rho^{(k+1)} = \rho^{(k)} \cdot \frac{1}{N}\sum_{\ell=1}^N \left[\left(\frac{y_\ell}{\rho^{(k)} * h_{\text{eff},\ell}}\right) * \tilde{h}_{\text{eff},\ell}\right]$$

**OTF-weighted (recommended):**

The standard approach gives equal weight to all views, which can dilute high-frequency information when combining views with different OTF characteristics (e.g., saturated doughnut has HF, Gaussian has LF).

A better approach is **frequency-dependent weighting** based on OTF magnitude:

$$w_\ell(\mathbf{k}) = \frac{|\mathrm{OTF}_\ell(\mathbf{k})|^2}{\sum_{j=1}^N |\mathrm{OTF}_j(\mathbf{k})|^2 + \epsilon}$$

This gives more weight to the view with stronger signal at each frequency:
- **Low frequencies**: Both views contribute (~50/50)
- **High frequencies**: Saturated doughnut dominates (has signal)
- **Doughnut nulls**: Gaussian fills in (no signal from doughnut there)

The OTF-weighted update applies weighting in Fourier domain:

$$\rho^{(k+1)} = \rho^{(k)} \cdot \sum_{\ell=1}^N \mathcal{F}^{-1}\left\{ w_\ell(\mathbf{k}) \cdot \mathcal{F}\left[\left(\frac{y_\ell}{\rho^{(k)} * h_{\text{eff},\ell}}\right) * \tilde{h}_{\text{eff},\ell}\right] \right\}$$

This fuses complementary frequency coverage (e.g., GE + SDE, or GE + multiple SLE orientations) while respecting each view's spectral contribution.

### 6.3 Least-squares gradient descent (BGD-style)

An alternative objective:

$$\min_{\rho \ge 0} \sum_{\ell=1}^N \left\| y_\ell - \rho * h_{\text{eff},\ell} \right\|_2^2$$

In Fourier domain, letting $\tilde{y}_\ell=\mathcal{F}(y_\ell)$, $\tilde{\rho}=\mathcal{F}(\rho)$:

$$\min_{\rho} \sum_{\ell=1}^N \left\| \tilde{y}_\ell - \mathrm{OTF}_{\text{eff},\ell} \cdot \tilde{\rho} \right\|_2^2$$

The gradient in Fourier domain is:

$$\nabla_{\tilde{\rho}} = -2\sum_{\ell=1}^N \mathrm{OTF}_{\text{eff},\ell}^* \left(\tilde{y}_\ell - \mathrm{OTF}_{\text{eff},\ell} \cdot \tilde{\rho}\right)$$

and update:

$$\tilde{\rho}^{(k+1)} = \tilde{\rho}^{(k)} - \eta \nabla_{\tilde{\rho}}$$

Project to $\rho\ge 0$ by inverse FFT then clamping:

$$\rho \leftarrow \max(\rho,0)$$

---

## 7) Minimal “pipeline” summary (implementation-minded)

Given a modulation $\ell$:

1. Compute focused field $\mathbf{E}_\ell$ → intensity $I_\ell(\mathbf{r})$
2. Apply saturation: $g_\ell(\mathbf{r}) = g(I_\ell(\mathbf{r}))$
3. Build effective PSF:
   $$h_{\text{eff},\ell}(\mathbf{r}) = g_\ell(\mathbf{r}) \big(h_{\text{det}}(\mathbf{r})\otimes p\big)$$
4. FFT: $\mathrm{OTF}_{\text{eff},\ell}=\mathcal{F}\{h_{\text{eff},\ell}\}$
5. Reconstruct $\rho$ from one or multiple $(y_\ell,\mathrm{OTF}_{\text{eff},\ell})$ using RL or LS-GD

---

## 8) Practical sanity checks

* Increasing illumination power $P$ should increase saturation → broaden effective frequency support (up to noise/bleaching constraints).
* If your sample is highly scattering/absorbing, expect model mismatch; multiview or blind/regularized methods may be needed.
* Always verify $h_{\text{eff}}$ is properly normalized for your deconvolution convention.

---

## 9) Pinhole Model and Airy Unit Calculation

### 9.1 Airy Unit Definition

The **Airy Unit (AU)** is the standard measure for confocal pinhole size. At the pinhole plane:

$$d_{\text{AU}} = \frac{1.22 \lambda_{\text{em}} M}{\text{NA}_{\text{obj}}}$$

where:
- $\lambda_{\text{em}}$: emission wavelength
- $M$: magnification from sample to pinhole plane
- $\text{NA}_{\text{obj}}$: objective numerical aperture

At the sample plane (M=1):

$$d_{\text{AU}} = \frac{1.22 \lambda}{\text{NA}}$$

The **first zero** of the Airy disk occurs at radius $r = 0.61\lambda/\text{NA}$, so $d_{\text{AU}}$ equals the diameter of the first Airy disk.

### 9.2 Pinhole Size Choices

| Pinhole Size | Effect |
|--------------|--------|
| **< 0.5 AU** | Near-ideal confocal (point pinhole approximation), maximum sectioning, lower signal |
| **1 AU** | Standard confocal (good tradeoff: sectioning vs signal) |
| **> 2 AU** | Approaching widefield (reduced sectioning, higher signal) |

### 9.3 Implementation

The pinhole transmission function $p(x,y)$ is a circular disk:

$$p(x,y) = \begin{cases} 1 & \text{if } \sqrt{x^2+y^2} \le r_{\text{pinhole}} \\ 0 & \text{otherwise} \end{cases}$$

For $\alpha$ AU pinhole:
$$r_{\text{pinhole}} = \frac{\alpha \cdot d_{\text{AU}}}{2}$$

Detection PSF with pinhole:
$$h_{\text{det,pinhole}} = h_{\text{det}} \otimes p$$

**Important:** The effective PSF uses **element-wise multiplication** (·), not convolution:
$$h_{\text{eff}} = g(I) \cdot h_{\text{det,pinhole}}$$

---

## 10) Simulation Parameters (Reference Implementation)

### 10.1 Optical System Parameters

From `params.py` (PSFConfig):

| Parameter | Value | Description |
|-----------|-------|-------------|
| Wavelength ($\lambda$) | 632.8 nm | Excitation wavelength |
| Entrance pupil ($D_0$) | 3.32 mm | Pupil diameter |
| Focal length ($f$) | 1.8 mm | Objective focal length |
| Refractive index ($n$) | 1.518 | Immersion medium (oil) |
| **NA** | **~1.4** | $n \sin(\arcsin(D_0/2f))$ |

### 10.2 Computed Optical Properties

| Property | Formula | Value |
|----------|---------|-------|
| Diffraction limit | $\lambda/(2\text{NA})$ | ~226 nm |
| 1 Airy Unit diameter | $1.22\lambda/\text{NA}$ | ~551 nm |
| Output pixel size | (from FFT scaling) | ~24 nm |
| 1 AU in pixels | $d_{\text{AU}}/\text{pixel size}$ | ~23 pixels |

### 10.3 Pinhole Configuration

| Setting | Value |
|---------|-------|
| Pinhole size | 0.2 AU |
| Pinhole diameter | ~4.6 pixels |
| Pinhole radius | ~2.3 pixels |

*Note: 0.2 AU is a small pinhole (near-ideal confocal). For stronger pinhole effects, use 1-2 AU.*

### 10.4 Saturation Parameters

From `params.py` (SaturationConfig):

| Parameter | Value | Description |
|-----------|-------|-------------|
| $\lambda$ | 532 nm | Wavelength for saturation calc |
| $\sigma_{01}$ | $10^{-20}$ m² | Absorption cross section |
| $k_0$ | $10^8$ s⁻¹ | Decay rate (~1/τ) |
| Coefficient $a$ | 3.245 | Saturation coefficient |

### 10.5 Image Processing Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| PSF size | 512×512 | Full PSF array |
| PSF crop (edgetaper) | 32×32 | For edge tapering |
| Edge crop | 32 pixels | Remove from each edge |
| Result size | 448×448 | Final cropped result |
| RL iterations | 150-200 | Deconvolution iterations |

### 10.6 Multiview Deconvolution Options

| Option | Description |
|--------|-------------|
| `weights=None` | Equal weights (default) - simple average over views |
| `weights=[w1, w2, ...]` | Scalar weights - manually weight each view |
| `otf_weighted=True` | **Recommended** - frequency-dependent OTF-based weighting |

**OTF-weighted blending** automatically:
- Gives more weight to saturated doughnut at high frequencies
- Uses Gaussian to fill doughnut OTF nulls  
- Preserves high-frequency details better than equal weighting

Example usage:
```python
# OTF-weighted multiview (recommended)
obj_recon = torch_deconv_2d(images, psfs, num_iter=200, otf_weighted=True)

# Scalar weights (manual)
obj_recon = torch_deconv_2d(images, psfs, num_iter=200, weights=[0.7, 0.3])

# Equal weights (default)
obj_recon = torch_deconv_2d(images, psfs, num_iter=200)
```

---

*End.*
