"""
NFOMM Saturation Models

Two saturation models are available:
1. Simple: g(I) = I / (1 + I * saturation_level)
   - Works with normalized PSFs
   - Easy to tune with saturation_level parameter
   
2. Physics: S1 = k01 / (coeff * k_exc + k0)
   - Full physics-based model from NFOMM paper
   - Requires physical constants (wavelength, cross-section, etc.)
"""

import torch
from typing import Optional
from .params import saturation_config


def _simple_saturation(psf: torch.Tensor, saturation_level: float) -> torch.Tensor:
    """
    Simple saturation model: g(I) = I / (I_sat + I)
    
    This models fluorescence saturation:
    - Low I (I << I_sat): g(I) ≈ I/I_sat (linear)
    - High I (I >> I_sat): g(I) → 1 (saturates to 1)
    
    saturation_level = 1/I_sat, so:
    - saturation_level = 0: linear (no saturation, g=I)
    - saturation_level > 0: stronger saturation
    - saturation_level = 10: I_sat = 0.1, strong saturation
    
    For doughnut PSF: ring saturates to flat plateau (~1), hole stays at 0,
    creating sharp edge that yields high-frequency OTF content.
    
    Args:
        psf: illumination PSF (B, C, H, W)
        saturation_level: higher = stronger saturation (I_sat = 1/(1+saturation_level))
    
    Returns:
        psf_emi: emission PSF (B, 1, H, W)
    """
    intensity = psf[:, :1] if psf.shape[1] > 1 else psf
    intensity_norm = intensity / (intensity.max() + 1e-10)
    
    if saturation_level <= 0:
        return intensity_norm
    
    # I_sat = 1 / (1 + saturation_level)
    # saturation_level=0 → I_sat=1 (weak), saturation_level=9 → I_sat=0.1 (strong)
    I_sat = 1.0 / (1.0 + saturation_level)
    
    # g(I) = I / (I_sat + I), asymptotes to 1 as I → ∞
    psf_emi = intensity_norm / (I_sat + intensity_norm)
    return psf_emi


def _physics_saturation(
    psf: torch.Tensor,
    wavelength_m: float,
    sigma01: float,
    k0: float,
    coeff: float,
    k_exc_scale: Optional[float],
    reduce_channels: str,
    kf: Optional[float],
    eps: float,
    h_planck: float,
    c_light: float,
) -> torch.Tensor:
    """
    Physics-based saturation model from NFOMM paper.
    
    S1 = k01 / (coeff * k_exc + k0)
    
    where:
        k01 = σ01 * I / (hc/λ)  (absorption rate)
        k_exc = k01 or k_exc_scale * I
        k0 = decay rate constant
        coeff = 3.245 (from paper)
    
    Args:
        psf: illumination PSF (B, C, H, W), intensity in W/m^2
        wavelength_m: excitation wavelength in meters
        sigma01: absorption cross section (m^2)
        k0: decay rate constant (s^-1)
        coeff: saturation coefficient (default 3.245)
        k_exc_scale: if set, use k_exc = k_exc_scale * I instead of k01
        reduce_channels: how to reduce channels ("sum", "mean", "first")
        kf: fluorescence decay rate (s^-1), if provided output = kf * S1
        eps: numerical stability constant
        h_planck: Planck's constant
        c_light: speed of light
    
    Returns:
        S1 or kf*S1: emission PSF (B, 1, H, W)
    """
    # Reduce channels to scalar intensity
    if reduce_channels == "sum":
        intensity = psf.sum(dim=1, keepdim=True)
    elif reduce_channels == "mean":
        intensity = psf.mean(dim=1, keepdim=True)
    elif reduce_channels == "first":
        intensity = psf[:, :1]
    else:
        raise ValueError(f"reduce_channels must be 'sum', 'mean', or 'first', got {reduce_channels}")
    
    # Photon energy: E = hc/λ
    photon_energy = (h_planck * c_light) / wavelength_m
    
    # Absorption rate: k01 = σ01 * I / E
    k01 = sigma01 * intensity / (photon_energy + eps)
    
    # Excitation rate
    if k_exc_scale is None:
        k_exc = k01
    else:
        k_exc = k_exc_scale * intensity
    
    # Excited state population: S1 = k01 / (coeff * k_exc + k0)
    denominator = coeff * k_exc + k0
    s1 = k01 / (denominator + eps)
    
    # Return S1 or emission rate
    if kf is None:
        return s1
    return kf * s1


def nfomm_saturation_emission(
    psf: torch.Tensor,
    mode: str = "simple",
    saturation_level: float = 1.0,
    wavelength_m: Optional[float] = None,
    sigma01: Optional[float] = None,
    k0: Optional[float] = None,
    coeff: Optional[float] = None,
    k_exc_scale: Optional[float] = None,
    reduce_channels: Optional[str] = None,
    kf: Optional[float] = None,
    eps: Optional[float] = None,
    config: Optional[type(saturation_config)] = None,
) -> torch.Tensor:
    """
    Compute emission PSF from illumination PSF using saturation model.
    
    Args:
        psf: illumination PSF (B, C, H, W)
        mode: "simple" or "physics"
        saturation_level: for simple mode, controls saturation strength
        
    Physics mode args (use config defaults if not provided):
        wavelength_m: excitation wavelength (m)
        sigma01: absorption cross section (m^2)
        k0: decay rate (s^-1)
        coeff: saturation coefficient
        k_exc_scale: excitation rate scale factor
        reduce_channels: channel reduction method
        kf: fluorescence decay rate (s^-1)
        eps: numerical stability
        config: SaturationConfig instance
    
    Returns:
        psf_emi: emission PSF (B, 1, H, W)
    """
    if psf.ndim != 4:
        raise ValueError(f"psf must be 4D (B,C,H,W), got shape {tuple(psf.shape)}")
    
    if mode == "simple":
        return _simple_saturation(psf, saturation_level)
    
    if mode == "physics":
        cfg = config if config is not None else saturation_config
        return _physics_saturation(
            psf=psf,
            wavelength_m=wavelength_m if wavelength_m is not None else cfg.wavelength_m,
            sigma01=sigma01 if sigma01 is not None else cfg.sigma01,
            k0=k0 if k0 is not None else cfg.k0,
            coeff=coeff if coeff is not None else cfg.coeff,
            k_exc_scale=k_exc_scale if k_exc_scale is not None else cfg.k_exc_scale,
            reduce_channels=reduce_channels if reduce_channels is not None else cfg.reduce_channels,
            kf=kf if kf is not None else cfg.kf,
            eps=eps if eps is not None else cfg.eps,
            h_planck=cfg.h_planck,
            c_light=cfg.c_light,
        )
    
    raise ValueError(f"mode must be 'simple' or 'physics', got {mode}")


if __name__ == "__main__":
    # Test simple mode
    psf = torch.rand(1, 1, 64, 64)
    out_simple = nfomm_saturation_emission(psf, mode="simple", saturation_level=10.0)
    print(f"Simple mode: input max={psf.max():.4f}, output max={out_simple.max():.4f}")
    
    # Test physics mode
    out_physics = nfomm_saturation_emission(psf, mode="physics")
    print(f"Physics mode: output shape={out_physics.shape}, max={out_physics.max():.6e}")
