"""NFOMM parameters configuration"""

from dataclasses import dataclass, field
from typing import Optional
import torch
import math


@dataclass
class DeviceConfig:
    """Device and GPU configuration"""
    gpu_id: int = 0
    gpu_name: str = field(init=False)
    device: torch.device = field(init=False)
    
    def __post_init__(self):
        self.gpu_name = f"cuda:{self.gpu_id}"
        self.device = torch.device(self.gpu_name if torch.cuda.is_available() else "cpu")


@dataclass
class ObjectConfig:
    """Toy object creation parameters"""
    size: int = 512
    center_offset_1: tuple = (-20, 20, -5, 5)  # (y_start, y_end, x_start, x_end)
    center_offset_2: tuple = (-5, 5, -20, 20)  # (y_start, y_end, x_start, x_end)
    square_1: tuple = (-30, -20, -30, -20)      # (y_start, y_end, x_start, x_end)
    square_2: tuple = (20, 30, 20, 30)          # (y_start, y_end, x_start, x_end)


@dataclass
class PSFConfig:
    """PSF generation parameters"""
    N: int = 512  # Base input size (like MATLAB N=[100,100])
    N_fourier: int = 8192  # Fourier domain size - increased 2x for finer sampling (4096*2)
    psf_size: int = 1024  # Final cropped PSF size - increased 2x to maintain FOV
    
    # Physical parameters (matching MATLAB scalFT2d_common.m)
    range_mm: float = 3.84  # Physical range of input coordinates in mm (like MATLAB range=[3.84,3.84])
    D0_mm: float = 3.32  # Entrance pupil diameter in mm
    f_mm: float = 1.8  # Focal length in mm
    n: float = 1.518  # Refractive index
    wave_length_mm: float = 632.8e-6  # Wavelength in mm (like MATLAB lambda=632.8e-6)
    z0_mm: float = 0.0  # Defocus distance in mm
    
    # Legacy parameters (kept for compatibility)
    wave_length: float = 632.8  # nm (deprecated, use wave_length_mm)
    aperture_radius: float = 0.4  # Used in create_doughnut_psf and create_gaussian_psf
    aperture_radius_default: float = 0.5  # Default in EasyPSF2DST (deprecated, use D0_mm)
    input_sample_interval: float = 1.0  # Deprecated, calculated from range_mm/N
    
    batch_size: int = 1
    psf_same: bool = False
    zero_phase: bool = False
    
    # Computed properties
    def __post_init__(self):
        """Calculate derived optical parameters"""
        # Input pixel size in mm (like MATLAB: x=linspace(-range/2,range/2,N))
        self.input_pixel_size_mm = self.range_mm / self.N
        
        # NA calculation (matching MATLAB: Theta1 = asin(R0/f), NA = n*sin(theta_max))
        # Maximum angle: theta_max = asin(D0/(2*f))
        theta_max_rad = math.asin(self.D0_mm / (2 * self.f_mm))
        self.NA = self.n * math.sin(theta_max_rad)
        
        # Output image range in mm (from MATLAB: ImgRange = N*lambda*f/range/n)
        # This is the TOTAL physical range of the output image
        output_range_mm = self.N * self.wave_length_mm * self.f_mm / (self.range_mm * self.n)
        
        # Output pixel size in mm = output_range / N_fourier (the FFT size determines sampling)
        self.output_pixel_size_mm = output_range_mm / self.N_fourier
        
        # Maximum angle (for reference)
        self.theta_max = theta_max_rad
        
        # Diffraction limit: minimum resolvable feature size = lambda/(2*NA)
        self.diffraction_limit_mm = self.wave_length_mm / (2 * self.NA)
        
        # Check sampling: should have at least 2 pixels per diffraction limit (Nyquist)
        self.sampling_ratio = self.diffraction_limit_mm / self.output_pixel_size_mm


@dataclass
class SaturationConfig:
    """Saturation model parameters"""
    wavelength_m: float = 532e-9  # meters
    sigma01: float = 1e-20  # m^2 (absorption cross section)
    k0: float = 1e8  # s^-1 (constant rate term, ~ 1/tau)
    coeff: float = 3.245  # Saturation coefficient from paper
    k_exc_scale: Optional[float] = None  # If None, use k_exc = k01
    kf: Optional[float] = None  # Fluorescence decay rate (s^-1)
    eps: float = 1e-12  # Numerical stability
    reduce_channels: str = "sum"  # Options: "sum", "mean", "first"
    return_s1: bool = True  # If True, return S1; if False, return emission rate
    
    # Physical constants (not configurable)
    h_planck: float = field(default=6.62607015e-34, init=False)  # J*s
    c_light: float = field(default=299792458.0, init=False)  # m/s


@dataclass
class ImagingConfig:
    """Imaging and demo parameters"""
    save_dir: str = "nfomm/results/"
    visualize: bool = True
    random_seed: int = 2
    regenerate_psf: bool = True  # If False, load existing PSFs if available


@dataclass
class FileConfig:
    """File saving parameters"""
    tiff_bit_depth: int = 16  # 16-bit TIFF (0-65535)


# Default instances
device_config = DeviceConfig()
object_config = ObjectConfig()
psf_config = PSFConfig()
saturation_config = SaturationConfig()
imaging_config = ImagingConfig()
file_config = FileConfig()

# Convenience accessors
device = device_config.device
