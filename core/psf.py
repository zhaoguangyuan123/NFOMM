import torch
import math
from .utils import gen_meshgrid2D, central_crop, gen_vortex_phase
from .params import psf_config, device_config, PSFConfig
from torch.fft import fft2, fftshift, ifftshift


class EasyPSF2DST(torch.nn.Module):
    """
    2D PSF generator from phase mask, similar to EasyPSF1DST
    
    The code was adapted from Shangting You's Matlab Code for the vectorial/scalar diffraction psf simulation.
    reference: 
    """
    def __init__(self, psf_config=psf_config, device_config=device_config, base_N=None):
        super().__init__()
        self.psf_config = psf_config
        self.device_config = device_config
        self.wave_length_mm = psf_config.wave_length_mm
        self.N = psf_config.N
        
        # Use base_N for aperture definition (like MATLAB uses input size, not FFT size)
        # This keeps the physical aperture size constant regardless of FFT size
        if base_N is None:
            base_N = psf_config.N
        
        # Generate physical coordinates in mm (like MATLAB: x=linspace(-rangeX/2,rangeX/2,Nx))
        # input_pixel_size_mm = range_mm / base_N
        input_pixel_size_mm = psf_config.range_mm / base_N
        FX, FY = gen_meshgrid2D((base_N, base_N), 
                                input_sample_interval=input_pixel_size_mm, 
                                input_type="shape")
        r = torch.sqrt(FX**2 + FY**2)
        
        # Aperture in physical units (matches MATLAB: A0=1.0*(R0<D0/2))
        # D0/2 is the physical radius in mm
        aperture_radius_mm = psf_config.D0_mm / 2.0
        self.aperture_base = (r < aperture_radius_mm).to(device_config.device)
        
        # If N > base_N, we need to pad/expand the aperture
        if psf_config.N > base_N:
            # Pad aperture to match FFT size
            pad_size = (psf_config.N - base_N) // 2
            self.aperture = torch.nn.functional.pad(
                self.aperture_base[None, None, :, :], 
                (pad_size, pad_size, pad_size, pad_size), 
                mode='constant', 
                value=0
            )[0, 0, :, :]
        else:
            self.aperture = self.aperture_base
        
    def forward(self, in_phase):
        # Ensure in_phase is 4D (B, C, H, W) - this is the root cause fix
        if len(in_phase.shape) != 4:
            raise ValueError(f"in_phase must be 4D (B, C, H, W), got shape {in_phase.shape}. "
                           f"This indicates a bug in create_doughnut_psf or create_gaussian_psf.")
        
        E1x = torch.exp(1j * in_phase)
        E1x = E1x * self.aperture[None, None, :, :]
        
        # FFT is done at the size of E1x, which is in_phase size (should be N_fourier x N_fourier for better sampling)
        fft_size = E1x.shape[-1]
        U2x = fftshift(fft2(ifftshift(E1x, dim=(-2, -1)), dim=(-2, -1)), dim=(-2, -1))
        psf_2d = torch.abs(U2x)**2
        psf_2d = psf_2d / torch.sum(psf_2d, dim=(-2, -1), keepdim=True)
        
        # Ensure output is 4D (B, C, H, W)
        if len(psf_2d.shape) != 4:
            raise ValueError(f"PSF generation returned {len(psf_2d.shape)}D tensor with shape {psf_2d.shape}, expected 4D (B, C, H, W)")
        
        return psf_2d


def create_batch_psf_2d_from_phase(psf_config=psf_config, device_config=device_config,
                                   batch_size=None, N=None, psf_size=None, 
                                   psf_same=None, zero_phase=None):
    """Create batch of 2D PSFs from phase masks, similar to create_batch_psf for 1D"""
    if batch_size is None:
        batch_size = psf_config.batch_size
    if N is None:
        N = psf_config.N
    if psf_size is None:
        psf_size = psf_config.psf_size
    if psf_same is None:
        psf_same = psf_config.psf_same
    if zero_phase is None:
        zero_phase = psf_config.zero_phase
    
    if psf_same:
        if zero_phase:
            in_phase = torch.zeros(1, 1, N, N).repeat(batch_size, 1, 1, 1).to(device_config.device)
        else:
            in_phase = (torch.rand(1, 1, N, N).repeat(batch_size, 1, 1, 1) * 2 * math.pi).to(device_config.device)
    else:
        in_phase = (torch.rand(batch_size, 1, N, N) * 2 * math.pi).to(device_config.device)
    
    get_psf_2d = EasyPSF2DST(psf_config=psf_config, device_config=device_config).to(device_config.device)
    psf = get_psf_2d(in_phase)
    
    if psf_size < N:
        psf = central_crop(psf, psf_size, psf_size)
    
    return psf


def create_doughnut_psf(psf_config=psf_config, device_config=device_config, N=None, psf_size=None, N_fourier=None):
    """Create doughnut PSF using vortex phase mask with larger Fourier domain for better sampling"""
    if N is None:
        N = psf_config.N
    if psf_size is None:
        psf_size = psf_config.psf_size
    if N_fourier is None:
        N_fourier = psf_config.N_fourier
    
    # Create phase mask at base input size (like MATLAB: 100x100 input)
    # Then pad to N_fourier for FFT (like MATLAB: FFT at 3000x3000)
    base_input_size = psf_config.N
    # Use physical pixel size: range_mm / N (like MATLAB: x=linspace(-rangeX/2,rangeX/2,Nx))
    input_pixel_size_mm = psf_config.range_mm / base_input_size
    vortex_phase = gen_vortex_phase((base_input_size, base_input_size), 
                                    input_sample_interval=input_pixel_size_mm, 
                                    device=device_config.device)
    in_phase_base = vortex_phase[None, None, :, :].to(device_config.device)
    
    # Pad phase mask to FFT size (zero-padding like MATLAB)
    pad_size = (N_fourier - base_input_size) // 2
    in_phase = torch.nn.functional.pad(in_phase_base, (pad_size, pad_size, pad_size, pad_size), mode='constant', value=0)
    
    # Create PSF generator with larger N for FFT
    # Use base input size for aperture definition, but FFT at N_fourier
    # This matches MATLAB: aperture at input size (100x100), FFT at larger size (3000x3000)
    temp_config = PSFConfig(N=N_fourier, psf_size=psf_size,
                           range_mm=psf_config.range_mm,
                           D0_mm=psf_config.D0_mm,
                           f_mm=psf_config.f_mm,
                           n=psf_config.n,
                           wave_length_mm=psf_config.wave_length_mm,
                           z0_mm=psf_config.z0_mm)
    get_psf_2d = EasyPSF2DST(psf_config=temp_config, device_config=device_config, base_N=base_input_size).to(device_config.device)
    print(f"NA: {psf_config.NA:.4f}, Input pixel size: {psf_config.input_pixel_size_mm*1000:.4f} um, Output pixel size: {psf_config.output_pixel_size_mm*1000:.4f} um")
    print(f"FFT size: {in_phase.shape[-1]}x{in_phase.shape[-2]} (N_fourier={N_fourier}), aperture base size: {base_input_size}")
    print(f"DEBUG: in_phase shape before forward: {in_phase.shape}, expected [1, 1, {N_fourier}, {N_fourier}]")
    psf = get_psf_2d(in_phase)
    print(f"DEBUG: psf shape after forward: {psf.shape}, expected [1, 1, {N_fourier}, {N_fourier}]")
    aperture = get_psf_2d.aperture
    
    # Crop PSF to final size, but keep phase mask and aperture at full Fourier domain size
    psf = central_crop(psf, psf_size, psf_size)
    phase_mask_full = in_phase  # Keep full size for better sampling visualization
    aperture_full = aperture[None, None, :, :]  # Keep full size
    
    # Also provide cropped versions for consistency
    phase_mask = central_crop(in_phase, psf_size, psf_size)
    
    return psf, phase_mask, phase_mask_full, aperture_full


def create_gaussian_psf(psf_config=psf_config, device_config=device_config, N=None, psf_size=None, N_fourier=None):
    """Create Gaussian PSF using zero phase mask with larger Fourier domain for better sampling"""
    if N is None:
        N = psf_config.N
    if psf_size is None:
        psf_size = psf_config.psf_size
    if N_fourier is None:
        N_fourier = psf_config.N_fourier
    
    # Create phase mask at base input size (like MATLAB: 100x100 input)
    # Then pad to N_fourier for FFT (like MATLAB: FFT at 3000x3000)
    base_input_size = psf_config.N
    # Physical coordinates are handled by aperture, phase mask is just zeros for Gaussian
    in_phase_base = torch.zeros(1, 1, base_input_size, base_input_size).to(device_config.device)
    
    # Pad phase mask to FFT size (zero-padding like MATLAB)
    pad_size = (N_fourier - base_input_size) // 2
    in_phase = torch.nn.functional.pad(in_phase_base, (pad_size, pad_size, pad_size, pad_size), mode='constant', value=0)
    
    # Create PSF generator with larger N for FFT
    # Use base input size for aperture definition, but FFT at N_fourier
    # This matches MATLAB: aperture at input size (100x100), FFT at larger size (3000x3000)
    # Create temp config with physical parameters for FFT size
    temp_config = PSFConfig(N=N_fourier, psf_size=psf_size,
                           range_mm=psf_config.range_mm,
                           D0_mm=psf_config.D0_mm,
                           f_mm=psf_config.f_mm,
                           n=psf_config.n,
                           wave_length_mm=psf_config.wave_length_mm,
                           z0_mm=psf_config.z0_mm)
    get_psf_2d = EasyPSF2DST(psf_config=temp_config, device_config=device_config, base_N=base_input_size).to(device_config.device)
    print(f"FFT size: {in_phase.shape[-1]}x{in_phase.shape[-2]} (N_fourier={N_fourier}), aperture base size: {base_input_size}")
    print(f"NA: {psf_config.NA:.4f}, Input pixel: {psf_config.input_pixel_size_mm*1000:.4f} um, Output pixel: {psf_config.output_pixel_size_mm*1000:.4f} um")
    print(f"Diffraction limit: {psf_config.diffraction_limit_mm*1000:.4f} um, Sampling ratio: {psf_config.sampling_ratio:.2f}x")
    psf = get_psf_2d(in_phase)
    aperture = get_psf_2d.aperture
    
    # Crop PSF to final size, but keep phase mask and aperture at full Fourier domain size
    psf = central_crop(psf, psf_size, psf_size)
    phase_mask_full = in_phase  # Keep full size for better sampling visualization
    aperture_full = aperture[None, None, :, :]  # Keep full size
    
    # Also provide cropped versions for consistency
    phase_mask = central_crop(in_phase, psf_size, psf_size)
    
    return psf, phase_mask, phase_mask_full, aperture_full