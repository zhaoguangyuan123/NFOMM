"""Utility functions for 2D deconvolution demo
Copied from main codebase to make this folder self-contained
"""

import os
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import tifffile


def central_crop(variable, tw, th):
    """Crop the last two dimensions with tw and th"""
    w = variable.shape[-2]
    h = variable.shape[-1]
    x1 = int(round((w - tw) / 2.0))
    y1 = int(round((h - th) / 2.0))
    return variable[..., x1 : x1 + tw, y1 : y1 + th]


def gen_meshgrid2D(input, input_sample_interval=None, input_type="field"):
    if input_type == "field":
        input_shape = input.shape
    elif input_type == "shape":
        input_shape = input

    xx = torch.linspace(-input_shape[-2] // 2, input_shape[-2] // 2, steps=input_shape[-2])
    yy = torch.linspace(-input_shape[-1] // 2, input_shape[-1] // 2,  steps=input_shape[-1])
    gx, gy = torch.meshgrid(xx, yy,  indexing='xy')
    if input_sample_interval != None:
        gx, gy = gx*input_sample_interval, gy*input_sample_interval
    return gx, gy


def cond_mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def convert_tensor_to_cpu(input):
    if torch.is_tensor(input):
        if input.device.type != 'cpu':
            input = input.detach().cpu().numpy()
    return input


def show(input, title="image", cut=False, cmap='gray',
         clim=None, rect_list=None, hist=False, save=False,
         save_name='picture', log_scale=False):

    if log_scale:
        if torch.is_tensor(input):
            input = torch.log(input)
        else:
            input = np.log(input)

    if torch.is_tensor(input):
        if input.device.type != 'cpu':
            print('detect the cuda')
            input = input.cpu()

    if hist:
        fig = plt.figure(figsize=(10, 5))
        ax = fig.add_subplot(121)
    else:
        fig = plt.figure(figsize=(5, 5))
        ax = fig.add_subplot(111)

    ax.title.set_text(title)
    if cut:
        img = ax.imshow(input, cmap=cmap, vmin=0, vmax=1)
    else:
        img = ax.imshow(input, cmap=cmap)
    plt.colorbar(img, ax=ax)

    if rect_list is not None:
        rect = patches.Rectangle(
            rect_list[0], rect_list[1], rect_list[2],
            linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)

    if hist:
        ax_ = fig.add_subplot(122)
        n, bins, patches = ax_.hist(input.flatten(), 100)
        ax_.title.set_text(title)

    if save is True:
        save_path = save_name + '.png'
        plt.savefig(save_path)
        print(f"Saved {title} to {save_path}")
    plt.show()


def conv2d(obj, psf, shape="same"):
    """Torch 2D Spatial convolution done via multiplication in Fourier domain
    Padding step is necessary; otherwise it will be nonlinear circular convolution.
    """
    if len(obj.shape) != 4:
        raise ValueError(f"obj must be 4D (B, C, H, W), got shape {obj.shape} with {len(obj.shape)} dims")
    if len(psf.shape) != 4:
        raise ValueError(f"psf must be 4D (B, C, H, W), got shape {psf.shape} with {len(psf.shape)} dims")
    
    im_height = obj.shape[-2]
    im_width = obj.shape[-1]
    output_size_x = obj.shape[-2] + psf.shape[-2] - 1
    output_size_y = obj.shape[-1] + psf.shape[-1] - 1

    p2d_psf = (0, output_size_y - psf.shape[-1], 0, output_size_x - psf.shape[-2])
    p2d_obj = (0, output_size_y - obj.shape[-1], 0, output_size_x - obj.shape[-2])
    psf_padded = F.pad(psf, p2d_psf, mode="constant", value=0)
    obj_padded = F.pad(obj, p2d_obj, mode="constant", value=0)

    obj_fft = torch.fft.fft2(obj_padded)
    otf_padded = torch.fft.fft2(psf_padded)

    frequency_conv = obj_fft * otf_padded
    convolved = torch.fft.ifft2(frequency_conv)
    convolved = torch.abs(convolved)
    
    if shape=="same":
        convolved = central_crop(convolved, im_height, im_width)
        if len(convolved.shape) != 4:
            raise ValueError(f"central_crop returned tensor with shape {convolved.shape}, expected 4D (B, C, H, W)")
    else:
        raise NotImplementedError

    return convolved


def gen_vortex_phase(wave_shape, input_sample_interval, device):
    FX, FY = gen_meshgrid2D(wave_shape, input_sample_interval, input_type="shape")
    theta = torch.atan2(FX, FY).to(device)
    return theta.to(device)


def tensor_to_numpy_2d(tensor):
    """Convert tensor to numpy array, handling BCHW format"""
    if torch.is_tensor(tensor):
        tensor = tensor.cpu().detach()
        if tensor.dim() == 4:
            tensor = tensor[0, 0]
        elif tensor.dim() == 3:
            tensor = tensor[0]
    return tensor.numpy()


def normalize_for_tiff(arr):
    """Normalize array to 0-65535 range for 16-bit TIFF"""
    arr = arr.astype(np.float32)
    arr_min = arr.min()
    arr_max = arr.max()
    if arr_max > arr_min:
        arr = (arr - arr_min) / (arr_max - arr_min)
    else:
        arr = np.zeros_like(arr)
    return (arr * 65535).astype(np.uint16)


def save_tiff(arr, filepath):
    """Save array as TIFF file"""
    tifffile.imwrite(filepath, arr)


def create_2d_object(size, object_config, device):
    """Create a 2D test object"""
    size = object_config.size
    obj = torch.zeros(1, 1, size, size).to(device)
    center = size // 2
    
    y1_s, y1_e, x1_s, x1_e = object_config.center_offset_1
    obj[0, 0, center+y1_s:center+y1_e, center+x1_s:center+x1_e] = 1.0
    
    y2_s, y2_e, x2_s, x2_e = object_config.center_offset_2
    obj[0, 0, center+y2_s:center+y2_e, center+x2_s:center+x2_e] = 1.0
    
    y3_s, y3_e, x3_s, x3_e = object_config.square_1
    obj[0, 0, center+y3_s:center+y3_e+10, center+x3_s:center+x3_e+10] = 1.0
    
    y4_s, y4_e, x4_s, x4_e = object_config.square_2
    obj[0, 0, center+y4_s:center+y4_e, center+x4_s:center+x4_e] = 1.0
    
    return obj
    

def normalize_psf(psf):
    """Normalize PSF to max norm along batch dimension"""
    max_vals = torch.max(psf.reshape(psf.shape[0], -1), dim=1, keepdim=True)[0]
    max_vals = max_vals.reshape(psf.shape[0], 1, 1, 1)
    return psf / (max_vals + 1e-10)


def create_pinhole_disk(size, radius_pixels, device=None):
    """
    Create a circular pinhole disk mask.
    
    Args:
        size: (H, W) or int for square
        radius_pixels: radius of pinhole in pixels
        device: torch device
    
    Returns:
        Tensor (1, 1, H, W) with 1 inside disk, 0 outside
    """
    if isinstance(size, int):
        H, W = size, size
    else:
        H, W = size
    
    y = torch.arange(H, device=device) - H // 2
    x = torch.arange(W, device=device) - W // 2
    yy, xx = torch.meshgrid(y, x, indexing='ij')
    r = torch.sqrt(xx.float()**2 + yy.float()**2)
    
    disk = (r <= radius_pixels).float()
    # Normalize so integral = 1
    disk = disk / (disk.sum() + 1e-10)
    
    return disk.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)


def compute_airy_unit_pixels(wavelength_mm, NA, pixel_size_mm, magnification=1.0):
    """
    Compute 1 Airy Unit in pixels.
    
    1 AU diameter = 1.22 * λ * M / NA
    
    Args:
        wavelength_mm: wavelength in mm
        NA: numerical aperture
        pixel_size_mm: pixel size at sample plane in mm
        magnification: magnification to pinhole plane (default 1 for sample plane)
    
    Returns:
        Airy unit diameter in pixels
    """
    au_diameter_mm = 1.22 * wavelength_mm * magnification / NA
    au_diameter_pixels = au_diameter_mm / pixel_size_mm
    return au_diameter_pixels


def _circular_pad(u, pad_scale):
    """Circular padding the 2D image to target scale"""
    w, h = u.shape[-2], u.shape[-1]
    w_padded, h_padded = int(w * pad_scale), int(h * pad_scale)
    ww = int(round((w_padded - w) / 2.0))
    hh = int(round((h_padded - h) / 2.0))
    p2d = (hh, hh, ww, ww)
    u_padded = F.pad(u, p2d, mode="constant", value=0)
    return u_padded


def _autocorrelation2d(h):
    """Compute autocorrelation of a signal along the last two dimensions"""
    Fhsq = torch.abs(torch.fft.fft2(h))
    a = torch.abs(torch.fft.ifft2(Fhsq))
    return a / (a.max() + 1e-10)


def _compute_weighting_for_tapering(h, H, W):
    """Compute edge tapering weight from PSF autocorrelation"""
    # Pad PSF to image size
    h = _circular_pad(h, H / h.shape[-2])
    # Project PSF onto each axis (keeping singleton dim for 2D autocorr)
    h_proj0 = h.sum(dim=-2, keepdim=True)
    h_proj1 = h.sum(dim=-1, keepdim=True)
    # Compute 2D autocorrelation on projections
    autocorr_h_proj0 = _autocorrelation2d(h_proj0)
    autocorr_h_proj1 = _autocorrelation2d(h_proj1)
    return (1 - autocorr_h_proj0) * (1 - autocorr_h_proj1)


def edgetaper2d(img, psf, num_iterations=1):
    """
    Edge-taper image to reduce boundary ringing in deconvolution.
    
    Blends original image with blurred version at edges using PSF autocorrelation.
    Reference: https://github.com/AndreiDavydov/Poisson_Denoiser
    
    Args:
        img: (B, C, H, W) input image
        psf: (B, C, H, W) point spread function
        num_iterations: number of tapering iterations (more = smoother edges)
    
    Returns:
        Edge-tapered image (B, C, H, W)
    """
    H, W = img.shape[-2], img.shape[-1]
    alpha = _compute_weighting_for_tapering(psf, H, W)
    
    result = img
    for _ in range(num_iterations):
        blurred_img = conv2d(result, psf)
        result = alpha * result + (1 - alpha) * blurred_img
    
    return result

def analyze_saturation_effect(psf_path, save_path, sat_levels=None):
    """
    Analyze how different saturation levels affect emission PSF and OTF.
    
    Args:
        psf_path: path to doughnut PSF tensor (.pt file)
        save_path: path to save the comparison plot
        sat_levels: list of saturation levels to compare (default: [0, 1, 3, 10, 50, 100])
    """
    if sat_levels is None:
        sat_levels = [0, 1, 3, 10, 50, 100]
    
    # Load doughnut PSF
    psf = torch.load(psf_path)[0, 0].cpu().numpy()
    H, W = psf.shape
    center = H // 2
    
    # Normalize
    psf_norm = psf / psf.max()
    
    # Saturation function
    def saturate(I, level):
        if level <= 0:
            return I
        I_sat = 1.0 / (1.0 + level)
        return I / (I_sat + I)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot OTF for different saturation levels
    for sat_level in sat_levels:
        psf_sat = saturate(psf_norm, sat_level)
        otf = np.abs(np.fft.fftshift(np.fft.fft2(psf_sat)))
        otf_norm = otf / otf.max()
        otf_db = 20 * np.log10(otf_norm + 1e-10)
        otf_db = np.clip(otf_db, -80, 0)
        
        line = otf_db[center, center:]
        freq = np.arange(len(line))
        axes[0].plot(freq[:150], line[:150], label=f'sat={sat_level}')
    
    axes[0].set_xlabel('Spatial Frequency')
    axes[0].set_ylabel('OTF (dB)')
    axes[0].set_title('OTF of Saturated Doughnut (emission PSF only)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim(-80, 5)
    
    # Plot PSF profiles
    for sat_level in sat_levels:
        psf_sat = saturate(psf_norm, sat_level)
        line = psf_sat[center, center:center+30]
        line_norm = line / line.max()
        axes[1].plot(range(30), line_norm, label=f'sat={sat_level}')
    
    axes[1].set_xlabel('Distance from center (pixels)')
    axes[1].set_ylabel('Normalized intensity')
    axes[1].set_title('Emission PSF profile (half-line from center)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Saved saturation comparison: {save_path}")
    plt.close(fig)


def analyze_otf(psf_dict, save_dir, freq_limit=100):
    """
    Compute OTF from PSFs and plot line profiles through center.
    
    Args:
        psf_dict: dict with PSF tensors, keys should include:
            'psf_gaussian', 'psf_doughnut', 'psf_eff_gaussian', 'psf_eff_saturated'
        save_dir: directory to save OTF plots
        freq_limit: frequency range for x-axis limits
    """
    cond_mkdir(save_dir)
    
    # Extract PSFs to numpy
    psfs = {}
    expected_keys = ['psf_gaussian', 'psf_doughnut', 'psf_eff_gaussian', 'psf_eff_saturated']
    for key in expected_keys:
        if key in psf_dict:
            psfs[key] = psf_dict[key][0, 0].cpu().numpy()
    
    print(f"  Found {len(psfs)}/{len(expected_keys)} PSFs for OTF analysis")
    
    # Compute OTF = |FFT(PSF)|, shifted and normalized
    otfs = {}
    for key, psf_np in psfs.items():
        otf = np.abs(np.fft.fftshift(np.fft.fft2(psf_np)))
        otfs[key] = otf / (otf.max() + 1e-10)
    
    if not otfs:
        print("Warning: No PSFs found for OTF analysis")
        return
    
    # Extract center line (horizontal through center)
    size = list(otfs.values())[0].shape[0]
    center = size // 2
    freq_axis = np.arange(size) - center
    
    lines = {key: otf[center, :] for key, otf in otfs.items()}
    
    # Plot 1: Effective PSFs comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    if 'psf_eff_gaussian' in lines:
        ax.plot(freq_axis, lines['psf_eff_gaussian'], 'b-', linewidth=2, label='Gaussian confocal (effective)')
    if 'psf_eff_saturated' in lines:
        ax.plot(freq_axis, lines['psf_eff_saturated'], 'r-', linewidth=2, label='Doughnut saturated (effective)')
    ax.set_xlabel('Spatial Frequency', fontsize=12)
    ax.set_ylabel('OTF Magnitude (normalized)', fontsize=12)
    ax.set_title('OTF Line Profile - Effective PSFs', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-freq_limit, freq_limit)
    fig.tight_layout()
    save_path = os.path.join(save_dir, "otf_effective_psfs.png")
    fig.savefig(save_path, dpi=150)
    print(f"  Saved {save_path}")
    plt.close(fig)
    
    # Plot 2: All PSFs comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    if 'psf_gaussian' in lines:
        ax.plot(freq_axis, lines['psf_gaussian'], 'b--', linewidth=1.5, label='Gaussian (illumination)')
    if 'psf_doughnut' in lines:
        ax.plot(freq_axis, lines['psf_doughnut'], 'r--', linewidth=1.5, label='Doughnut (illumination)')
    if 'psf_eff_gaussian' in lines:
        ax.plot(freq_axis, lines['psf_eff_gaussian'], 'b-', linewidth=2, label='Gaussian confocal (effective)')
    if 'psf_eff_saturated' in lines:
        ax.plot(freq_axis, lines['psf_eff_saturated'], 'r-', linewidth=2, label='Doughnut saturated (effective)')
    ax.set_xlabel('Spatial Frequency', fontsize=12)
    ax.set_ylabel('OTF Magnitude (normalized)', fontsize=12)
    ax.set_title('OTF Line Profile Comparison', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-freq_limit, freq_limit)
    fig.tight_layout()
    save_path = os.path.join(save_dir, "otf_all_psfs.png")
    fig.savefig(save_path, dpi=150)
    print(f"  Saved {save_path}")
    plt.close(fig)
    
    # Plot 3: Positive frequencies only
    fig, ax = plt.subplots(figsize=(10, 6))
    half = len(freq_axis) // 2
    if 'psf_eff_gaussian' in lines:
        ax.plot(freq_axis[half:], lines['psf_eff_gaussian'][half:], 'b-', linewidth=2, label='Gaussian confocal (effective)')
    if 'psf_eff_saturated' in lines:
        ax.plot(freq_axis[half:], lines['psf_eff_saturated'][half:], 'r-', linewidth=2, label='Doughnut saturated (effective)')
    ax.set_xlabel('Spatial Frequency', fontsize=12)
    ax.set_ylabel('OTF Magnitude (normalized)', fontsize=12)
    ax.set_title('OTF Line Profile - Positive Frequencies', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, freq_limit)
    fig.tight_layout()
    save_path = os.path.join(save_dir, "otf_effective_positive.png")
    fig.savefig(save_path, dpi=150)
    print(f"  Saved {save_path}")
    plt.close(fig)
    
    # Plot 4: LOG SCALE - to see high frequency tails
    fig, ax = plt.subplots(figsize=(10, 6))
    for key in ['psf_gaussian', 'psf_doughnut', 'psf_eff_gaussian', 'psf_eff_saturated']:
        if key in lines:
            data_log = 20 * np.log10(lines[key] + 1e-10)
            data_log = np.clip(data_log, -80, 0)  # clip noise floor
            style = '--' if key in ['psf_gaussian', 'psf_doughnut'] else '-'
            color = 'r' if 'doughnut' in key or 'saturated' in key else 'b'
            label = key.replace('psf_', '').replace('_', ' ')
            ax.plot(freq_axis, data_log, color=color, linestyle=style, linewidth=2, label=label)
    ax.set_xlabel('Spatial Frequency', fontsize=12)
    ax.set_ylabel('OTF Magnitude (dB)', fontsize=12)
    ax.set_title('OTF Line Profile - LOG SCALE', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-freq_limit, freq_limit)
    ax.set_ylim(-80, 5)
    fig.tight_layout()
    save_path = os.path.join(save_dir, "otf_log_scale.png")
    fig.savefig(save_path, dpi=150)
    print(f"  Saved {save_path}")
    plt.close(fig)
    
    print(f"OTF plots saved to {save_dir}")
    
    # Plot 5: PSF spatial profiles (1D line through center)
    fig, ax = plt.subplots(figsize=(10, 6))
    size = list(psfs.values())[0].shape[0]
    center = size // 2
    spatial_axis = np.arange(size) - center
    
    for key, psf_np in psfs.items():
        line = psf_np[center, :]
        line_norm = line / (line.max() + 1e-10)
        style = '--' if key in ['psf_gaussian', 'psf_doughnut'] else '-'
        color = 'r' if 'doughnut' in key or 'saturated' in key else 'b'
        label = key.replace('psf_', '').replace('_', ' ')
        ax.plot(spatial_axis, line_norm, color=color, linestyle=style, linewidth=2, label=label)
    
    ax.set_xlabel('Position (pixels from center)', fontsize=12)
    ax.set_ylabel('PSF Intensity (normalized)', fontsize=12)
    ax.set_title('PSF Spatial Profile - Line Through Center', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-50, 50)
    fig.tight_layout()
    save_path = os.path.join(save_dir, "psf_profiles.png")
    fig.savefig(save_path, dpi=150)
    print(f"  Saved {save_path}")
    plt.close(fig)
