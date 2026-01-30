"""
Fourier Ring Correlation (FRC) for resolution analysis.

FRC measures the normalized cross-correlation between two images 
in Fourier space as a function of spatial frequency (ring radius).

Usage:
    python -m nfomm.core.frc
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from typing import Tuple, Dict, Optional


def compute_frc(img1: np.ndarray, img2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute Fourier Ring Correlation between two 2D images.
    
    Args:
        img1: First image (H, W)
        img2: Second image (H, W), same shape as img1
        
    Returns:
        freq: Normalized spatial frequencies (0 to 0.5)
        frc: FRC values at each frequency
    """
    assert img1.shape == img2.shape, f"Shape mismatch: {img1.shape} vs {img2.shape}"
    assert img1.ndim == 2, f"Expected 2D images, got {img1.ndim}D"
    
    H, W = img1.shape
    
    # FFT of both images
    fft1 = np.fft.fftshift(np.fft.fft2(img1))
    fft2 = np.fft.fftshift(np.fft.fft2(img2))
    
    # Create frequency coordinate grid
    cy, cx = H // 2, W // 2
    y, x = np.ogrid[:H, :W]
    radius = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    
    # Maximum radius (Nyquist)
    max_radius = min(cx, cy)
    
    # Compute FRC for each ring
    n_rings = max_radius
    freq = np.zeros(n_rings)
    frc = np.zeros(n_rings)
    
    for r in range(n_rings):
        # Ring mask (pixels at distance r to r+1)
        ring_mask = (radius >= r) & (radius < r + 1)
        
        if ring_mask.sum() == 0:
            continue
        
        # Extract ring values
        f1_ring = fft1[ring_mask]
        f2_ring = fft2[ring_mask]
        
        # FRC = Re(sum(F1 * conj(F2))) / sqrt(sum(|F1|^2) * sum(|F2|^2))
        numerator = np.real(np.sum(f1_ring * np.conj(f2_ring)))
        denominator = np.sqrt(np.sum(np.abs(f1_ring) ** 2) * np.sum(np.abs(f2_ring) ** 2))
        
        freq[r] = r / (2 * max_radius)  # Normalized frequency (0 to 0.5)
        frc[r] = numerator / denominator if denominator > 0 else 0
    
    return freq, frc


def compute_frc_threshold(n_pixels_per_ring: np.ndarray, threshold_type: str = "half_bit") -> np.ndarray:
    """
    Compute FRC threshold curve.
    
    Args:
        n_pixels_per_ring: Number of pixels in each ring
        threshold_type: "half_bit" (default), "one_bit", or "fixed" (0.143)
        
    Returns:
        threshold: Threshold values for each ring
    """
    if threshold_type == "fixed":
        return np.ones_like(n_pixels_per_ring, dtype=float) * 0.143
    
    # Information-based thresholds
    # half-bit: (0.5 + 2.4142/sqrt(n)) / (1.5 + 1.4142/sqrt(n))
    # one-bit: (0.5 + 1/sqrt(n)) / (1 + 1/sqrt(n))
    n = np.maximum(n_pixels_per_ring, 1)  # Avoid division by zero
    
    if threshold_type == "half_bit":
        threshold = (0.5 + 2.4142 / np.sqrt(n)) / (1.5 + 1.4142 / np.sqrt(n))
    elif threshold_type == "one_bit":
        threshold = (0.5 + 1 / np.sqrt(n)) / (1 + 1 / np.sqrt(n))
    else:
        raise ValueError(f"Unknown threshold type: {threshold_type}")
    
    return threshold


def find_resolution(freq: np.ndarray, frc: np.ndarray, threshold: float = 0.143) -> float:
    """
    Find resolution where FRC crosses threshold.
    
    Args:
        freq: Normalized frequencies
        frc: FRC values
        threshold: Threshold value (default 0.143)
        
    Returns:
        Resolution in normalized frequency units (lower = better)
    """
    crossings = np.where(frc < threshold)[0]
    
    if len(crossings) == 0:
        return freq[-1]  # Never crosses, return max frequency
    
    return freq[crossings[0]]


def plot_frc(frc_results: Dict[str, Tuple[np.ndarray, np.ndarray]], 
             save_path: Optional[str] = None,
             pixel_size: Optional[float] = None,
             title: str = "Fourier Ring Correlation") -> None:
    """
    Plot FRC curves for multiple image pairs.
    
    Args:
        frc_results: Dict mapping name -> (freq, frc) tuples
        save_path: Path to save figure (optional)
        pixel_size: Physical pixel size for axis labels (optional)
        title: Plot title
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(frc_results)))
    
    for (name, (freq, frc)), color in zip(frc_results.items(), colors):
        ax.plot(freq, frc, label=name, color=color, linewidth=2)
        
        # Find and mark resolution
        res_freq = find_resolution(freq, frc)
        ax.axvline(res_freq, color=color, linestyle='--', alpha=0.5)
    
    # Threshold line
    ax.axhline(0.143, color='gray', linestyle=':', label='1/7 threshold')
    ax.axhline(0.5, color='gray', linestyle='-.', alpha=0.5, label='0.5 threshold')
    
    ax.set_xlabel('Spatial Frequency (normalized)' if pixel_size is None 
                  else f'Spatial Frequency (1/{pixel_size:.2f} units)')
    ax.set_ylabel('FRC')
    ax.set_title(title)
    ax.set_xlim(0, 0.5)
    ax.set_ylim(-0.1, 1.1)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved FRC plot to {save_path}")
    
    plt.close(fig)


def tensor_to_numpy_2d(tensor: torch.Tensor) -> np.ndarray:
    """Convert 4D tensor (B,C,H,W) to 2D numpy array."""
    if tensor.ndim == 4:
        return tensor[0, 0].cpu().numpy()
    elif tensor.ndim == 2:
        return tensor.cpu().numpy()
    else:
        raise ValueError(f"Expected 2D or 4D tensor, got {tensor.ndim}D")


def analyze_frc(ground_truth: torch.Tensor, 
                reconstructions: Dict[str, torch.Tensor],
                save_dir: str) -> Dict[str, float]:
    """
    Analyze FRC between ground truth and multiple reconstructions.
    
    Args:
        ground_truth: Ground truth image tensor (B,C,H,W) or (H,W)
        reconstructions: Dict mapping name -> reconstruction tensor
        save_dir: Directory to save results
        
    Returns:
        Dict mapping name -> resolution (normalized frequency where FRC < 0.143)
    """
    import os
    os.makedirs(save_dir, exist_ok=True)
    
    gt = tensor_to_numpy_2d(ground_truth)
    
    frc_results = {}
    resolutions = {}
    
    for name, recon in reconstructions.items():
        recon_np = tensor_to_numpy_2d(recon)
        
        # Ensure same shape (crop to smaller)
        min_h = min(gt.shape[0], recon_np.shape[0])
        min_w = min(gt.shape[1], recon_np.shape[1])
        
        gt_crop = gt[:min_h, :min_w]
        recon_crop = recon_np[:min_h, :min_w]
        
        freq, frc = compute_frc(gt_crop, recon_crop)
        frc_results[name] = (freq, frc)
        resolutions[name] = find_resolution(freq, frc)
        
        print(f"{name}: Resolution = {resolutions[name]:.4f} (normalized freq)")
    
    # Plot
    plot_frc(frc_results, save_path=os.path.join(save_dir, "frc_analysis.png"),
             title="FRC: Reconstructions vs Ground Truth")
    
    # Save numerical results
    results_path = os.path.join(save_dir, "frc_results.txt")
    with open(results_path, 'w') as f:
        f.write("FRC Resolution Analysis\n")
        f.write("=" * 50 + "\n")
        f.write("Resolution = frequency where FRC < 0.143\n")
        f.write("Lower frequency = worse resolution\n\n")
        for name, res in resolutions.items():
            f.write(f"{name}: {res:.4f}\n")
    print(f"Saved FRC results to {results_path}")
    
    return resolutions


if __name__ == "__main__":
    import os
    
    # Test with dummy images
    print("Testing FRC computation...")
    
    # Create test images
    size = 256
    x = np.linspace(-1, 1, size)
    X, Y = np.meshgrid(x, x)
    
    # Ground truth: high frequency pattern
    gt = np.sin(20 * np.pi * X) * np.sin(20 * np.pi * Y)
    gt = (gt + 1) / 2  # Normalize to [0, 1]
    
    # Blurred version
    from scipy.ndimage import gaussian_filter
    blurred = gaussian_filter(gt, sigma=2)
    
    # Noisy version  
    noisy = gt + 0.1 * np.random.randn(*gt.shape)
    
    # Compute FRC
    freq_blur, frc_blur = compute_frc(gt, blurred)
    freq_noisy, frc_noisy = compute_frc(gt, noisy)
    
    print(f"Blurred resolution: {find_resolution(freq_blur, frc_blur):.4f}")
    print(f"Noisy resolution: {find_resolution(freq_noisy, frc_noisy):.4f}")
    
    # Plot
    frc_results = {
        "Blurred (sigma=2)": (freq_blur, frc_blur),
        "Noisy (std=0.1)": (freq_noisy, frc_noisy)
    }
    
    os.makedirs("nfomm/results/frc_test", exist_ok=True)
    plot_frc(frc_results, save_path="nfomm/results/frc_test/frc_test.png",
             title="FRC Test: Blurred vs Noisy")
    
    print("\nTest completed. Check nfomm/results/frc_test/frc_test.png")
