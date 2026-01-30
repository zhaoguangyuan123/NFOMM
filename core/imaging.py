
"""
nohup python -m nfomm.core.imaging > nfomm/results/nfomm_imaging.log 2>&1 & tail -f nfomm/results/nfomm_imaging.log
"""

import os
import sys

# Handle both direct execution and module execution
try:
    from .utils import conv2d, show, cond_mkdir, tensor_to_numpy_2d, create_2d_object, create_pinhole_disk, compute_airy_unit_pixels, central_crop
    from .params import device_config, object_config, psf_config, imaging_config
    from .psf import create_doughnut_psf, create_gaussian_psf
    from .saturation import nfomm_saturation_emission
except ImportError:
    # Add parent directory to path for direct execution
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    from nfomm.core.utils import conv2d, show, cond_mkdir, tensor_to_numpy_2d, create_2d_object, create_pinhole_disk, compute_airy_unit_pixels, central_crop
    from nfomm.core.params import device_config, object_config, psf_config, imaging_config
    from nfomm.core.psf import create_doughnut_psf, create_gaussian_psf
    from nfomm.core.saturation import nfomm_saturation_emission

import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2


class Imaging:
    """Imaging class for 2D object creation, blurring, and saving"""
    
    def __init__(self, device_config=device_config, object_config=object_config, 
                 psf_config=psf_config, imaging_config=imaging_config, saturation_level=1.0):
        """
        Args:
            saturation_level: Scaling factor for PSF intensity before saturation.
                Higher value = stronger saturation effect.
                - 0.0: No saturation (linear, h_emi = h_ill)
                - 1.0: Default saturation
                - >1.0: Stronger saturation (simulate higher excitation power)
        """
        self.device_config = device_config
        self.object_config = object_config
        self.psf_config = psf_config
        self.imaging_config = imaging_config
        self.device = device_config.device
        self.saturation_level = saturation_level
    
    def load_obj(self, image_path, target_size=512):
        """Load image from file and pad to target_size x target_size"""
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        img = img.astype(np.float32) / 255.0
        
        H, W = img.shape
        pad_h = target_size - H
        pad_w = target_size - W
        
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left
        
        img_padded = np.pad(img, ((pad_top, pad_bottom), (pad_left, pad_right)), mode='constant', constant_values=0)
        
        obj = torch.tensor(img_padded, dtype=torch.float32).to(self.device)
        obj = obj[None, None, :, :]
        
        return obj
    
    def load_obj_crop_resize(self, image_path, target_size=512):
        """
        Load image, trim only white borders, center crop to square, resize.
        Preserves black space around the pattern.
        """
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        img = img.astype(np.float32) / 255.0
        
        H, W = img.shape
        
        # Only trim WHITE rows (top/bottom borders) - keep black columns
        row_means = img.mean(axis=1)
        non_white_rows = np.where(row_means < 0.95)[0]
        
        if len(non_white_rows) > 0:
            r_start = non_white_rows[0]
            r_end = non_white_rows[-1] + 1
            img = img[r_start:r_end, :]
        
        H, W = img.shape
        
        # Center crop to square (preserving black space)
        crop_size = min(H, W)
        top = (H - crop_size) // 2
        left = (W - crop_size) // 2
        img_cropped = img[top:top + crop_size, left:left + crop_size]
        
        # Resize to target_size
        img_resized = cv2.resize(img_cropped, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
        
        obj = torch.tensor(img_resized, dtype=torch.float32).to(self.device)
        obj = obj[None, None, :, :]
        
        return obj
    
 
    def create_blurred_image(self, obj, psf):
        """Create blurred image using single PSF"""
        blurred = conv2d(obj, psf)
        return blurred
    
    def create_pinhole(self, psf_size, pinhole_au=0.2):
        """
        Create pinhole disk mask.
        
        Args:
            psf_size: size of PSF array
            pinhole_au: pinhole size in Airy Units (default 0.2)
        
        Returns:
            pinhole: (1, 1, H, W) normalized pinhole disk
        """
        au_diameter_pixels = compute_airy_unit_pixels(
            wavelength_mm=self.psf_config.wave_length_mm,
            NA=self.psf_config.NA,
            pixel_size_mm=self.psf_config.output_pixel_size_mm
        )
        pinhole_radius_pixels = (pinhole_au * au_diameter_pixels) / 2
        pinhole = create_pinhole_disk(psf_size, pinhole_radius_pixels, device=self.device)
        return pinhole, au_diameter_pixels, pinhole_radius_pixels
    
    def create_detection_psf_with_pinhole(self, psf_det, pinhole):
        """
        Create detection PSF convolved with pinhole: h_det_pinhole = h_det ⊗ p
        
        Args:
            psf_det: detection PSF (typically Gaussian)
            pinhole: pinhole transmission function
        
        Returns:
            psf_det_pinhole: detection PSF with pinhole effect
        """
        psf_det_pinhole = conv2d(psf_det, pinhole)
        # No normalization here - preserve relative intensities
        return psf_det_pinhole
    
    def create_emission_psf(self, psf_ill, saturated=True):
        """
        Create emission PSF: h_emi = g(I) or h_emi = I (linear).
        
        For saturated imaging, applies nonlinear saturation g(I).
        For linear imaging, returns the illumination PSF as-is.
        
        The saturation_level (set in __init__) scales the PSF intensity before
        applying saturation: higher level = stronger saturation effect.
        
        Args:
            psf_ill: illumination PSF (intensity I)
            saturated: if True, apply saturation g(I); if False, return linear (h_emi = h_ill)
        
        Returns:
            psf_emi: emission PSF (normalized)
        """
        # Use saturation_level=0 for linear, saturation_level>0 for saturated
        level = self.saturation_level if saturated else 0
        psf_emi = nfomm_saturation_emission(psf_ill, mode="simple", saturation_level=level)
        return psf_emi
    
    def create_effective_psf(self, psf_emi, psf_det_pinhole, debug=False):
        """
        Create effective PSF: h_eff = h_emi · h_det_pinhole
        
        Element-wise multiplication of emission PSF with detection PSF (with pinhole).
        
        Args:
            psf_emi: emission PSF (g(I) for saturated, or h_ill for linear)
            psf_det_pinhole: detection PSF convolved with pinhole
        
        Returns:
            psf_eff: effective PSF (normalized)
        """
        if debug:
            H, W = psf_emi.shape[-2:]
            center = H // 2
            # Find where max of each PSF is
            emi_line = psf_emi[0, 0, center, :].cpu().numpy()
            det_line = psf_det_pinhole[0, 0, center, :].cpu().numpy()
            emi_max_idx = emi_line.argmax()
            det_half_max = det_line.max() / 2
            det_fwhm_left = center - (det_line[:center][::-1] > det_half_max).argmax()
            det_fwhm_right = center + (det_line[center:] > det_half_max).argmax()
            
            print(f"  DEBUG create_effective_psf:")
            print(f"    psf_emi center value: {psf_emi[0, 0, center, center].item():.6f}")
            print(f"    psf_emi max: {psf_emi.max().item():.6f}")
            print(f"    psf_emi ring peak at pixel: {emi_max_idx} (offset {emi_max_idx - center} from center)")
            print(f"    psf_det_pinhole center: {psf_det_pinhole[0, 0, center, center].item():.6f}")
            print(f"    psf_det_pinhole FWHM: ~{det_fwhm_right - det_fwhm_left} pixels")
        
        psf_eff = psf_emi * psf_det_pinhole  # element-wise multiply
        # Normalize by sum (energy conservation) for deconvolution
        psf_eff = psf_eff / (psf_eff.sum() + 1e-10)
        
        if debug:
            eff_line = psf_eff[0, 0, center, :].cpu().numpy()
            eff_max_idx = eff_line.argmax()
            print(f"    psf_eff center: {psf_eff[0, 0, center, center].item():.6f}")
            print(f"    psf_eff max: {psf_eff.max().item():.6f}")
            print(f"    psf_eff max at pixel: {eff_max_idx} (offset {eff_max_idx - center} from center)")
        
        return psf_eff
    
    def save_psf_tensor(self, psf, phase_mask, save_dir, prefix):
        """Save PSF and phase mask as PyTorch tensors"""
        cond_mkdir(save_dir)
        psf_path = os.path.join(save_dir, f"{prefix}_psf.pt")
        phase_path = os.path.join(save_dir, f"{prefix}_phase_mask.pt")
        torch.save(psf.cpu(), psf_path)
        torch.save(phase_mask.cpu(), phase_path)
        print(f"Saved {prefix} PSF tensor to {psf_path}")
        print(f"Saved {prefix} phase mask tensor to {phase_path}")
    
    def load_psf_tensor(self, save_dir, prefix):
        """Load PSF and phase mask from saved PyTorch tensors"""
        psf_path = os.path.join(save_dir, f"{prefix}_psf.pt")
        phase_path = os.path.join(save_dir, f"{prefix}_phase_mask.pt")
        
        if os.path.exists(psf_path) and os.path.exists(phase_path):
            psf = torch.load(psf_path).to(self.device)
            phase_mask = torch.load(phase_path).to(self.device)
            print(f"Loaded {prefix} PSF from {psf_path}")
            return psf, phase_mask
        return None, None
    
    def save_phase_mask(self, phase_mask, save_dir, prefix):
        """Save phase mask as PNG file"""
        cond_mkdir(save_dir)
        
        phase_np = tensor_to_numpy_2d(phase_mask)
        
        # Normalize phase to 0-2π range for visualization
        phase_norm = (phase_np - phase_np.min()) / (phase_np.max() - phase_np.min() + 1e-10)
        phase_png = (phase_norm * 255).astype(np.uint8)
        
        phase_path = os.path.join(save_dir, f"{prefix}_phase_mask.png")
        plt.imsave(phase_path, phase_png, cmap='hsv')
        
        print(f"Saved {prefix} phase mask to {phase_path}")
    
    def save_psf_and_image(self, psf, img, obj, save_dir, prefix, phase_mask=None):
        """Save PSF, image, and ground truth object as PNG files"""
        cond_mkdir(save_dir)
        
        psf_np = tensor_to_numpy_2d(psf)
        img_np = tensor_to_numpy_2d(img)
        obj_np = tensor_to_numpy_2d(obj)
        
        # Normalize to 0-1 range for PNG
        def normalize_for_png(arr):
            arr = arr.astype(np.float32)
            arr_min = arr.min()
            arr_max = arr.max()
            if arr_max > arr_min:
                arr = (arr - arr_min) / (arr_max - arr_min)
            else:
                arr = np.zeros_like(arr)
            return (arr * 255).astype(np.uint8)
        
        psf_png = normalize_for_png(psf_np)
        img_png = normalize_for_png(img_np)
        obj_png = normalize_for_png(obj_np)
        
        psf_path = os.path.join(save_dir, f"{prefix}_psf.png")
        img_path = os.path.join(save_dir, f"{prefix}_img.png")
        obj_path = os.path.join(save_dir, f"{prefix}_obj.png")
        
        plt.imsave(psf_path, psf_png, cmap='gray')
        plt.imsave(img_path, img_png, cmap='gray')
        plt.imsave(obj_path, obj_png, cmap='gray')
        
        print(f"Saved {prefix} PSF to {psf_path}")
        print(f"Saved {prefix} image to {img_path}")
        print(f"Saved {prefix} object to {obj_path}")
        
        if phase_mask is not None:
            self.save_phase_mask(phase_mask, save_dir, prefix)
    
    def run_demo(self, obj_size=None, N=None, psf_size=None, save_dir=None, visualize=None, 
                 results_dir=None, image_path=None, regenerate_psf=None, pinhole_au=0.2,
                 use_crop_resize=False, psf_conv_size=None):
        """
        Run the complete imaging demo with NFOMM forward model.
        
        Generates:
        - Raw PSFs (doughnut, gaussian)
        - Pinhole and detection PSF with pinhole
        - Effective PSFs (linear confocal gaussian, saturated doughnut)
        - Blurred images using correct effective PSFs
        
        Args:
            pinhole_au: pinhole size in Airy Units (default 0.2)
            use_crop_resize: if True, crop to square and resize (for non-square images like star_test)
            psf_conv_size: PSF size for convolution (default: same as psf_size, use smaller for efficiency)
        
        Returns:
            dict with all PSFs, images, and parameters
        """
        if obj_size is None:
            obj_size = self.object_config.size
        if N is None:
            N = self.psf_config.N
        if psf_size is None:
            psf_size = self.psf_config.psf_size
        if save_dir is None:
            save_dir = self.imaging_config.save_dir
        if visualize is None:
            visualize = self.imaging_config.visualize
        if regenerate_psf is None:
            regenerate_psf = self.imaging_config.regenerate_psf
        if results_dir is None:
            results_dir = os.path.join(save_dir, "results")
        
        cond_mkdir(results_dir)
        torch.manual_seed(self.imaging_config.random_seed)
        
        # Load or create object
        if image_path is not None:
            if use_crop_resize:
                obj_gt = self.load_obj_crop_resize(image_path, target_size=obj_size)
            else:
                obj_gt = self.load_obj(image_path, target_size=obj_size)
        else:
            obj_gt = create_2d_object(obj_size)
        if visualize:
            show(obj_gt[0, 0].cpu(), title="Ground Truth Object", save=True, save_name=os.path.join(results_dir, "ground_truth_object"))
        
        # Try to load PSFs if not regenerating
        psf_doughnut, phase_mask_doughnut = None, None
        psf_gaussian, phase_mask_gaussian = None, None
        
        if not regenerate_psf:
            print("Attempting to load existing PSFs...")
            psf_doughnut, phase_mask_doughnut = self.load_psf_tensor(save_dir, "doughnut")
            psf_gaussian, phase_mask_gaussian = self.load_psf_tensor(save_dir, "gaussian")
        
        # Generate raw PSFs if needed
        if psf_doughnut is None or regenerate_psf:
            if regenerate_psf:
                print("Regenerating doughnut PSF...")
            else:
                print("Creating doughnut PSF...")
            N_fourier = self.psf_config.N_fourier
            psf_doughnut, phase_mask_doughnut, _, _ = create_doughnut_psf(N=N, psf_size=psf_size, N_fourier=N_fourier)
            self.save_psf_tensor(psf_doughnut, phase_mask_doughnut, save_dir, "doughnut")
        else:
            print("Using existing doughnut PSF")
        
        if psf_gaussian is None or regenerate_psf:
            if regenerate_psf:
                print("Regenerating Gaussian PSF...")
            else:
                print("Creating Gaussian PSF...")
            N_fourier = self.psf_config.N_fourier
            psf_gaussian, phase_mask_gaussian, _, _ = create_gaussian_psf(N=N, psf_size=psf_size, N_fourier=N_fourier)
            self.save_psf_tensor(psf_gaussian, phase_mask_gaussian, save_dir, "gaussian")
        else:
            print("Using existing Gaussian PSF")
        
        # ===== NFOMM Forward Model =====
        print(f"\nBuilding NFOMM forward model (pinhole={pinhole_au} AU, saturation_level={self.saturation_level})...")
        
        # Create pinhole
        pinhole, au_diameter_pixels, pinhole_radius_pixels = self.create_pinhole(psf_size, pinhole_au)
        print(f"1 AU = {au_diameter_pixels:.2f} pixels, pinhole diameter = {pinhole_radius_pixels*2:.2f} pixels")
        
        # Detection PSF with pinhole: h_det_pinhole = h_det ⊗ p
        # Use Gaussian as detection PSF
        psf_det_pinhole = self.create_detection_psf_with_pinhole(psf_gaussian, pinhole)
        
        # Emission PSFs
        psf_emi_gaussian = self.create_emission_psf(psf_gaussian, saturated=False)  # linear: h_emi = h_ill
        psf_emi_doughnut = self.create_emission_psf(psf_doughnut, saturated=True)   # saturated: h_emi = g(I)
        
        # Effective PSFs: h_eff = h_emi · h_det_pinhole
        psf_eff_gaussian = self.create_effective_psf(psf_emi_gaussian, psf_det_pinhole)
        print("Creating saturated effective PSF...")
        psf_eff_saturated = self.create_effective_psf(psf_emi_doughnut, psf_det_pinhole, debug=True)
        
        # Crop PSFs for convolution if psf_conv_size is specified
        if psf_conv_size is not None and psf_conv_size < psf_size:
            print(f"Cropping PSFs from {psf_size}x{psf_size} to {psf_conv_size}x{psf_conv_size} for convolution...")
            psf_eff_gaussian_conv = central_crop(psf_eff_gaussian, psf_conv_size, psf_conv_size)
            psf_eff_saturated_conv = central_crop(psf_eff_saturated, psf_conv_size, psf_conv_size)
            psf_doughnut_conv = central_crop(psf_doughnut, psf_conv_size, psf_conv_size)
            psf_gaussian_conv = central_crop(psf_gaussian, psf_conv_size, psf_conv_size)
        else:
            psf_eff_gaussian_conv = psf_eff_gaussian
            psf_eff_saturated_conv = psf_eff_saturated
            psf_doughnut_conv = psf_doughnut
            psf_gaussian_conv = psf_gaussian
        
        # Generate blurred images using CORRECT effective PSFs
        print("Generating images with effective PSFs...")
        img_gaussian_confocal = self.create_blurred_image(obj_gt, psf_eff_gaussian_conv)
        img_saturated = self.create_blurred_image(obj_gt, psf_eff_saturated_conv)
        
        # Also generate linear (non-confocal) images for comparison
        img_doughnut_linear = self.create_blurred_image(obj_gt, psf_doughnut_conv)
        img_gaussian_linear = self.create_blurred_image(obj_gt, psf_gaussian_conv)
        
        if visualize:
            show(psf_doughnut[0, 0].cpu(), title="Doughnut PSF (raw)", save=True, save_name=os.path.join(results_dir, "psf_doughnut_raw"))
            show(psf_gaussian[0, 0].cpu(), title="Gaussian PSF (raw)", save=True, save_name=os.path.join(results_dir, "psf_gaussian_raw"))
            show(pinhole[0, 0].cpu(), title=f"Pinhole ({pinhole_au} AU)", save=True, save_name=os.path.join(results_dir, "pinhole"))
            show(psf_det_pinhole[0, 0].cpu(), title="Detection PSF with pinhole", save=True, save_name=os.path.join(results_dir, "psf_det_pinhole"))
            show(psf_emi_gaussian[0, 0].cpu(), title="Emission PSF (Gaussian, linear)", save=True, save_name=os.path.join(results_dir, "psf_emi_gaussian"))
            show(psf_emi_doughnut[0, 0].cpu(), title="Emission PSF (Doughnut, saturated)", save=True, save_name=os.path.join(results_dir, "psf_emi_doughnut"))
            show(psf_eff_gaussian[0, 0].cpu(), title="Effective PSF (Gaussian confocal)", save=True, save_name=os.path.join(results_dir, "psf_eff_gaussian"))
            show(psf_eff_saturated[0, 0].cpu(), title="Effective PSF (Doughnut saturated)", save=True, save_name=os.path.join(results_dir, "psf_eff_saturated"))
            show(img_gaussian_confocal[0, 0].cpu(), title="Image (Gaussian confocal)", save=True, save_name=os.path.join(results_dir, "image_gaussian_confocal"))
            show(img_saturated[0, 0].cpu(), title="Image (Doughnut saturated)", save=True, save_name=os.path.join(results_dir, "image_saturated"))
        
        print(f"All visualization images saved to {results_dir}")
        print("Demo completed!")
        
        # Return comprehensive results dict
        return {
            'obj_gt': obj_gt,
            # Raw PSFs (illumination)
            'psf_doughnut': psf_doughnut,
            'psf_gaussian': psf_gaussian,
            # Pinhole and detection
            'pinhole': pinhole,
            'pinhole_au': pinhole_au,
            'au_diameter_pixels': au_diameter_pixels,
            'psf_det_pinhole': psf_det_pinhole,
            # Emission PSFs
            'psf_emi_gaussian': psf_emi_gaussian,
            'psf_emi_doughnut': psf_emi_doughnut,
            # Effective PSFs
            'psf_eff_gaussian': psf_eff_gaussian,
            'psf_eff_saturated': psf_eff_saturated,
            # Blurred images (with correct effective PSFs)
            'img_gaussian_confocal': img_gaussian_confocal,
            'img_saturated': img_saturated,
            # Linear images (for comparison)
            'img_doughnut_linear': img_doughnut_linear,
            'img_gaussian_linear': img_gaussian_linear,
            # Parameters
            'saturation_level': self.saturation_level,
        }


if __name__ == "__main__":
    imaging = Imaging()
    image_path = "nfomm/data/barbara_gray.jpg"
    results = imaging.run_demo(obj_size=512, N=512, psf_size=512, image_path=image_path, pinhole_au=0.2)
    print(f"\nReturned keys: {list(results.keys())}")
