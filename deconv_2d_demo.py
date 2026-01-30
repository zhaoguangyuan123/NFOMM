"""2D Multiview Deconvolution Demo
Reconstruct object from blurred images using multiview Richardson-Lucy deconvolution
Following NFOMM documentation and similar to 1D case in easy_imging.py

Run with nohup:
"""

import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt

# Handle both direct execution and module execution
try:
    from .core.utils import show, cond_mkdir, edgetaper2d, central_crop, analyze_saturation_effect, analyze_otf
    from .core.params import imaging_config
    from .core.imaging import Imaging
    from .core.multiview_deconv import torch_deconv_2d
    from .core.frc import analyze_frc
except ImportError:
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from nfomm.core.utils import show, cond_mkdir, edgetaper2d, central_crop, analyze_saturation_effect, analyze_otf
    from nfomm.core.params import imaging_config
    from nfomm.core.imaging import Imaging
    from nfomm.core.multiview_deconv import torch_deconv_2d
    from nfomm.core.frc import analyze_frc




if __name__ == "__main__":
    # ===== Generate PSFs and images using Imaging class =====
    # All NFOMM forward model logic (pinhole, saturation, effective PSFs) is in Imaging
    # saturation_level: higher = stronger saturation effect (simulates higher excitation power)
    # 1.0 is too weak to see difference, use ~100-1000 for visible saturation
    imaging = Imaging(saturation_level=20.0)  # Higher saturation now viable with 4x finer PSF sampling
    PINHOLE_AU = 0.2
    
    # Object selection: "barbara" or "star"
    OBJECT_TYPE = "star"  # Change to "barbara" for Barbara image
    
    if OBJECT_TYPE == "star":
        image_path = "nfomm/data/star_test.png"
        use_crop_resize = True  # Star image is rectangular, needs crop+resize
    else:
        image_path = "nfomm/data/barbara_gray.jpg"
        use_crop_resize = False
    
    print(f"Running NFOMM imaging demo with object: {OBJECT_TYPE}...")
    # psf_size=1024 with N_fourier=8192 gives 2x finer PSF sampling
    # psf_conv_size=64 crops PSF for convolution (same physical area as old 32x32 at 2x resolution)
    PSF_CONV_SIZE = 64
    results = imaging.run_demo(
        obj_size=512, N=512, psf_size=1024, 
        image_path=image_path, visualize=False, pinhole_au=PINHOLE_AU,
        use_crop_resize=use_crop_resize, psf_conv_size=PSF_CONV_SIZE
    )
    
    # Extract from results dict
    obj_gt = results['obj_gt']
    psf_doughnut = results['psf_doughnut']
    psf_gaussian = results['psf_gaussian']
    psf_emi_gaussian = results['psf_emi_gaussian']
    psf_emi_doughnut = results['psf_emi_doughnut']
    psf_eff_gaussian = results['psf_eff_gaussian']
    psf_eff_saturated = results['psf_eff_saturated']
    pinhole = results['pinhole']
    psf_det_pinhole = results['psf_det_pinhole']
    img_gaussian_confocal = results['img_gaussian_confocal']
    img_saturated = results['img_saturated']
    img_doughnut_linear = results['img_doughnut_linear']
    img_gaussian_linear = results['img_gaussian_linear']
    
    print(f"Pinhole: {PINHOLE_AU} AU = {results['au_diameter_pixels']*PINHOLE_AU:.2f} pixels diameter")
    print(f"Saturation level: {results['saturation_level']}")
    
    # ===== Create output directories (organized by object type) =====
    results_dir = os.path.join(imaging_config.save_dir, OBJECT_TYPE)
    psf_dir = os.path.join(results_dir, "psf")
    psf_full_dir = os.path.join(psf_dir, "full")
    psf_crop_dir = os.path.join(psf_dir, "cropped")
    blurred_dir = os.path.join(results_dir, "blurred")
    gt_dir = os.path.join(results_dir, "ground_truth")
    recon_dir = os.path.join(results_dir, "reconstruction")
    saturated_dir = os.path.join(results_dir, "saturated")
    for d in [results_dir, psf_dir, psf_full_dir, psf_crop_dir, blurred_dir, gt_dir, recon_dir, saturated_dir]:
        cond_mkdir(d)
    
    # ===== Processing parameters =====
    # With N_fourier=8192 (2x), psf_size=1024 (2x), use 64x64 crop (2x of old 32)
    PSF_CROP_SIZE = 64  # 2x finer sampling to resolve sharp saturation edges
    NUM_ITER = 150
    EDGE_CROP = 64
    
    # Star test has black edges naturally, no need to crop results
    # Barbara needs edge crop to remove boundary artifacts
    SKIP_RESULT_CROP = (OBJECT_TYPE == "star")
    RESULT_SIZE = 512 if SKIP_RESULT_CROP else 512 - EDGE_CROP * 2  # 512 for star, 384 for barbara
    
    # ===== Save PSFs (full and cropped) =====
    print("\nSaving PSFs...")
    
    # Crop PSFs for visualization
    psf_doughnut_crop = central_crop(psf_doughnut, PSF_CROP_SIZE, PSF_CROP_SIZE)
    psf_gaussian_crop = central_crop(psf_gaussian, PSF_CROP_SIZE, PSF_CROP_SIZE)
    pinhole_crop = central_crop(pinhole, PSF_CROP_SIZE, PSF_CROP_SIZE)
    psf_det_pinhole_crop = central_crop(psf_det_pinhole, PSF_CROP_SIZE, PSF_CROP_SIZE)
    psf_emi_gaussian_crop = central_crop(psf_emi_gaussian, PSF_CROP_SIZE, PSF_CROP_SIZE)
    psf_emi_doughnut_crop = central_crop(psf_emi_doughnut, PSF_CROP_SIZE, PSF_CROP_SIZE)
    psf_eff_gaussian_crop = central_crop(psf_eff_gaussian, PSF_CROP_SIZE, PSF_CROP_SIZE)
    psf_eff_saturated_crop = central_crop(psf_eff_saturated, PSF_CROP_SIZE, PSF_CROP_SIZE)
    
    # Illumination PSFs - full size
    show(psf_doughnut[0, 0].cpu(), title="PSF Doughnut (illumination)", save=True, 
         save_name=os.path.join(psf_full_dir, "doughnut_ill"))
    show(psf_gaussian[0, 0].cpu(), title="PSF Gaussian (illumination)", save=True, 
         save_name=os.path.join(psf_full_dir, "gaussian_ill"))
    show(pinhole[0, 0].cpu(), title=f"Pinhole ({PINHOLE_AU} AU)", save=True, 
         save_name=os.path.join(psf_full_dir, "pinhole"))
    show(psf_det_pinhole[0, 0].cpu(), title="Detection PSF with pinhole", save=True, 
         save_name=os.path.join(psf_full_dir, "gaussian_det_pinhole"))
    show(psf_emi_gaussian[0, 0].cpu(), title="Emission PSF (Gaussian, linear)", save=True, 
         save_name=os.path.join(psf_full_dir, "gaussian_emi"))
    show(psf_emi_doughnut[0, 0].cpu(), title="Emission PSF (Doughnut, saturated)", save=True, 
         save_name=os.path.join(psf_full_dir, "doughnut_emi_saturated"))
    show(psf_eff_gaussian[0, 0].cpu(), title="Effective PSF (Gaussian confocal)", save=True, 
         save_name=os.path.join(psf_full_dir, "gaussian_eff_confocal"))
    show(psf_eff_saturated[0, 0].cpu(), title="Effective PSF (Doughnut saturated)", save=True, 
         save_name=os.path.join(psf_full_dir, "doughnut_eff_saturated"))
    
    # Cropped PSFs
    show(psf_doughnut_crop[0, 0].cpu(), title="PSF Doughnut (illumination)", save=True, 
         save_name=os.path.join(psf_crop_dir, "doughnut_ill"))
    show(psf_gaussian_crop[0, 0].cpu(), title="PSF Gaussian (illumination)", save=True, 
         save_name=os.path.join(psf_crop_dir, "gaussian_ill"))
    show(pinhole_crop[0, 0].cpu(), title=f"Pinhole ({PINHOLE_AU} AU)", save=True, 
         save_name=os.path.join(psf_crop_dir, "pinhole"))
    show(psf_det_pinhole_crop[0, 0].cpu(), title="Detection PSF with pinhole", save=True, 
         save_name=os.path.join(psf_crop_dir, "gaussian_det_pinhole"))
    show(psf_emi_gaussian_crop[0, 0].cpu(), title="Emission PSF (Gaussian)", save=True, 
         save_name=os.path.join(psf_crop_dir, "gaussian_emi"))
    show(psf_emi_doughnut_crop[0, 0].cpu(), title="Emission PSF (Doughnut saturated)", save=True, 
         save_name=os.path.join(psf_crop_dir, "doughnut_emi_saturated"))
    show(psf_eff_gaussian_crop[0, 0].cpu(), title="Effective PSF (Gaussian confocal)", save=True, 
         save_name=os.path.join(psf_crop_dir, "gaussian_eff_confocal"))
    show(psf_eff_saturated_crop[0, 0].cpu(), title="Effective PSF (Doughnut saturated)", save=True, 
         save_name=os.path.join(psf_crop_dir, "doughnut_eff_saturated"))
    
    # ===== OTF Analysis =====
    print("\nComputing OTFs and plotting line curves...")
    otf_dir = os.path.join(psf_dir, "otf")
    analyze_otf(results, otf_dir, freq_limit=100)
    
    # ===== Saturation Effect Analysis =====
    print("\nAnalyzing saturation effect on emission PSF...")
    psf_tensor_path = os.path.join(imaging_config.save_dir, "doughnut_psf.pt")
    sat_comparison_path = os.path.join(otf_dir, "saturation_comparison.png")
    analyze_saturation_effect(psf_tensor_path, sat_comparison_path)
    
    # ===== Save blurred images =====
    print("Saving blurred images...")
    show(img_doughnut_linear[0, 0].cpu(), title="Blurred (Doughnut linear)", save=True, 
         save_name=os.path.join(blurred_dir, "doughnut_linear"))
    show(img_gaussian_linear[0, 0].cpu(), title="Blurred (Gaussian linear)", save=True, 
         save_name=os.path.join(blurred_dir, "gaussian_linear"))
    show(img_gaussian_confocal[0, 0].cpu(), title="Blurred (Gaussian confocal)", save=True, 
         save_name=os.path.join(blurred_dir, "gaussian_confocal"))
    show(img_saturated[0, 0].cpu(), title="Blurred (Doughnut saturated)", save=True, 
         save_name=os.path.join(blurred_dir, "saturated"))
    
    # ===== Save ground truth =====
    print("Saving ground truth...")
    show(obj_gt[0, 0].cpu(), title="Ground Truth Object", save=True, 
         save_name=os.path.join(gt_dir, "full"))
    if not SKIP_RESULT_CROP:
        obj_gt_display = central_crop(obj_gt, RESULT_SIZE, RESULT_SIZE)
        show(obj_gt_display[0, 0].cpu(), title=f"Ground Truth (cropped {RESULT_SIZE}x{RESULT_SIZE})", save=True, 
             save_name=os.path.join(gt_dir, "cropped"))
    
    # ===== Edge tapering =====
    print(f"\nApplying edge tapering (PSF crop size={PSF_CROP_SIZE})...")
    psf_eff_gaussian_cropped = central_crop(psf_eff_gaussian, PSF_CROP_SIZE, PSF_CROP_SIZE)
    psf_eff_saturated_cropped = central_crop(psf_eff_saturated, PSF_CROP_SIZE, PSF_CROP_SIZE)
    
    img_gaussian_confocal_tapered = edgetaper2d(img_gaussian_confocal, psf_eff_gaussian_cropped)
    img_saturated_tapered = edgetaper2d(img_saturated, psf_eff_saturated_cropped)
    
    # ===== Reconstruction =====
    print("\n" + "="*60)
    print("2D Multiview Deconvolution (NFOMM)")
    print("="*60)
    
    # Single-view: Gaussian confocal
    print("\nSingle-view deconvolution (Gaussian confocal)...")
    with torch.no_grad():
        obj_deconv_gaussian = torch_deconv_2d(img_gaussian_confocal_tapered, psf_eff_gaussian, num_iter=NUM_ITER)
    obj_deconv_gaussian_out = central_crop(obj_deconv_gaussian, RESULT_SIZE, RESULT_SIZE) if not SKIP_RESULT_CROP else obj_deconv_gaussian
    show(obj_deconv_gaussian_out[0, 0].cpu(), 
         title="Reconstruction (Gaussian confocal)", 
         save=True, 
         save_name=os.path.join(recon_dir, "gaussian_confocal"))
    
    # Single-view: Doughnut saturated
    print("Single-view deconvolution (Doughnut saturated)...")
    with torch.no_grad():
        obj_deconv_saturated = torch_deconv_2d(img_saturated_tapered, psf_eff_saturated, num_iter=NUM_ITER)
    obj_deconv_saturated_out = central_crop(obj_deconv_saturated, RESULT_SIZE, RESULT_SIZE) if not SKIP_RESULT_CROP else obj_deconv_saturated
    show(obj_deconv_saturated_out[0, 0].cpu(), 
         title="Reconstruction (Doughnut saturated)", 
         save=True, 
         save_name=os.path.join(recon_dir, "doughnut_saturated_singleview"))
    
    # Multiview: Doughnut saturated + Gaussian confocal
    # Use OTF-weighted blending: weight at each freq ∝ |OTF|² 
    # This automatically gives more weight to doughnut at HF, gaussian at LF/nulls
    print("Multiview deconvolution (OTF-weighted)...")
    images_multiview = torch.cat([img_saturated_tapered, img_gaussian_confocal_tapered], dim=0)
    psfs_multiview = torch.cat([psf_eff_saturated, psf_eff_gaussian], dim=0)
    with torch.no_grad():
        obj_deconv_multiview = torch_deconv_2d(images_multiview, psfs_multiview, num_iter=200, otf_weighted=True)
    obj_deconv_multiview_out = central_crop(obj_deconv_multiview, RESULT_SIZE, RESULT_SIZE) if not SKIP_RESULT_CROP else obj_deconv_multiview
    show(obj_deconv_multiview_out[0, 0].cpu(), 
         title="Reconstruction (Doughnut + Gaussian multiview)", 
         save=True, 
         save_name=os.path.join(recon_dir, "doughnut_gaussian_multiview"))
    
    # ===== FRC Analysis =====
    print("\n" + "="*60)
    print("FRC Resolution Analysis")
    print("="*60)
    
    # Get ground truth at same size as reconstructions (512 for star, cropped for others)
    obj_gt_for_frc = obj_gt if SKIP_RESULT_CROP else central_crop(obj_gt, RESULT_SIZE, RESULT_SIZE)
    
    reconstructions = {
        "Gaussian confocal": obj_deconv_gaussian_out,
        "Doughnut saturated": obj_deconv_saturated_out,
        "Multiview (Doughnut+Gaussian)": obj_deconv_multiview_out
    }
    
    frc_dir = os.path.join(results_dir, "frc")
    resolutions = analyze_frc(obj_gt_for_frc, reconstructions, frc_dir)
    
    print(f"\nAll results saved to {results_dir}")
    print("Deconvolution demo completed!")
