from .utils import normalize_psf, conv2d
import torch
import torch.fft as fft


def compute_otf_weights(psfs, target_size, eps=1e-10):
    """
    Compute frequency-dependent weights based on OTF magnitude.
    Weight at each frequency ∝ |OTF|² / Σ|OTF|²
    
    Args:
        psfs: (B, C, H, W) PSFs
        target_size: (H, W) tuple - size to compute OTF at (matches image size)
        
    Returns:
        weights: (B, C, target_H, target_W) frequency-dependent weights (sum to 1 at each freq)
    """
    B, C, psf_H, psf_W = psfs.shape
    target_H, target_W = target_size
    device = psfs.device
    
    # Pad PSFs to target size for OTF computation
    pad_h = target_H - psf_H
    pad_w = target_W - psf_W
    
    if pad_h >= 0 and pad_w >= 0:
        # PSF smaller than target - zero pad
        psfs_padded = torch.zeros(B, C, target_H, target_W, device=device)
        psfs_padded[:, :, :psf_H, :psf_W] = psfs
    else:
        # PSF larger than target - center crop
        start_h = (psf_H - target_H) // 2
        start_w = (psf_W - target_W) // 2
        psfs_padded = psfs[:, :, start_h:start_h+target_H, start_w:start_w+target_W]
    
    # Compute OTFs at target resolution
    otfs = fft.fft2(psfs_padded)
    otf_power = torch.abs(otfs) ** 2  # |OTF|²
    
    # Sum over views at each frequency
    total_power = otf_power.sum(dim=0, keepdim=True) + eps  # (1, C, H, W)
    
    # Normalize weights at each frequency
    weights = otf_power / total_power  # (B, C, H, W)
    
    return weights


def torch_deconv_2d(images, psfs, num_iter=50, weights=None, otf_weighted=False):
    """
    2D Richardson-Lucy deconvolution (single-view or multiview)
    
    Single-view (NFOMM Section 6.1):
    ρ^(k+1) = ρ^(k) · [((y / (ρ^(k) * h_eff)) * h̃_eff)]
    
    Multiview (NFOMM Section 6.2):
    ρ^(k+1) = ρ^(k) · Σ_ℓ w_ℓ [((y_ℓ / (ρ^(k) * h_eff,ℓ)) * h̃_eff,ℓ)]
    
    Args:
        images: Tensor of shape (B, C, H, W) - blurred images (B=1 for single-view, B>1 for multiview)
        psfs: Tensor of shape (B, C, H, W) - corresponding PSFs
        num_iter: Number of iterations
        weights: Optional list of scalar weights for each view (default: equal weights)
        otf_weighted: If True, use frequency-dependent OTF-based weighting (recommended for multiview)
                      Weight at each freq ∝ |OTF|² - gives more weight to view with stronger signal
    
    Returns:
        Reconstructed object of shape (1, C, H, W)
    """
    assert images.shape[0] == psfs.shape[0], 'Number of images and PSFs must match'
    
    # Fix root cause: ensure 4D tensors by removing extra singleton dimensions
    # This happens when PSF/image creation accidentally adds extra dimensions
    if len(images.shape) == 5:
        # [1, 1, 1, H, W] -> [1, 1, H, W] by removing middle singleton dim
        if images.shape[0] == 1 and images.shape[1] == 1 and images.shape[2] == 1:
            images = images.squeeze(2)
        else:
            # Merge first two batch dims: [B1, B2, C, H, W] -> [B1*B2, C, H, W]
            images = images.reshape(-1, images.shape[2], images.shape[3], images.shape[4])
    elif len(images.shape) == 3:
        images = images[None, :, :, :]
    elif len(images.shape) == 2:
        images = images[None, None, :, :]
    
    if len(psfs.shape) == 5:
        # [1, 1, 1, H, W] -> [1, 1, H, W] by removing middle singleton dim
        if psfs.shape[0] == 1 and psfs.shape[1] == 1 and psfs.shape[2] == 1:
            psfs = psfs.squeeze(2)
        else:
            # Merge first two batch dims: [B1, B2, C, H, W] -> [B1*B2, C, H, W]
            psfs = psfs.reshape(-1, psfs.shape[2], psfs.shape[3], psfs.shape[4])
    elif len(psfs.shape) == 3:
        psfs = psfs[None, :, :, :]
    elif len(psfs.shape) == 2:
        psfs = psfs[None, None, :, :]
    
    # Now ensure they are exactly 4D
    if len(images.shape) != 4 or len(psfs.shape) != 4:
        raise ValueError(f"After shape fixing, tensors must be 4D (B, C, H, W). Got images.shape={images.shape}, psfs.shape={psfs.shape}. "
                        f"This indicates a bug in PSF generation or image creation.")
    
    # Normalize PSFs
    psfs = normalize_psf(psfs)
    
    B, C, H, W = images.shape
    device = images.device
    
    # Initialize estimate (uniform)
    im_deconv = torch.full((1, C, H, W), 0.5, device=device)
    
    # Create flipped PSFs for backprojection (h̃ = h(-r))
    psfs_mirror = torch.flip(psfs, (-2, -1))
    
    eps = 1e-14
    
    for i in range(num_iter):
        # Forward: ρ^(k) * h_eff,ℓ for each view
        convs = []
        for b in range(B):
            conv = conv2d(im_deconv, psfs[b:b+1]) + eps
            if len(conv.shape) != 4:
                raise ValueError(f"conv2d returned tensor with shape {conv.shape}, expected 4D (B, C, H, W)")
            convs.append(conv)
        
        # Relative blur: y_ℓ / (ρ^(k) * h_eff,ℓ)
        relative_blurs = []
        for b in range(B):
            relative_blur = images[b:b+1] / convs[b]
            relative_blurs.append(relative_blur)
        
        # Backproject: (relative_blur * h̃_eff,ℓ) for each view
        backprojs = []
        for b in range(B):
            backproj = conv2d(relative_blurs[b], psfs_mirror[b:b+1])
            backprojs.append(backproj)
        
        # Weighted average over views (works for both single-view B=1 and multiview B>1)
        if otf_weighted and B > 1:
            # OTF-based frequency-dependent weighting
            # Weight each backprojection in Fourier domain by |OTF|² / Σ|OTF|²
            otf_weights = compute_otf_weights(psfs, (H, W))  # (B, C, H, W) at image resolution
            
            weighted_sum = torch.zeros_like(im_deconv)
            for b in range(B):
                # Apply frequency-dependent weight in Fourier domain
                bp_fft = fft.fft2(backprojs[b])
                weighted_bp_fft = bp_fft * otf_weights[b:b+1]
                weighted_bp = torch.real(fft.ifft2(weighted_bp_fft))
                weighted_sum = weighted_sum + weighted_bp
            
            im_deconv = im_deconv * weighted_sum
        elif weights is not None:
            # Custom scalar weights - must sum to 1 for proper normalization
            weights_tensor = torch.tensor(weights, device=device, dtype=torch.float32)
            weights_tensor = weights_tensor / weights_tensor.sum()  # Normalize
            weighted_sum = sum(w * bp for w, bp in zip(weights_tensor, backprojs))
            im_deconv = im_deconv * weighted_sum
        else:
            # Equal weights (original behavior)
            sum_backproj = torch.stack(backprojs, dim=0).sum(dim=0)
            im_deconv = im_deconv * (sum_backproj / B)
    
    return torch.abs(im_deconv)