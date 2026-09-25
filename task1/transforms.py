import torch
import torch.nn.functional as F
import numpy as np

def rgb_to_hsv(rgb):
    # rgb: [B, 3, H, W] in [0, 1]
    r, g, b = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    max_c, _ = torch.max(rgb, dim=1)
    min_c, _ = torch.min(rgb, dim=1)
    delta = max_c - min_c + 1e-7

    # Hue
    h = torch.zeros_like(max_c)
    mask_r = (max_c == r)
    mask_g = (max_c == g) & (~mask_r)
    mask_b = (max_c == b) & (~mask_r) & (~mask_g)

    h[mask_r] = ((g[mask_r] - b[mask_r]) / delta[mask_r]) % 6.0
    h[mask_g] = ((b[mask_g] - r[mask_g]) / delta[mask_g]) + 2.0
    h[mask_b] = ((r[mask_b] - g[mask_b]) / delta[mask_b]) + 4.0
    h = h / 6.0

    # Saturation
    s = torch.zeros_like(max_c)
    mask_non_zero = (max_c > 0)
    s[mask_non_zero] = delta[mask_non_zero] / max_c[mask_non_zero]

    # Value
    v = max_c
    return torch.stack([h, s, v], dim=1)

def hsv_to_rgb(hsv):
    # hsv: [B, 3, H, W] in [0, 1]
    h, s, v = hsv[:, 0] * 6.0, hsv[:, 1], hsv[:, 2]
    c = v * s
    x = c * (1.0 - torch.abs((h % 2.0) - 1.0))
    m = v - c

    zeros = torch.zeros_like(h)
    i = torch.floor(h).long() % 6

    rgb_chunks = [
        torch.stack([c, x, zeros], dim=1),
        torch.stack([x, c, zeros], dim=1),
        torch.stack([zeros, c, x], dim=1),
        torch.stack([zeros, x, c], dim=1),
        torch.stack([x, zeros, c], dim=1),
        torch.stack([c, zeros, x], dim=1)
    ]

    out = torch.zeros_like(hsv)
    for k in range(6):
        mask = (i == k).unsqueeze(1).repeat(1, 3, 1, 1)
        out[mask] = rgb_chunks[k][mask]

    return out + m.unsqueeze(1)

def apply_grayscale(imgs):
    # imgs: [B, 3, H, W] in [0, 1]
    r, g, b = imgs[:, 0:1], imgs[:, 1:2], imgs[:, 2:3]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return gray.repeat(1, 3, 1, 1)

def apply_hue_rotation_180(imgs):
    # Shift hue by 180 degrees (+0.5 mod 1.0)
    hsv = rgb_to_hsv(imgs)
    hsv[:, 0] = (hsv[:, 0] + 0.5) % 1.0
    return torch.clamp(hsv_to_rgb(hsv), 0.0, 1.0)

def apply_translation(imgs, shift, direction='north'):
    # imgs: [B, 3, H, W]
    if shift == 0:
        return imgs.clone()
    
    B, C, H, W = imgs.shape
    # Pad symmetrically using reflection padding
    padded = F.pad(imgs, (shift, shift, shift, shift), mode='reflect')
    
    # Original image coordinates in padded tensor are [shift : shift+H, shift : shift+W]
    if direction == 'north': # shift object up (crop downwards)
        return padded[:, :, 2 * shift : 2 * shift + H, shift : shift + W]
    elif direction == 'south': # shift object down (crop upwards)
        return padded[:, :, 0 : H, shift : shift + W]
    elif direction == 'east': # shift object right (crop leftwards)
        return padded[:, :, shift : shift + H, 0 : W]
    elif direction == 'west': # shift object left (crop rightwards)
        return padded[:, :, shift : shift + H, 2 * shift : 2 * shift + W]
    else:
        raise ValueError(f"Unknown direction: {direction}")

def apply_patch_shuffle_4x4(imgs, seed=6304):
    # imgs: [B, 3, 224, 224]
    B, C, H, W = imgs.shape
    assert H == 224 and W == 224, "Expected 224x224 images"
    grid = 4
    patch_size = 56 # 224 // 4
    
    # Extract patches: [B, 16, 3, 56, 56]
    patches = []
    for r in range(grid):
        for c in range(grid):
            p = imgs[:, :, r * patch_size : (r + 1) * patch_size, c * patch_size : (c + 1) * patch_size]
            patches.append(p)
    patches = torch.stack(patches, dim=1) # [B, 16, 3, 56, 56]
    
    rng = np.random.RandomState(seed)
    shuffled_imgs = []
    
    for i in range(B):
        # Generate one non-identity permutation per image
        perm = rng.permutation(16)
        while np.array_equal(perm, np.arange(16)):
            perm = rng.permutation(16)
            
        shuffled_p = patches[i, perm] # [16, 3, 56, 56]
        
        # Reassemble 4x4 image
        rows = []
        for r in range(grid):
            cols = []
            for c in range(grid):
                cols.append(shuffled_p[r * grid + c])
            rows.append(torch.cat(cols, dim=-1))
        reconstructed = torch.cat(rows, dim=-2)
        shuffled_imgs.append(reconstructed)
        
    return torch.stack(shuffled_imgs, dim=0)
