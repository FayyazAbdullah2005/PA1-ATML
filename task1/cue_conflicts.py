import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
from tqdm import tqdm

# Add task1/adain to sys.path
sys.path.append(os.path.abspath('task1/adain'))
sys.path.append(os.path.abspath('models/adain'))
try:
    import task1.adain.net as net
    import task1.adain.function as function
except ImportError:
    import net
    import function

CLASS_PAIRS = [
    (0, 1), # (airplane, bird)
    (2, 3), # (car, cat)
    (9, 5), # (truck, dog)
    (7, 6), # (horse, frog)
    (4, 8)  # (deer, monkey)
]

def compute_ssim(img1, img2):
    # img1, img2: [1, 1, H, W] in [0, 1]
    # Simple windowed SSIM
    C1 = (0.01) ** 2
    C2 = (0.03) ** 2
    
    mu1 = F.avg_pool2d(img1, kernel_size=11, stride=1, padding=5)
    mu2 = F.avg_pool2d(img2, kernel_size=11, stride=1, padding=5)
    
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2
    
    sigma1_sq = F.avg_pool2d(img1 * img1, kernel_size=11, stride=1, padding=5) - mu1_sq
    sigma2_sq = F.avg_pool2d(img2 * img2, kernel_size=11, stride=1, padding=5) - mu2_sq
    sigma12 = F.avg_pool2d(img1 * img2, kernel_size=11, stride=1, padding=5) - mu1_mu2
    
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean().item()

def compute_sobel_edge_correlation(img1, img2):
    # Grayscale conversion
    g1 = 0.2989 * img1[:, 0:1] + 0.5870 * img1[:, 1:2] + 0.1140 * img1[:, 2:3]
    g2 = 0.2989 * img2[:, 0:1] + 0.5870 * img2[:, 1:2] + 0.1140 * img2[:, 2:3]
    
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32, device=img1.device).view(1, 1, 3, 3)
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32, device=img1.device).view(1, 1, 3, 3)
    
    e1_x = F.conv2d(g1, sobel_x, padding=1)
    e1_y = F.conv2d(g1, sobel_y, padding=1)
    edge1 = torch.sqrt(e1_x ** 2 + e1_y ** 2 + 1e-8).view(-1)
    
    e2_x = F.conv2d(g2, sobel_x, padding=1)
    e2_y = F.conv2d(g2, sobel_y, padding=1)
    edge2 = torch.sqrt(e2_x ** 2 + e2_y ** 2 + 1e-8).view(-1)
    
    # Pearson correlation
    e1_norm = edge1 - edge1.mean()
    e2_norm = edge2 - edge2.mean()
    denom = (e1_norm.norm() * e2_norm.norm()) + 1e-8
    return (torch.dot(e1_norm, e2_norm) / denom).item()

class AdaINGenerator:
    def __init__(self, vgg_path=None, decoder_path=None, device='cpu'):
        if vgg_path is None:
            vgg_path = 'task1/adain/vgg_normalised.pth' if os.path.exists('task1/adain/vgg_normalised.pth') else 'models/adain/vgg_normalised.pth'
        if decoder_path is None:
            decoder_path = 'task1/adain/decoder.pth' if os.path.exists('task1/adain/decoder.pth') else 'models/adain/decoder.pth'
        self.device = device
        self.vgg = net.vgg
        self.vgg.load_state_dict(torch.load(vgg_path, map_location=device))
        self.vgg.eval().to(device)
        
        self.decoder = net.decoder
        self.decoder.load_state_dict(torch.load(decoder_path, map_location=device))
        self.decoder.eval().to(device)
        
        # We extract features up to relu4_1
        self.enc_1 = nn.Sequential(*list(self.vgg.children())[:4]).eval().to(device)
        self.enc_2 = nn.Sequential(*list(self.vgg.children())[4:11]).eval().to(device)
        self.enc_3 = nn.Sequential(*list(self.vgg.children())[11:18]).eval().to(device)
        self.enc_4 = nn.Sequential(*list(self.vgg.children())[18:31]).eval().to(device)

    def encode(self, x):
        h1 = self.enc_1(x)
        h2 = self.enc_2(h1)
        h3 = self.enc_3(h2)
        h4 = self.enc_4(h3)
        return h4

    def stylize(self, content, style, alpha=0.8):
        with torch.no_grad():
            feat_c = self.encode(content)
            feat_s = self.encode(style)
            feat_adain = function.adaptive_instance_normalization(feat_c, feat_s)
            feat_interp = alpha * feat_adain + (1.0 - alpha) * feat_c
            stylized = self.decoder(feat_interp)
            return torch.clamp(stylized, 0.0, 1.0)

def generate_cue_conflicts(dataset, class_pairs=CLASS_PAIRS, alpha=0.8, target_per_dir=22, ssim_thresh=0.35, edge_thresh=0.40, device='cpu'):
    generator = AdaINGenerator(device=device)
    
    # Organize dataset samples by class
    class_indices = {c: [] for c in range(10)}
    for idx in range(len(dataset)):
        _, target = dataset[idx]
        class_indices[target].append(idx)
        
    accepted_conflicts = []
    accepted_count = 0
    rejected_count = 0
    
    # Generate for each pair in both directions
    directed_pairs = []
    for cA, cB in class_pairs:
        directed_pairs.append((cA, cB))
        directed_pairs.append((cB, cA))
        
    print(f"Generating AdaIN cue conflicts (alpha={alpha}, SSIM >= {ssim_thresh}, Edge Corr >= {edge_thresh})...")
    
    for content_class, style_class in directed_pairs:
        c_idxs = class_indices[content_class]
        s_idxs = class_indices[style_class]
        
        pair_accepted = 0
        pair_rejected = 0
        
        # Iterate through available samples
        for ci in c_idxs:
            if pair_accepted >= target_per_dir:
                break
            # Random style image from style class
            si = np.random.choice(s_idxs)
            
            c_img, _ = dataset[ci]
            s_img, _ = dataset[si]
            
            c_tensor = c_img.unsqueeze(0).to(device)
            s_tensor = s_img.unsqueeze(0).to(device)
            
            stylized = generator.stylize(c_tensor, s_tensor, alpha=alpha)
            
            # Compute rejection criteria
            c_gray = 0.2989 * c_tensor[:, 0:1] + 0.5870 * c_tensor[:, 1:2] + 0.1140 * c_tensor[:, 2:3]
            st_gray = 0.2989 * stylized[:, 0:1] + 0.5870 * stylized[:, 1:2] + 0.1140 * stylized[:, 2:3]
            
            ssim_val = compute_ssim(c_gray, st_gray)
            edge_corr = compute_sobel_edge_correlation(c_tensor, stylized)
            
            is_valid = (ssim_val >= ssim_thresh) and (edge_corr >= edge_thresh)
            
            if is_valid:
                pair_accepted += 1
                accepted_count += 1
                accepted_conflicts.append({
                    'image': stylized.cpu().squeeze(0),
                    'content_image': c_img,
                    'style_image': s_img,
                    'content_class': content_class,
                    'style_class': style_class,
                    'ssim': ssim_val,
                    'edge_corr': edge_corr,
                    'status': 'accepted'
                })
            else:
                pair_rejected += 1
                rejected_count += 1
                if pair_rejected <= 3: # Keep a few rejected examples for the report
                    accepted_conflicts.append({
                        'image': stylized.cpu().squeeze(0),
                        'content_image': c_img,
                        'style_image': s_img,
                        'content_class': content_class,
                        'style_class': style_class,
                        'ssim': ssim_val,
                        'edge_corr': edge_corr,
                        'status': 'rejected'
                    })
                    
    valid_conflicts = [item for item in accepted_conflicts if item['status'] == 'accepted']
    print(f"Generated total cue-conflict candidates: {accepted_count + rejected_count}")
    print(f"Accepted: {accepted_count} (Valid for evaluation: {len(valid_conflicts)})")
    print(f"Rejected: {rejected_count}")
    
    return accepted_conflicts, accepted_count, rejected_count
