import torch

class MahalanobisScorer:
    """
    Mahalanobis distance-based novelty score.
    Computes distance from known-class feature clusters.
    """
    def __init__(self, num_classes=10):
        self.num_classes = num_classes
        self.class_means = None
        self.inv_cov_diag = None

    def fit(self, features, labels):
        """
        Estimate class means \mu_c and one shared diagonal covariance \Sigma
        from unaugmented CIFAR-10 training features.
        """
        d = features.size(1)
        self.class_means = torch.zeros(self.num_classes, d, device=features.device)
        
        for c in range(self.num_classes):
            mask = (labels == c)
            if mask.sum() > 0:
                self.class_means[c] = features[mask].mean(dim=0)
                
        # Estimate shared diagonal covariance
        cov_diag = torch.zeros(d, device=features.device)
        n_samples = features.size(0)
        
        for c in range(self.num_classes):
            mask = (labels == c)
            if mask.sum() > 0:
                diff = features[mask] - self.class_means[c]
                cov_diag += (diff ** 2).sum(dim=0)
                
        cov_diag = cov_diag / n_samples
        # Add 10^-6 to every diagonal entry
        cov_diag += 1e-6
        
        self.inv_cov_diag = 1.0 / cov_diag

    def compute_score(self, features):
        """
        u_Mah(x) = min_c (f(x) - \mu_c)^T \Sigma^{-1} (f(x) - \mu_c)
        """
        assert self.class_means is not None and self.inv_cov_diag is not None, "Must call fit() first."
        
        n = features.size(0)
        c = self.num_classes
        
        # features: (n, d)
        # class_means: (c, d)
        # inv_cov_diag: (d)
        
        # We can compute the distance for each class
        # distance[i, j] = sum_d (features[i, d] - class_means[j, d])^2 * inv_cov_diag[d]
        
        feat_expanded = features.unsqueeze(1) # (n, 1, d)
        means_expanded = self.class_means.unsqueeze(0) # (1, c, d)
        
        diff = feat_expanded - means_expanded # (n, c, d)
        dist = (diff ** 2) * self.inv_cov_diag.view(1, 1, -1)
        dist = dist.sum(dim=2) # (n, c)
        
        min_dist, _ = torch.min(dist, dim=1)
        return min_dist
