from varen import VAREN
import kaolin as kal
import torch

class VARENPcd(VAREN):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.pcd = None
    
    def generate_pcd(self, num_samples=10000, noise_scale=0.001, distance_threshold=.15, **kwargs):
        vertices = self.forward(**kwargs).vertices
        faces = torch.tensor(self.faces)

        batch_size = vertices.shape[0]
        sampled_points, _ = kal.ops.mesh.sample_points(vertices, faces, num_samples, areas=None, face_features=None)

        noise = torch.randn_like(sampled_points) * noise_scale

        sampled_points += noise
        # Select a random point
        n_patches = 10
        random_indices = torch.randint(low=0, high=sampled_points.shape[1], size=(batch_size, n_patches))  # Shape: (B, 15)
        batch_indices = torch.arange(batch_size).unsqueeze(1).expand(batch_size, n_patches)  # Shape: [B, K]

        # Perform multi-indexing
        random_points = sampled_points[batch_indices, random_indices]  # Shape: [B, K, 3]  

        # Calculate distances from the random point to all other points
        l2_dist = torch.norm(sampled_points[:, :, None, :] - random_points[:, None, :, :], dim=-1)  # Shape: [B, N, K]     
        cheb_dist = torch.max(torch.abs(sampled_points[:, :, None, :] - random_points[:, None, :, :]), dim=-1)[0]  # [B, N, K]

        distances = torch.mean(torch.stack([cheb_dist, l2_dist], dim=0),dim=0)  # [B, N, K]
        mask = distances > distance_threshold

        mask = mask.all(dim=-1)
        # Filter points

        filtered_points = sampled_points[mask]

        return filtered_points 
