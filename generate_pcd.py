"""
Example code for generating point clouds from VAREN.

This code generates a point cloud from a random pose and saves it as a .ply file.

Example usage:
    python generate_pcd.py


"""

import torch
import trimesh

from src.varen_wrapper import VARENPcd

def main():
    varen = VARENPcd(model_path="/home/dperrett/Documents/Data/VAREN/models/VAREN")
    
    
    pose = (torch.rand(1, varen.NUM_JOINTS * 3) - 0.5) * 0.3
    pcd = varen.generate_pcd(body_pose = pose, num_samples=250000, noise_scale=0.005)
    


    pcd = pcd.squeeze().detach().cpu()

    mesh = trimesh.Trimesh(vertices=pcd, vertex_colors=[255, 0, 0])
    mesh.export("pcd.ply")



if __name__ == "__main__":
    main()