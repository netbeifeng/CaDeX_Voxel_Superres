import numpy as np


# Transforms
class PointcloudNoise(object):
    """Point cloud noise transformation class.

    It adds noise to point cloud data.

    Args:
        stddev (int): standard deviation
    """

    def __init__(self, stddev):
        self.stddev = stddev

    def __call__(self, data):
        """Calls the transformation.

        Args:
            data (dictionary): data dictionary
        """
        data_out = data.copy()
        points = data[None]
        noise = self.stddev * np.random.randn(*points.shape)
        noise = noise.astype(np.float32)
        data_out[None] = points + noise
        return data_out


class SubsamplePointcloud(object):
    """Point cloud subsampling transformation class.

    It subsamples the point cloud data.

    Args:
        N (int): number of points to be subsampled
    """

    def __init__(self, N):
        self.N = N

    def __call__(self, data):
        """Calls the transformation.

        Args:
            data (dict): data dictionary
        """
        data_out = data.copy()
        points = data[None]

        indices = np.random.randint(points.shape[0], size=self.N)
        data_out[None] = points[indices, :]

        if "normals" in data.keys():
            normals = data["normals"]
            data_out["normals"] = normals[indices, :]

        return data_out


class SubsamplePoints(object):
    """Points subsampling transformation class.

    It subsamples the points data.

    Args:
        N (int): number of points to be subsampled
    """

    def __init__(self, N):
        self.N = N

    def __call__(self, data):
        """Calls the transformation.

        Args:
            data (dictionary): data dictionary
        """
        points = data[None]
        occ = data["occ"]

        data_out = data.copy()
        if isinstance(self.N, int):
            if points.ndim == 2:
                idx = np.random.randint(points.shape[0], size=self.N)
                data_out.update(
                    {
                        None: points[idx, :],
                        "occ": occ[idx],
                    }
                )
            elif points.ndim == 3:
                points_list, occ_list = [], []
                for tdx in range(points.shape[0]):
                    idx = np.random.randint(points.shape[1], size=self.N)
                    points_list.append(points[tdx, idx, :][np.newaxis, ...])
                    occ_list.append(occ[tdx, idx][np.newaxis, ...])
                data_out.update(
                    {
                        None: np.concatenate(points_list, axis=0),
                        "occ": np.concatenate(occ_list, axis=0),
                    }
                )
                pass
            else:
                raise RuntimeError("Data loader not support 2020.12.29")
        else:
            Nt_out, Nt_in = self.N
            occ_binary = occ >= 0.5
            points0 = points[~occ_binary]
            points1 = points[occ_binary]

            idx0 = np.random.randint(points0.shape[0], size=Nt_out)
            idx1 = np.random.randint(points1.shape[0], size=Nt_in)

            points0 = points0[idx0, :]
            points1 = points1[idx1, :]
            points = np.concatenate([points0, points1], axis=0)

            occ0 = np.zeros(Nt_out, dtype=np.float32)
            occ1 = np.ones(Nt_in, dtype=np.float32)
            occ = np.concatenate([occ0, occ1], axis=0)

            volume = occ_binary.sum() / len(occ_binary)
            volume = volume.astype(np.float32)

            data_out.update(
                {
                    None: points,
                    "occ": occ,
                    "volume": volume,
                }
            )
        return data_out

import torch
from torch_cluster import fps

class VoxelSeq(object):
    
    def __init__(self, voxel_res, xyz_range, xyz_padding):
        self.res = voxel_res
        self.xyz_range = np.array(xyz_range)
        self.xyz_padding = xyz_padding
        self.size = (self.xyz_range[3:] - self.xyz_range[:3]) / self.res
        self.centers = np.zeros((self.res, self.res, self.res, 3), dtype=np.float32)
        grid_indices = np.indices((self.res, self.res, self.res)).transpose(1, 2, 3, 0)
        self.centers = self.xyz_range[:3] + (grid_indices + 0.5) * self.size
        self.sampling = 512
    def __call__(self, data):
        # print("VoxelSeq")
        # print(data.keys())
        points = data[None]
        # print(points.shape, data['time'])
        # print(self.get_voxel_grid(points).shape)
        # raise
        data_out = data.copy()
        data_out[None] = self.get_voxel_grid(points)
        data_out["centers"] = self.get_active_voxel_centers(data_out[None])
        return data_out
    
    def get_voxel_grid(self, surface):
        B, N, _ = surface.shape
        surface = torch.from_numpy(surface)
        indices = ((surface - self.xyz_range[:3]) / self.size).long()
        indices = torch.clamp(indices, min=0, max=self.res - 1) # [B, N, 3]
        voxel_grid = torch.zeros((B, 1, self.res, self.res, self.res), dtype=torch.int64)

        increment = torch.ones((B, N), dtype=torch.int64)
        flat_indices = indices[:, :, 0] * self.res**2 + indices[:, :, 1] * self.res + indices[:, :, 2]
        # point_to_voxel = flat_indices.view(B, -1)
        voxel_grid.view(B, -1).scatter_add_(1, flat_indices.view(B, -1), increment)
        voxel_grid = voxel_grid.view(B, 1, self.res, self.res, self.res)
        voxel_grid = (voxel_grid > 0).float()
        return voxel_grid
    
    def get_active_voxel_centers(self, voxel_grid):
        B, _, D, H, W = voxel_grid.shape

        # List to hold centers for each batch
        all_centers = []

        for b in range(B):
            active_indices = (voxel_grid[b].squeeze(0)).nonzero(as_tuple=False)
            # print(active_indices.shape)
            # ....
            # get all centers first apply fps to get sampled points
            all_active_centers = self.centers[
                active_indices[:, 0], 
                active_indices[:, 1], 
                active_indices[:, 2]
            ].reshape(-1, 3)
            all_active_centers = torch.from_numpy(all_active_centers).to(voxel_grid.device)
            
            if active_indices.shape[0] < self.sampling:
                sampled_indices = torch.randint(0, active_indices.shape[0], (self.sampling,), dtype=torch.long)
            else:
                sampled_indices = fps(all_active_centers, ratio=self.sampling/active_indices.shape[0])
                
            sampled_centers = all_active_centers[sampled_indices]
            all_centers.append(sampled_centers)
            
        active_centers = torch.stack(all_centers, dim=0).float().to(voxel_grid.device)

        return active_centers

class SubsamplePointcloudSeq(object):
    """Point cloud sequence subsampling transformation class.

    It subsamples the point cloud sequence data.

    Args:
        N (int): number of points to be subsampled
        connected_samples (bool): whether to obtain connected samples
        random (bool): whether to sub-sample randomly
    """

    def __init__(self, N, connected_samples=False, random=True):
        self.N = N
        self.connected_samples = connected_samples
        self.random = random

    def __call__(self, data):
        """Calls the transformation.

        Args:
            data (dictionary): data dictionary
        """
        data_out = data.copy()
        points = data[None]  # n_steps x T x 3
        n_steps, T, dim = points.shape
        N_max = min(self.N, T)
        if self.connected_samples or not self.random:
            indices = np.random.randint(T, size=self.N) if self.random else np.arange(N_max)
            data_out[None] = points[:, indices, :]
        else:
            indices = np.random.randint(T, size=(n_steps, self.N))
            data_out[None] = points[np.arange(n_steps).reshape(-1, 1), indices, :]
        return data_out


class SubsamplePointsSeq(object):
    """Points sequence subsampling transformation class.

    It subsamples the points sequence data.

    Args:
        N (int): number of points to be subsampled
        connected_samples (bool): whether to obtain connected samples
        random (bool): whether to sub-sample randomly
    """

    def __init__(self, N, connected_samples=False, random=True):
        self.N = N
        self.connected_samples = connected_samples
        self.random = random

    def __call__(self, data):
        """Calls the transformation.

        Args:
            data (dictionary): data dictionary
        """
        points = data[None]
        occ = data["occ"]
        data_out = data.copy()
        n_steps, T, dim = points.shape

        N_max = min(self.N, T)

        if self.connected_samples or not self.random:
            indices = np.random.randint(T, size=self.N) if self.random else np.arange(N_max)
            data_out.update(
                {
                    None: points[:, indices],
                    "occ": occ[:, indices],
                }
            )
        else:
            indices = np.random.randint(T, size=(n_steps, self.N))
            help_arr = np.arange(n_steps).reshape(-1, 1)
            data_out.update({None: points[help_arr, indices, :], "occ": occ[help_arr, indices, :]})
        return data_out
