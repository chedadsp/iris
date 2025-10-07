#!/usr/bin/env python3
"""
Spatial Utilities for Efficient Point Cloud Operations

Provides optimized spatial operations including spatial indexing, efficient
filtering, and batched operations for large point clouds.

Author: Dimitrije Stojanovic
Date: September 2025
Created: Code Review Performance Fixes
"""

import numpy as np
from typing import Tuple, Optional, List, Union
from sklearn.neighbors import NearestNeighbors, KDTree
import time
from .config import AnalysisConfig, FileConfig
from .error_handling import with_error_handling, validate_point_cloud_data


class SpatialIndex:
    """Spatial indexing for efficient point cloud operations."""
    
    def __init__(self, points: np.ndarray, leaf_size: int = 30):
        """
        Initialize spatial index.
        
        Args:
            points: Point cloud array (N, 3)
            leaf_size: Leaf size for KDTree (affects build vs query time)
        """
        validate_point_cloud_data(points)
        self.points = points
        self.kdtree = KDTree(points, leaf_size=leaf_size)
        self.leaf_size = leaf_size
    
    @with_error_handling("spatial_index_query", reraise=False, fallback_result=(np.array([]), np.array([])))
    def query_radius(self, query_points: np.ndarray, radius: float) -> Tuple[List, List]:
        """
        Query points within radius using spatial index.
        
        Args:
            query_points: Query point locations (M, 3)
            radius: Search radius
            
        Returns:
            Tuple of (indices, distances) for each query point
        """
        return self.kdtree.query_radius(query_points, radius, return_distance=True)
    
    @with_error_handling("spatial_index_knn", reraise=False, fallback_result=(np.array([]), np.array([])))
    def query_knn(self, query_points: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Query k nearest neighbors using spatial index.
        
        Args:
            query_points: Query point locations (M, 3)
            k: Number of nearest neighbors
            
        Returns:
            Tuple of (distances, indices)
        """
        return self.kdtree.query(query_points, k=k, return_distance=True)
    
    def query_box(self, bounds: List[float]) -> np.ndarray:
        """
        Query points within axis-aligned bounding box.
        
        Args:
            bounds: [x_min, x_max, y_min, y_max, z_min, z_max]
            
        Returns:
            Boolean mask for points inside the box
        """
        x_min, x_max, y_min, y_max, z_min, z_max = bounds
        
        # Vectorized box query - much faster than individual checks
        mask = (
            (self.points[:, 0] >= x_min) & (self.points[:, 0] <= x_max) &
            (self.points[:, 1] >= y_min) & (self.points[:, 1] <= y_max) &
            (self.points[:, 2] >= z_min) & (self.points[:, 2] <= z_max)
        )
        
        return mask


class OptimizedPointCloudOps:
    """Optimized point cloud operations using spatial indexing."""
    
    def __init__(self, points: np.ndarray):
        """
        Initialize with point cloud.
        
        Args:
            points: Point cloud array (N, 3)
        """
        validate_point_cloud_data(points)
        self.points = points
        self.spatial_index = None
        self._build_index_if_needed()
    
    def _build_index_if_needed(self):
        """Build spatial index if point cloud is large enough to benefit."""
        if len(self.points) > FileConfig.LARGE_FILE_THRESHOLD // 10:  # 100K points
            print(f"Building spatial index for {len(self.points):,} points...")
            start_time = time.time()
            self.spatial_index = SpatialIndex(self.points)
            build_time = time.time() - start_time
            print(f"Spatial index built in {build_time:.2f} seconds")
    
    @with_error_handling("box_filter", reraise=False, fallback_result=np.array([], dtype=bool))
    def filter_box_optimized(self, bounds: List[float]) -> np.ndarray:
        """
        Optimized box filtering using spatial index when available.
        
        Args:
            bounds: [x_min, x_max, y_min, y_max, z_min, z_max]
            
        Returns:
            Boolean mask for points inside the box
        """
        if self.spatial_index:
            return self.spatial_index.query_box(bounds)
        else:
            # Fallback to vectorized numpy operations
            x_min, x_max, y_min, y_max, z_min, z_max = bounds
            return (
                (self.points[:, 0] >= x_min) & (self.points[:, 0] <= x_max) &
                (self.points[:, 1] >= y_min) & (self.points[:, 1] <= y_max) &
                (self.points[:, 2] >= z_min) & (self.points[:, 2] <= z_max)
            )
    
    @with_error_handling("radius_neighbors", reraise=False, fallback_result=[])
    def find_neighbors_in_radius(self, query_points: np.ndarray, 
                                radius: float) -> List[np.ndarray]:
        """
        Find neighbors within radius for multiple query points.
        
        Args:
            query_points: Query point locations (M, 3)
            radius: Search radius
            
        Returns:
            List of neighbor indices for each query point
        """
        if self.spatial_index:
            indices, _ = self.spatial_index.query_radius(query_points, radius)
            return indices
        else:
            # Fallback using sklearn NearestNeighbors
            nbrs = NearestNeighbors(radius=radius)
            nbrs.fit(self.points)
            indices = nbrs.radius_neighbors(query_points, return_distance=False)
            return indices
    
    @with_error_handling("knn_search", reraise=False, fallback_result=(np.array([]), np.array([])))
    def find_k_nearest_neighbors(self, query_points: np.ndarray, 
                                k: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Find k nearest neighbors for multiple query points.
        
        Args:
            query_points: Query point locations (M, 3)
            k: Number of nearest neighbors
            
        Returns:
            Tuple of (distances, indices)
        """
        k = min(k, len(self.points))  # Ensure k doesn't exceed available points
        
        if self.spatial_index:
            return self.spatial_index.query_knn(query_points, k)
        else:
            # Fallback using sklearn NearestNeighbors
            nbrs = NearestNeighbors(n_neighbors=k)
            nbrs.fit(self.points)
            return nbrs.kneighbors(query_points)
    
    def compute_local_density(self, radius: float, 
                            sample_points: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Compute local point density efficiently.
        
        Args:
            radius: Radius for density computation
            sample_points: Points to compute density for (defaults to all points)
            
        Returns:
            Array of local density values
        """
        if sample_points is None:
            sample_points = self.points
        
        neighbor_lists = self.find_neighbors_in_radius(sample_points, radius)
        densities = np.array([len(neighbors) for neighbors in neighbor_lists])
        
        return densities.astype(np.float32)
    
    def downsample_uniform(self, target_points: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Uniform downsampling to target number of points.
        
        Args:
            target_points: Target number of points after downsampling
            
        Returns:
            Tuple of (downsampled_points, original_indices)
        """
        if len(self.points) <= target_points:
            return self.points.copy(), np.arange(len(self.points))
        
        # Calculate stride for uniform sampling
        stride = len(self.points) // target_points
        indices = np.arange(0, len(self.points), stride)[:target_points]
        
        return self.points[indices], indices
    
    def remove_outliers_statistical(self, k: int = 20, 
                                  std_ratio: float = 2.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Remove statistical outliers based on distance to neighbors.
        
        Args:
            k: Number of neighbors to consider
            std_ratio: Standard deviation ratio threshold
            
        Returns:
            Tuple of (filtered_points, inlier_mask)
        """
        print(f"Removing statistical outliers using {k} neighbors...")
        
        # Find k nearest neighbors for all points
        distances, _ = self.find_k_nearest_neighbors(self.points, k + 1)  # +1 to exclude self
        
        # Calculate mean distance to neighbors (excluding self at index 0)
        mean_distances = np.mean(distances[:, 1:], axis=1)
        
        # Statistical outlier detection
        mean_dist = np.mean(mean_distances)
        std_dist = np.std(mean_distances)
        threshold = mean_dist + std_ratio * std_dist
        
        inlier_mask = mean_distances <= threshold
        
        print(f"Removed {np.sum(~inlier_mask):,} outliers ({np.sum(~inlier_mask)/len(self.points)*100:.1f}%)")
        
        return self.points[inlier_mask], inlier_mask


class BatchProcessor:
    """Process large point clouds in batches to manage memory usage."""
    
    def __init__(self, batch_size: int = 50000):
        """
        Initialize batch processor.
        
        Args:
            batch_size: Number of points per batch
        """
        self.batch_size = batch_size
    
    def process_in_batches(self, points: np.ndarray, 
                          operation_func, *args, **kwargs) -> List:
        """
        Process point cloud in batches.
        
        Args:
            points: Input point cloud
            operation_func: Function to apply to each batch
            *args, **kwargs: Arguments for the operation function
            
        Returns:
            List of results from each batch
        """
        n_points = len(points)
        n_batches = (n_points + self.batch_size - 1) // self.batch_size
        
        print(f"Processing {n_points:,} points in {n_batches} batches of {self.batch_size:,}")
        
        results = []
        for i in range(n_batches):
            start_idx = i * self.batch_size
            end_idx = min((i + 1) * self.batch_size, n_points)
            
            batch = points[start_idx:end_idx]
            batch_result = operation_func(batch, *args, **kwargs)
            results.append(batch_result)
            
            if i % 10 == 0:  # Progress update every 10 batches
                print(f"Processed batch {i + 1}/{n_batches}")
        
        return results
    
    def combine_batch_results(self, results: List, 
                            combine_method: str = "concatenate") -> Union[np.ndarray, List]:
        """
        Combine results from batch processing.
        
        Args:
            results: List of batch results
            combine_method: How to combine results ("concatenate", "sum", "list")
            
        Returns:
            Combined result
        """
        if not results:
            return np.array([])
        
        if combine_method == "concatenate":
            return np.concatenate(results, axis=0)
        elif combine_method == "sum":
            return sum(results)
        elif combine_method == "list":
            return results
        else:
            raise ValueError(f"Unknown combine method: {combine_method}")


def optimize_point_cloud_filtering(points: np.ndarray, bounds: List[float],
                                 use_spatial_index: bool = True) -> np.ndarray:
    """
    Optimized point cloud filtering with automatic method selection.
    
    Args:
        points: Point cloud array (N, 3)
        bounds: [x_min, x_max, y_min, y_max, z_min, z_max]
        use_spatial_index: Whether to use spatial indexing for large clouds
        
    Returns:
        Boolean mask for points inside bounds
    """
    if len(points) < 10000 or not use_spatial_index:
        # For small point clouds, simple numpy operations are faster
        x_min, x_max, y_min, y_max, z_min, z_max = bounds
        return (
            (points[:, 0] >= x_min) & (points[:, 0] <= x_max) &
            (points[:, 1] >= y_min) & (points[:, 1] <= y_max) &
            (points[:, 2] >= z_min) & (points[:, 2] <= z_max)
        )
    else:
        # For large point clouds, use optimized operations
        ops = OptimizedPointCloudOps(points)
        return ops.filter_box_optimized(bounds)


if __name__ == "__main__":
    print("Spatial Utilities Module Test")
    
    # Generate test data
    np.random.seed(42)
    n_points = 100000
    points = np.random.randn(n_points, 3) * 10
    
    print(f"Testing with {n_points:,} points")
    
    # Test spatial index
    start_time = time.time()
    ops = OptimizedPointCloudOps(points)
    index_time = time.time() - start_time
    print(f"Index creation: {index_time:.3f} seconds")
    
    # Test box filtering
    bounds = [-5, 5, -5, 5, -2, 2]
    start_time = time.time()
    mask = ops.filter_box_optimized(bounds)
    filter_time = time.time() - start_time
    print(f"Box filtering: {filter_time:.3f} seconds, {np.sum(mask):,} points selected")
    
    # Test KNN
    query_points = np.random.randn(100, 3) * 5
    start_time = time.time()
    distances, indices = ops.find_k_nearest_neighbors(query_points, 10)
    knn_time = time.time() - start_time
    print(f"KNN search: {knn_time:.3f} seconds")
    
    print("Spatial utilities tests completed successfully!")