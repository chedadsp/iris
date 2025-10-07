#!/usr/bin/env python3
"""
PointTransformerV3 Integration for Vehicle Point Cloud Analysis

This module provides PointTransformerV3-based deep learning enhancements
for the existing E57 vehicle analysis pipeline.

Key Features:
- Point cloud semantic segmentation using PointTransformerV3
- Vehicle vs non-vehicle classification
- Interior vs exterior point classification
- Integration with existing E57 analysis workflow

Author: Dimitrije Stojanovic
Date: September 2025
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from pathlib import Path
import warnings

# Core dependencies
try:
    import torch
    import torch.nn as nn
    from einops import rearrange
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    warnings.warn("PyTorch dependencies not available. PTv3 functionality disabled.")

from .error_handling import with_error_handling, PointCloudError


class PointTransformerV3Block(nn.Module):
    """
    Simplified PointTransformerV3 block for point cloud processing.

    This is a minimal implementation focusing on the key transformer
    components without requiring the full Pointcept framework.
    """

    def __init__(self, in_channels: int, out_channels: int, num_heads: int = 8):
        super().__init__()
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch not available for PointTransformerV3")

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_heads = num_heads

        # Linear projections for attention
        self.to_qkv = nn.Linear(in_channels, out_channels * 3)
        self.attention_output = nn.Linear(out_channels, out_channels)

        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(out_channels, out_channels * 4),
            nn.ReLU(),
            nn.Linear(out_channels * 4, out_channels)
        )

        # Layer normalization
        self.norm1 = nn.LayerNorm(out_channels)
        self.norm2 = nn.LayerNorm(out_channels)

    def forward(self, x: torch.Tensor, coordinates: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the transformer block.

        Args:
            x: Point features [N, in_channels]
            coordinates: Point coordinates [N, 3] (unused in this simplified version)

        Returns:
            Enhanced point features [N, out_channels]
        """
        # Generate Q, K, V
        qkv = self.to_qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)

        # Simplified self-attention (without position encoding)
        attention_weights = torch.softmax(torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(self.out_channels), dim=-1)
        attention_output = torch.matmul(attention_weights, v)
        attention_output = self.attention_output(attention_output)

        # Residual connection and layer norm
        x_proj = nn.Linear(self.in_channels, self.out_channels).to(x.device)(x) if self.in_channels != self.out_channels else x
        x = self.norm1(x_proj + attention_output)

        # Feed-forward network with residual connection
        x = self.norm2(x + self.ffn(x))

        return x


class VehiclePointClassifier(nn.Module):
    """
    Vehicle-specific point cloud classifier using PointTransformerV3 architecture.

    This model classifies points into semantic categories relevant for vehicle analysis:
    - Background/Ground
    - Vehicle Exterior
    - Vehicle Interior (empty)
    - Human Body Parts (head, torso, limbs)
    - Vehicle Seats
    - Other Objects
    """

    def __init__(self, input_channels: int = 6, num_classes: int = 6, hidden_dim: int = 128):
        super().__init__()
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch not available for VehiclePointClassifier")

        self.input_channels = input_channels  # XYZ + RGB or XYZ + intensity + normals
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim

        # Input projection
        self.input_projection = nn.Linear(input_channels, hidden_dim)

        # Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            PointTransformerV3Block(hidden_dim, hidden_dim) for _ in range(3)
        ])

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, points: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for point classification.

        Args:
            points: Point coordinates [N, 3]
            features: Point features [N, input_channels]

        Returns:
            Class logits [N, num_classes]
        """
        # Project input features
        x = self.input_projection(features)

        # Pass through transformer blocks
        for block in self.transformer_blocks:
            x = block(x, points)

        # Classification
        logits = self.classifier(x)

        return logits


class PTv3VehicleAnalyzer:
    """
    PointTransformerV3-enhanced vehicle analysis integration.

    This class integrates PointTransformerV3 capabilities with the existing
    E57 vehicle analysis pipeline to provide enhanced semantic understanding.
    """

    def __init__(self, device: str = 'auto'):
        """
        Initialize the PTv3 analyzer.

        Args:
            device: Computing device ('auto', 'cpu', 'cuda', 'mps')
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch dependencies not available")

        self.device = self._setup_device(device)
        self.model = None
        self.is_trained = False

        # Class mapping for vehicle analysis with human detection
        self.class_names = {
            0: "ground",
            1: "vehicle_exterior",
            2: "vehicle_interior_empty",
            3: "human_body",
            4: "vehicle_seats",
            5: "other_objects"
        }

    def _setup_device(self, device: str) -> torch.device:
        """Setup the computing device."""
        if device == 'auto':
            if torch.cuda.is_available():
                return torch.device('cuda')
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return torch.device('mps')  # Apple Silicon GPU
            else:
                return torch.device('cpu')
        else:
            return torch.device(device)

    def _estimate_memory_usage(self, n_points: int, batch_size: int) -> float:
        """
        Estimate memory usage in GB for processing a batch.

        Args:
            n_points: Total number of points
            batch_size: Points per batch

        Returns:
            Estimated memory usage in GB
        """
        points_per_batch = min(batch_size, n_points)

        # Estimate memory for tensors:
        # - coordinates: [N, 3] * 4 bytes (float32)
        # - features: [N, 6] * 4 bytes (float32)
        # - logits: [N, 6] * 4 bytes (float32)
        # - probabilities: [N, 6] * 4 bytes (float32)
        # Total per point: (3 + 6 + 6 + 6) * 4 = 84 bytes

        bytes_per_point = 84
        total_bytes = points_per_batch * bytes_per_point

        # Add overhead for model parameters and intermediate calculations (2x multiplier)
        total_bytes *= 2

        # Convert to GB
        gb = total_bytes / (1024 ** 3)

        return gb

    def _get_available_memory_gb(self) -> float:
        """
        Get available memory on the device in GB.

        Returns:
            Available memory in GB
        """
        try:
            if self.device.type == 'cuda':
                # Get CUDA memory info
                torch.cuda.synchronize()
                free_memory = torch.cuda.mem_get_info()[0]
                return free_memory / (1024 ** 3)
            elif self.device.type == 'mps':
                # MPS doesn't have direct memory query, estimate conservatively
                # Assume 8GB available on typical Apple Silicon Macs
                return 4.0  # Conservative estimate
            else:
                # CPU - use system memory
                import psutil
                return psutil.virtual_memory().available / (1024 ** 3)
        except Exception:
            # Fallback to conservative estimate
            return 2.0

    def _chunk_large_pointcloud(self, points: np.ndarray, colors: Optional[np.ndarray] = None,
                               intensity: Optional[np.ndarray] = None,
                               max_chunk_size: int = 50000) -> List[Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]]:
        """
        Split large point clouds into manageable chunks.

        Args:
            points: Point coordinates [N, 3]
            colors: RGB colors [N, 3] (optional)
            intensity: Intensity values [N, 1] (optional)
            max_chunk_size: Maximum points per chunk

        Returns:
            List of (points_chunk, colors_chunk, intensity_chunk) tuples
        """
        n_points = points.shape[0]
        chunks = []

        for i in range(0, n_points, max_chunk_size):
            end_idx = min(i + max_chunk_size, n_points)
            points_chunk = points[i:end_idx]
            colors_chunk = colors[i:end_idx] if colors is not None else None
            intensity_chunk = intensity[i:end_idx] if intensity is not None else None
            chunks.append((points_chunk, colors_chunk, intensity_chunk))

        return chunks

    @with_error_handling()
    def process_large_pointcloud(self, points: np.ndarray, colors: Optional[np.ndarray] = None,
                                intensity: Optional[np.ndarray] = None) -> Dict[str, np.ndarray]:
        """
        Automatically process large point clouds with optimal memory management.

        This method automatically determines the best processing strategy based on:
        - Dataset size
        - Available memory
        - Device capabilities

        Args:
            points: Point coordinates [N, 3]
            colors: RGB colors [N, 3] (optional)
            intensity: Intensity values [N, 1] (optional)

        Returns:
            Dictionary with enhanced vehicle detection results
        """
        n_points = points.shape[0]
        available_memory = self._get_available_memory_gb()

        print(f"🔍 Processing {n_points:,} points on {self.device}")
        print(f"📊 Available memory: {available_memory:.1f} GB")

        # Determine processing strategy
        if n_points < 20000:
            # Small dataset - process normally
            print("✅ Small dataset: using standard processing")
            return self.enhance_vehicle_detection(points, colors, intensity, batch_size=n_points)

        elif n_points < 100000:
            # Medium dataset - use batch processing
            estimated_memory = self._estimate_memory_usage(n_points, 10000)
            if estimated_memory < available_memory * 0.8:  # Use 80% of available memory
                print("✅ Medium dataset: using batch processing")
                return self.enhance_vehicle_detection(points, colors, intensity, batch_size=10000)
            else:
                print("⚠️  Medium dataset with limited memory: using smaller batches")
                safe_batch_size = max(1000, int(10000 * available_memory * 0.8 / estimated_memory))
                return self.enhance_vehicle_detection(points, colors, intensity, batch_size=safe_batch_size)

        else:
            # Large dataset - use chunking
            print("⚠️  Large dataset: using chunked processing")
            chunk_size = min(50000, int(available_memory * 0.6 * 1024**3 / 84))  # Conservative chunk size
            chunks = self._chunk_large_pointcloud(points, colors, intensity, max_chunk_size=chunk_size)

            print(f"📦 Processing {len(chunks)} chunks of ~{chunk_size:,} points each")

            # Process each chunk and combine results
            all_results = []
            for i, (chunk_points, chunk_colors, chunk_intensity) in enumerate(chunks):
                print(f"  📦 Processing chunk {i+1}/{len(chunks)}")
                chunk_result = self.enhance_vehicle_detection(
                    chunk_points, chunk_colors, chunk_intensity, batch_size=5000
                )
                all_results.append(chunk_result)

                # Clear memory after each chunk
                if self.device.type in ['cuda', 'mps']:
                    if self.device.type == 'cuda':
                        torch.cuda.empty_cache()
                    elif hasattr(torch.mps, 'empty_cache'):
                        torch.mps.empty_cache()

            # Combine chunk results
            combined_result = self._combine_chunk_results(all_results)
            print("✅ Chunked processing completed")
            return combined_result

    def _combine_chunk_results(self, chunk_results: List[Dict[str, np.ndarray]]) -> Dict[str, np.ndarray]:
        """Combine results from multiple chunks."""
        if not chunk_results:
            return {}

        def safe_vstack(arrays_list, shape_2d=(0, 3)):
            """Safely stack arrays, returning empty array if no valid arrays."""
            valid_arrays = [arr for arr in arrays_list if len(arr) > 0]
            if valid_arrays:
                return np.vstack(valid_arrays)
            else:
                return np.array([]).reshape(shape_2d)

        def safe_concatenate(arrays_list, shape_1d=(0,)):
            """Safely concatenate arrays, returning empty array if no valid arrays."""
            valid_arrays = [arr for arr in arrays_list if len(arr) > 0]
            if valid_arrays:
                return np.concatenate(valid_arrays)
            else:
                return np.array([]).reshape(shape_1d)

        combined = {
            'ground_points': safe_vstack([r['ground_points'] for r in chunk_results]),
            'vehicle_exterior_points': safe_vstack([r['vehicle_exterior_points'] for r in chunk_results]),
            'vehicle_interior_points': safe_vstack([r['vehicle_interior_points'] for r in chunk_results]),
            'vehicle_all_points': safe_vstack([r['vehicle_all_points'] for r in chunk_results]),
            'other_points': safe_vstack([r['other_points'] for r in chunk_results]),
            'semantic_labels': safe_concatenate([r['semantic_labels'] for r in chunk_results]),
            'semantic_probabilities': safe_concatenate([r['semantic_probabilities'] for r in chunk_results],
                                                      shape_1d=(0, len(self.class_names))),
        }

        # Combine masks
        combined['masks'] = {}
        for mask_name in ['ground', 'vehicle_exterior', 'vehicle_interior', 'vehicle_all', 'other']:
            combined['masks'][mask_name] = safe_concatenate([r['masks'][mask_name] for r in chunk_results])

        return combined

    @with_error_handling()
    def initialize_model(self, input_channels: int = 6, num_classes: int = 6) -> None:
        """
        Initialize the PointTransformerV3 model.

        Args:
            input_channels: Number of input feature channels
            num_classes: Number of semantic classes
        """
        self.model = VehiclePointClassifier(
            input_channels=input_channels,
            num_classes=num_classes
        ).to(self.device)

        # Initialize weights
        self._initialize_weights()

        print(f"PTv3 model initialized on device: {self.device}")

    def _initialize_weights(self):
        """Initialize model weights."""
        for module in self.model.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    @with_error_handling()
    def prepare_point_features(self, points: np.ndarray, colors: Optional[np.ndarray] = None,
                             intensity: Optional[np.ndarray] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prepare point cloud data for PTv3 processing.

        Args:
            points: Point coordinates [N, 3]
            colors: RGB colors [N, 3] (optional)
            intensity: Intensity values [N, 1] (optional)

        Returns:
            Tuple of (coordinates, features) tensors
        """
        if points.shape[1] != 3:
            raise PointCloudError(f"Points must have 3 coordinates, got {points.shape[1]}")

        # Normalize coordinates
        coords = points.copy()
        coords = (coords - coords.mean(axis=0)) / (coords.std(axis=0) + 1e-8)

        # Prepare features
        features = [coords]  # XYZ coordinates as features

        if colors is not None:
            # Normalize colors to [0, 1]
            colors_norm = colors / 255.0 if colors.max() > 1.0 else colors
            features.append(colors_norm)
        elif intensity is not None:
            # Normalize intensity and repeat for 3 channels
            intensity_norm = (intensity - intensity.min()) / (intensity.max() - intensity.min() + 1e-8)
            features.append(np.repeat(intensity_norm, 3, axis=1))
        else:
            # Use dummy color features
            features.append(np.zeros((points.shape[0], 3)))

        # Concatenate all features
        features_array = np.concatenate(features, axis=1)

        # Convert to tensors
        coords_tensor = torch.from_numpy(points).float().to(self.device)
        features_tensor = torch.from_numpy(features_array).float().to(self.device)

        return coords_tensor, features_tensor

    @with_error_handling()
    def predict_point_semantics(self, points: np.ndarray, colors: Optional[np.ndarray] = None,
                               intensity: Optional[np.ndarray] = None,
                               batch_size: int = 10000) -> Dict[str, np.ndarray]:
        """
        Predict semantic labels for points using PTv3 with memory-efficient batch processing.

        Args:
            points: Point coordinates [N, 3]
            colors: RGB colors [N, 3] (optional)
            intensity: Intensity values [N, 1] (optional)
            batch_size: Number of points to process per batch (default: 10000)

        Returns:
            Dictionary with predictions and class probabilities
        """
        if self.model is None:
            self.initialize_model()

        n_points = points.shape[0]

        # Check memory requirements and adjust batch size if needed
        estimated_memory_gb = self._estimate_memory_usage(n_points, batch_size)
        if estimated_memory_gb > 2.0:  # Limit to 2GB per batch
            recommended_batch_size = max(1000, int(batch_size * 2.0 / estimated_memory_gb))
            print(f"⚠️  Large dataset detected ({n_points:,} points). Using batch size: {recommended_batch_size:,}")
            batch_size = recommended_batch_size

        # Process in batches
        all_predictions = []
        all_probabilities = []

        self.model.eval()
        with torch.no_grad():
            for i in range(0, n_points, batch_size):
                end_idx = min(i + batch_size, n_points)
                batch_points = points[i:end_idx]
                batch_colors = colors[i:end_idx] if colors is not None else None
                batch_intensity = intensity[i:end_idx] if intensity is not None else None

                # Prepare batch data
                coords, features = self.prepare_point_features(batch_points, batch_colors, batch_intensity)

                # Inference on batch
                logits = self.model(coords, features)
                probabilities = torch.softmax(logits, dim=-1)
                predictions = torch.argmax(probabilities, dim=-1)

                # Collect results
                all_predictions.append(predictions.cpu().numpy())
                all_probabilities.append(probabilities.cpu().numpy())

                # Clear GPU cache after each batch
                if self.device.type in ['cuda', 'mps']:
                    if self.device.type == 'cuda':
                        torch.cuda.empty_cache()
                    elif hasattr(torch.mps, 'empty_cache'):
                        torch.mps.empty_cache()

                # Progress indication for large datasets
                if n_points > 50000 and (i // batch_size) % 10 == 0:
                    progress = (end_idx / n_points) * 100
                    print(f"  Processing: {progress:.1f}% ({end_idx:,}/{n_points:,} points)")

        # Concatenate all batch results
        predictions_np = np.concatenate(all_predictions, axis=0)
        probabilities_np = np.concatenate(all_probabilities, axis=0)

        return {
            'predictions': predictions_np,
            'probabilities': probabilities_np,
            'class_names': self.class_names
        }

    @with_error_handling()
    def enhance_vehicle_detection(self, points: np.ndarray, colors: Optional[np.ndarray] = None,
                                 intensity: Optional[np.ndarray] = None,
                                 batch_size: int = 10000) -> Dict[str, np.ndarray]:
        """
        Enhanced vehicle detection using PTv3 semantic understanding.

        Args:
            points: Point coordinates [N, 3]
            colors: RGB colors [N, 3] (optional)
            intensity: Intensity values [N, 1] (optional)
            batch_size: Number of points to process per batch (default: 10000)

        Returns:
            Dictionary with enhanced vehicle detection results
        """
        # Get semantic predictions with batch processing
        semantics = self.predict_point_semantics(points, colors, intensity, batch_size=batch_size)
        predictions = semantics['predictions']

        # Extract different point categories
        ground_mask = predictions == 0
        vehicle_exterior_mask = predictions == 1
        vehicle_interior_mask = predictions == 2
        other_mask = predictions == 3

        # Combine vehicle points (exterior + interior)
        vehicle_mask = vehicle_exterior_mask | vehicle_interior_mask

        return {
            'ground_points': points[ground_mask],
            'vehicle_exterior_points': points[vehicle_exterior_mask],
            'vehicle_interior_points': points[vehicle_interior_mask],
            'vehicle_all_points': points[vehicle_mask],
            'other_points': points[other_mask],
            'semantic_labels': predictions,
            'semantic_probabilities': semantics['probabilities'],
            'masks': {
                'ground': ground_mask,
                'vehicle_exterior': vehicle_exterior_mask,
                'vehicle_interior': vehicle_interior_mask,
                'vehicle_all': vehicle_mask,
                'other': other_mask
            }
        }

    @with_error_handling()
    def detect_humans_in_vehicle(self, vehicle_interior_points: np.ndarray,
                                colors: Optional[np.ndarray] = None,
                                intensity: Optional[np.ndarray] = None,
                                batch_size: int = 10000) -> Dict[str, np.ndarray]:
        """
        Detect humans specifically within vehicle interior points.

        Args:
            vehicle_interior_points: Points known to be inside the vehicle [N, 3]
            colors: RGB colors [N, 3] (optional)
            intensity: Intensity values [N, 1] (optional)

        Returns:
            Dictionary with human detection results
        """
        if len(vehicle_interior_points) == 0:
            return {
                'human_points': np.array([]).reshape(0, 3),
                'seat_points': np.array([]).reshape(0, 3),
                'interior_empty_points': np.array([]).reshape(0, 3),
                'human_probability': np.array([]),
                'human_detected': False,
                'estimated_human_count': 0,
                'human_regions': []
            }

        # Get semantic predictions for interior points with batch processing
        semantics = self.predict_point_semantics(vehicle_interior_points, colors, intensity, batch_size=batch_size)
        predictions = semantics['predictions']
        probabilities = semantics['probabilities']

        # Extract human-related points
        human_mask = predictions == 3  # human_body class
        seat_mask = predictions == 4   # vehicle_seats class
        interior_empty_mask = predictions == 2  # vehicle_interior_empty class

        human_points = vehicle_interior_points[human_mask]
        seat_points = vehicle_interior_points[seat_mask]
        interior_empty_points = vehicle_interior_points[interior_empty_mask]

        # Calculate human detection confidence
        human_confidence = probabilities[human_mask, 3] if len(human_points) > 0 else np.array([])
        avg_human_confidence = np.mean(human_confidence) if len(human_confidence) > 0 else 0.0

        # Estimate number of humans using clustering
        estimated_count, human_regions = self._estimate_human_count(human_points)

        # Determine if humans are detected (threshold-based)
        human_detected = len(human_points) > 50 and avg_human_confidence > 0.3

        return {
            'human_points': human_points,
            'seat_points': seat_points,
            'interior_empty_points': interior_empty_points,
            'human_probability': human_confidence,
            'human_detected': human_detected,
            'estimated_human_count': estimated_count,
            'human_regions': human_regions,
            'average_confidence': avg_human_confidence,
            'total_interior_points': len(vehicle_interior_points),
            'human_percentage': len(human_points) / len(vehicle_interior_points) * 100 if len(vehicle_interior_points) > 0 else 0
        }

    def _estimate_human_count(self, human_points: np.ndarray) -> Tuple[int, List[np.ndarray]]:
        """
        Estimate the number of humans by clustering human points.

        Args:
            human_points: Points classified as human body parts

        Returns:
            Tuple of (estimated_count, list_of_human_regions)
        """
        if len(human_points) < 10:
            return 0, []

        try:
            from sklearn.cluster import DBSCAN

            # Use DBSCAN to cluster human points into individual humans
            # Parameters tuned for typical human dimensions in a car
            clustering = DBSCAN(eps=0.3, min_samples=10).fit(human_points)
            labels = clustering.labels_

            # Count clusters (excluding noise labeled as -1)
            unique_labels = set(labels)
            if -1 in unique_labels:
                unique_labels.remove(-1)  # Remove noise cluster

            human_regions = []
            for label in unique_labels:
                cluster_mask = labels == label
                cluster_points = human_points[cluster_mask]
                if len(cluster_points) > 20:  # Minimum points for a human
                    human_regions.append(cluster_points)

            return len(human_regions), human_regions

        except Exception as e:
            print(f"Warning: Human count estimation failed: {e}")
            return 1 if len(human_points) > 50 else 0, []

    @with_error_handling()
    def analyze_human_pose(self, human_points: np.ndarray) -> Dict[str, any]:
        """
        Analyze human pose and position within the vehicle.

        Args:
            human_points: Points classified as human body parts

        Returns:
            Dictionary with pose analysis results
        """
        if len(human_points) == 0:
            return {
                'pose_detected': False,
                'head_position': None,
                'torso_position': None,
                'sitting_position': None,
                'body_orientation': None
            }

        # Calculate centroid and bounding box
        centroid = np.mean(human_points, axis=0)
        min_bounds = np.min(human_points, axis=0)
        max_bounds = np.max(human_points, axis=0)

        # Estimate head position (highest Z points)
        z_threshold = np.percentile(human_points[:, 2], 85)  # Top 15% in height
        head_candidates = human_points[human_points[:, 2] >= z_threshold]
        head_position = np.mean(head_candidates, axis=0) if len(head_candidates) > 0 else None

        # Estimate torso position (middle region)
        z_min, z_max = min_bounds[2], max_bounds[2]
        torso_z_range = [z_min + 0.3 * (z_max - z_min), z_min + 0.7 * (z_max - z_min)]
        torso_mask = (human_points[:, 2] >= torso_z_range[0]) & (human_points[:, 2] <= torso_z_range[1])
        torso_candidates = human_points[torso_mask]
        torso_position = np.mean(torso_candidates, axis=0) if len(torso_candidates) > 0 else None

        # Determine sitting position (front/back of vehicle)
        # Assuming vehicle oriented along X-axis
        x_position = "front" if centroid[0] < 2.0 else "back"

        # Estimate body orientation using PCA
        body_orientation = self._estimate_body_orientation(human_points)

        return {
            'pose_detected': True,
            'head_position': head_position,
            'torso_position': torso_position,
            'sitting_position': x_position,
            'body_orientation': body_orientation,
            'centroid': centroid,
            'bounding_box': {
                'min': min_bounds,
                'max': max_bounds,
                'dimensions': max_bounds - min_bounds
            }
        }

    def _estimate_body_orientation(self, human_points: np.ndarray) -> Dict[str, float]:
        """Estimate body orientation using Principal Component Analysis."""
        try:
            from sklearn.decomposition import PCA

            pca = PCA(n_components=3)
            pca.fit(human_points)

            # First component is the main body direction
            main_direction = pca.components_[0]

            # Calculate orientation angles
            yaw = np.arctan2(main_direction[1], main_direction[0]) * 180 / np.pi
            pitch = np.arcsin(main_direction[2]) * 180 / np.pi

            return {
                'yaw_degrees': yaw,
                'pitch_degrees': pitch,
                'main_direction_vector': main_direction,
                'explained_variance_ratio': pca.explained_variance_ratio_[0]
            }
        except Exception as e:
            print(f"Warning: Body orientation estimation failed: {e}")
            return {
                'yaw_degrees': 0.0,
                'pitch_degrees': 0.0,
                'main_direction_vector': np.array([1, 0, 0]),
                'explained_variance_ratio': 0.0
            }

    def save_model(self, filepath: str) -> None:
        """Save the trained model."""
        if self.model is None:
            raise ValueError("No model to save")
        torch.save(self.model.state_dict(), filepath)
        print(f"Model saved to {filepath}")

    def load_model(self, filepath: str, input_channels: int = 6, num_classes: int = 6) -> None:
        """Load a trained model."""
        if self.model is None:
            self.initialize_model(input_channels, num_classes)
        self.model.load_state_dict(torch.load(filepath, map_location=self.device))
        self.is_trained = True
        print(f"Model loaded from {filepath}")

# Export main classes and functions
__all__ = [
    'PointTransformerV3Block',
    'VehiclePointClassifier',
    'PTv3VehicleAnalyzer',
    'TORCH_AVAILABLE'
]