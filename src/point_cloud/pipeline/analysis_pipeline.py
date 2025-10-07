#!/usr/bin/env python3
"""
Analysis Pipeline Orchestrator

Coordinates the execution of analysis steps in a configurable pipeline.
Uses the Chain of Responsibility pattern to process point cloud data
through a series of analysis steps.

Author: Dimitrije Stojanovic
Date: September 2025
"""

from pathlib import Path
from typing import List, Optional, Dict, Any
import time

from ..interfaces.processors import AnalysisStep
from ..models.point_cloud_data import AnalysisResults, PointCloudData
from ..loaders.file_loader_factory import FileLoaderFactory
from ..config import AnalysisConfig
from ..error_handling import PointCloudError


class AnalysisPipeline:
    """Orchestrates the execution of analysis steps."""

    def __init__(self, config: Optional[AnalysisConfig] = None):
        self.config = config or AnalysisConfig()
        self.steps: List[AnalysisStep] = []
        self.file_loader_factory = FileLoaderFactory()

    def add_step(self, step: AnalysisStep) -> 'AnalysisPipeline':
        """
        Add an analysis step to the pipeline.

        Args:
            step: Analysis step to add

        Returns:
            Self for method chaining
        """
        if not isinstance(step, AnalysisStep):
            raise ValueError(f"Step must implement AnalysisStep interface, got {type(step)}")

        self.steps.append(step)
        return self

    def remove_step(self, step_type: type) -> 'AnalysisPipeline':
        """
        Remove analysis steps of a specific type.

        Args:
            step_type: Type of step to remove

        Returns:
            Self for method chaining
        """
        self.steps = [step for step in self.steps if not isinstance(step, step_type)]
        return self

    def clear_steps(self) -> 'AnalysisPipeline':
        """Clear all analysis steps."""
        self.steps.clear()
        return self

    def run(self, file_path: Path, **kwargs) -> AnalysisResults:
        """
        Run the complete analysis pipeline.

        Args:
            file_path: Path to point cloud file
            **kwargs: Additional parameters passed to analysis steps

        Returns:
            Complete analysis results

        Raises:
            PointCloudError: If any step fails
        """
        print(f"Starting analysis pipeline for: {file_path}")
        start_time = time.time()

        try:
            # Load data
            print("Loading point cloud data...")
            data = self._load_data(file_path)

            # Initialize results
            results = AnalysisResults(raw_data=data)

            # Run analysis steps
            for i, step in enumerate(self.steps):
                step_start_time = time.time()
                print(f"Step {i+1}/{len(self.steps)}: {step.name}")

                # Validate input
                if not step.validate_input(results):
                    raise PointCloudError(f"Step '{step.name}' validation failed")

                # Execute step
                step.on_step_start(results)
                results = step.analyze(results, **kwargs)
                step.on_step_complete(results)

                step_time = time.time() - step_start_time
                print(f"Step '{step.name}' completed in {step_time:.2f}s")

            total_time = time.time() - start_time
            print(f"Analysis pipeline completed in {total_time:.2f}s")

            # Add pipeline metadata
            self._add_pipeline_metadata(results, total_time)

            return results

        except Exception as e:
            total_time = time.time() - start_time
            raise PointCloudError(f"Analysis pipeline failed after {total_time:.2f}s: {e}") from e

    def run_from_data(self, data: PointCloudData, **kwargs) -> AnalysisResults:
        """
        Run analysis pipeline on pre-loaded data.

        Args:
            data: Pre-loaded point cloud data
            **kwargs: Additional parameters passed to analysis steps

        Returns:
            Complete analysis results
        """
        print("Starting analysis pipeline from pre-loaded data...")
        start_time = time.time()

        try:
            # Initialize results
            results = AnalysisResults(raw_data=data)

            # Run analysis steps
            for i, step in enumerate(self.steps):
                step_start_time = time.time()
                print(f"Step {i+1}/{len(self.steps)}: {step.name}")

                # Validate input
                if not step.validate_input(results):
                    raise PointCloudError(f"Step '{step.name}' validation failed")

                # Execute step
                step.on_step_start(results)
                results = step.analyze(results, **kwargs)
                step.on_step_complete(results)

                step_time = time.time() - step_start_time
                print(f"Step '{step.name}' completed in {step_time:.2f}s")

            total_time = time.time() - start_time
            print(f"Analysis pipeline completed in {total_time:.2f}s")

            # Add pipeline metadata
            self._add_pipeline_metadata(results, total_time)

            return results

        except Exception as e:
            total_time = time.time() - start_time
            raise PointCloudError(f"Analysis pipeline failed after {total_time:.2f}s: {e}") from e

    def get_step_names(self) -> List[str]:
        """Get list of step names in the pipeline."""
        return [step.name for step in self.steps]

    def get_step_count(self) -> int:
        """Get number of steps in the pipeline."""
        return len(self.steps)

    def _load_data(self, file_path: Path) -> PointCloudData:
        """Load point cloud data using the file loader factory."""
        return self.file_loader_factory.load_file(file_path)

    def _add_pipeline_metadata(self, results: AnalysisResults, execution_time: float) -> None:
        """Add pipeline execution metadata to results."""
        pipeline_metadata = {
            'pipeline_steps': self.get_step_names(),
            'step_count': self.get_step_count(),
            'execution_time_seconds': execution_time,
            'pipeline_config': {
                'dbscan_eps': self.config.DBSCAN_EPS,
                'dbscan_min_samples': self.config.DBSCAN_MIN_SAMPLES,
                'ground_threshold': self.config.GROUND_HEIGHT_THRESHOLD,
                'vehicle_max_height': self.config.VEHICLE_MAX_HEIGHT,
            }
        }

        # Add to raw data metadata
        results.raw_data.metadata.update({
            'pipeline_execution': pipeline_metadata
        })