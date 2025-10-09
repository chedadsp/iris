#!/usr/bin/env python3
"""
AI Analyzer Processor

Integrates PointTransformerV3 AI analysis for enhanced vehicle detection
and human detection capabilities.

Author: Dimitrije Stojanovic
Date: September 2025
"""

from typing import Optional, Dict, Any

from ..interfaces.processors import AnalysisStep
from ..models.point_cloud_data import AnalysisResults
from ..config import AnalysisConfig


class AIAnalyzer(AnalysisStep):
    """AI-powered analysis using PointTransformerV3."""

    def __init__(self, config: AnalysisConfig, enable_ptv3: bool = True):
        self.config = config
        self.ptv3_analyzer = None
        self.enabled = enable_ptv3

        # Initialize PTv3 analyzer if available
        if self.enabled:
            try:
                from ..ptv3_integration import PTv3VehicleAnalyzer, TORCH_AVAILABLE
                if TORCH_AVAILABLE:
                    self.ptv3_analyzer = PTv3VehicleAnalyzer(device='auto')
                    print("✅ PointTransformerV3 integration enabled")
                else:
                    print("⚠️  PyTorch not available for PointTransformerV3")
                    self.enabled = False
            except ImportError:
                print("⚠️  PointTransformerV3 integration not available")
                self.enabled = False
            except Exception as e:
                print(f"⚠️  PointTransformerV3 initialization failed: {e}")
                self.enabled = False

    @property
    def name(self) -> str:
        return "AI Analysis"

    def validate_input(self, results: AnalysisResults) -> bool:
        """Validate input and check if AI analysis is enabled."""
        return (self.enabled and
                results.raw_data is not None and
                results.raw_data.points is not None)

    def analyze(self, results: AnalysisResults, **kwargs) -> AnalysisResults:
        """
        Run AI-enhanced analysis using PointTransformerV3.

        Args:
            results: Current analysis results
            **kwargs: Additional parameters

        Returns:
            Updated analysis results with AI enhancements
        """
        if not self.enabled:
            print("⚠️  AI analysis disabled or unavailable")
            return results

        if not self.validate_input(results):
            print("⚠️  Invalid input for AI analysis")
            return results

        print("🧠 Running PointTransformerV3 enhanced analysis...")

        try:
            # Run PTv3 enhanced vehicle detection
            ptv3_results = self._run_vehicle_analysis(results)
            results.ptv3_results = ptv3_results

            # Extract AI-enhanced results
            if ptv3_results:
                self._update_with_ai_results(results, ptv3_results)

            # Run human detection if interior points are available
            if results.has_interior_detection:
                human_results = self._detect_humans(results)
                results.human_detection_results = human_results

            # Compare traditional vs AI results
            self._compare_methods(results)

        except Exception as e:
            print(f"❌ AI analysis failed: {e}")

        return results

    def _run_vehicle_analysis(self, results: AnalysisResults) -> Optional[Dict[str, Any]]:
        """Run PTv3 enhanced vehicle detection."""
        if not self.ptv3_analyzer:
            return None

        try:
            ptv3_results = self.ptv3_analyzer.process_large_pointcloud(
                results.raw_data.points,
                colors=results.raw_data.colors,
                intensity=results.raw_data.intensity
            )

            print(f"✅ PTv3 Analysis Results:")
            print(f"   Ground points: {len(ptv3_results.get('ground_points', []))}")
            print(f"   Vehicle exterior: {len(ptv3_results.get('vehicle_exterior_points', []))}")
            print(f"   Vehicle interior: {len(ptv3_results.get('vehicle_interior_points', []))}")
            print(f"   Total vehicle: {len(ptv3_results.get('vehicle_all_points', []))}")

            return ptv3_results

        except Exception as e:
            print(f"❌ PTv3 vehicle analysis failed: {e}")
            return None

    def _update_with_ai_results(self, results: AnalysisResults, ptv3_results: Dict[str, Any]) -> None:
        """Update results with AI-enhanced data."""
        # Store PTv3 enhanced results
        results.vehicle_points_ptv3 = ptv3_results.get('vehicle_all_points')
        results.interior_points_ptv3 = ptv3_results.get('vehicle_interior_points')
        results.ground_points_ptv3 = ptv3_results.get('ground_points')

    def _detect_humans(self, results: AnalysisResults) -> Optional[Dict[str, Any]]:
        """Detect humans in vehicle interior."""
        if not self.ptv3_analyzer:
            return None

        # Use PTv3 interior points if available, otherwise traditional
        interior_points = (results.interior_points_ptv3
                          if results.interior_points_ptv3 is not None
                          else results.interior_points)

        if interior_points is None or len(interior_points) == 0:
            print("❌ No interior points available for human detection")
            return None

        print(f"👥 Analyzing {len(interior_points):,} interior points for human detection...")

        try:
            human_results = self.ptv3_analyzer.detect_humans_in_vehicle(
                interior_points,
                colors=results.raw_data.colors,
                intensity=results.raw_data.intensity
            )

            if human_results.get('human_detected', False):
                print(f"✅ Humans detected in vehicle!")
                print(f"   Human points: {len(human_results.get('human_points', []))}")
                print(f"   Estimated count: {human_results.get('estimated_human_count', 0)}")
                print(f"   Confidence: {human_results.get('average_confidence', 0):.2f}")

                # Store human points for visualization
                results.human_points = human_results.get('human_points')
                results.seat_points = human_results.get('seat_points')
            else:
                print("❌ No humans detected in vehicle interior")

            return human_results

        except Exception as e:
            print(f"❌ Human detection failed: {e}")
            return None

    def _compare_methods(self, results: AnalysisResults) -> None:
        """Compare traditional vs AI analysis results."""
        if not results.ptv3_results:
            return

        print("\n🔍 Analysis Method Comparison:")
        print("=" * 50)

        # Traditional results
        traditional_ground = len(results.ground_points) if results.ground_points is not None else 0
        traditional_vehicle = len(results.vehicle_points) if results.vehicle_points is not None else 0
        traditional_interior = len(results.interior_points) if results.interior_points is not None else 0

        # PTv3 results
        ptv3_ground = len(results.ptv3_results.get('ground_points', []))
        ptv3_vehicle = len(results.ptv3_results.get('vehicle_all_points', []))
        ptv3_interior = len(results.ptv3_results.get('vehicle_interior_points', []))

        print(f"Traditional Analysis:")
        print(f"  Ground points: {traditional_ground:,}")
        print(f"  Vehicle points: {traditional_vehicle:,}")
        print(f"  Interior points: {traditional_interior:,}")

        print(f"\nAI Enhanced Analysis:")
        print(f"  Ground points: {ptv3_ground:,}")
        print(f"  Vehicle points: {ptv3_vehicle:,}")
        print(f"  Interior points: {ptv3_interior:,}")

        print(f"\nDifferences (AI - Traditional):")
        print(f"  Ground: {ptv3_ground - traditional_ground:+,}")
        print(f"  Vehicle: {ptv3_vehicle - traditional_vehicle:+,}")
        print(f"  Interior: {ptv3_interior - traditional_interior:+,}")

    def is_enabled(self) -> bool:
        """Check if AI analysis is enabled and available."""
        return self.enabled and self.ptv3_analyzer is not None