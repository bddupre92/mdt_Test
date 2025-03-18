"""
Test Cases for Visualization Components.

This module provides test cases for validating the functionality and performance
of the visualization components in the application layer.
"""

import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any

from tests.application.test_harness import ApplicationTestCase, ApplicationTestHarness
from tests.theory.validation.synthetic_generators.visualization_generators import (
    DatasetGenerator,
    ChartGenerator
)
from core.application.visualization import (
    VisualizationEngine,
    ChartRenderer,
    InteractionHandler,
    PerformanceMonitor
)

class TestDataVisualization:
    """Test cases for data visualization components."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.engine = VisualizationEngine()
        self.renderer = ChartRenderer()
        self.handler = InteractionHandler()
        self.data_gen = DatasetGenerator()
        self.chart_gen = ChartGenerator()
    
    def test_chart_rendering(self):
        """Test rendering of various chart types and data visualizations."""
        # Define visualization scenarios
        chart_scenarios = {
            "time_series": {
                "type": "line_chart",
                "data_points": 1000,
                "variables": ["migraine_intensity", "stress_level", "sleep_quality"],
                "annotations": True
            },
            "correlation": {
                "type": "scatter_plot",
                "data_points": 500,
                "variables": ["trigger_exposure", "symptom_severity"],
                "regression": True
            },
            "distribution": {
                "type": "histogram",
                "data_points": 200,
                "variables": ["episode_duration", "recovery_time"],
                "kde": True
            }
        }
        
        # Create test case
        test_case = ApplicationTestCase(
            name="chart_rendering",
            inputs={
                "visualization_data": self.data_gen.generate_datasets(),
                "chart_scenarios": chart_scenarios,
                "render_config": {
                    "resolution": "high",
                    "color_scheme": "scientific",
                    "interactive": True
                }
            },
            expected_outputs={
                "rendering_quality": {
                    "visual_accuracy": 0.95,
                    "data_integrity": 1.0,
                    "annotation_clarity": 0.9
                },
                "performance_metrics": {
                    "render_time": 0.5,    # seconds
                    "memory_usage": 100,    # MB
                    "frame_rate": 60        # fps
                },
                "interaction_responsiveness": {
                    "zoom_latency": 0.1,    # seconds
                    "pan_smoothness": 0.9,
                    "tooltip_delay": 0.05   # seconds
                }
            },
            tolerance={
                "quality": 0.05,    # ±5% tolerance
                "timing": 0.1,      # ±100ms tolerance
                "memory": 20        # ±20MB tolerance
            },
            metadata={
                "description": "Validate chart rendering capabilities",
                "device_info": "standard_resolution",
                "browser_compatibility": ["chrome", "firefox", "safari"]
            }
        )
        
        # Create validation function
        def validate_visualization(
            visualization_data: Dict[str, Any],
            chart_scenarios: Dict[str, Dict[str, Any]],
            render_config: Dict[str, Any]
        ) -> Dict[str, Any]:
            # Render charts
            rendered_charts = {}
            for chart_type, config in chart_scenarios.items():
                chart = self.renderer.render_chart(
                    data=visualization_data[chart_type],
                    config=config,
                    render_settings=render_config
                )
                rendered_charts[chart_type] = chart
            
            # Evaluate rendering quality
            quality = self.engine.evaluate_quality(
                charts=rendered_charts,
                expected_data=visualization_data
            )
            
            # Measure performance
            performance = PerformanceMonitor().measure_metrics(
                charts=rendered_charts,
                render_config=render_config
            )
            
            # Test interaction responsiveness
            responsiveness = self.handler.test_interactions(
                charts=rendered_charts,
                interaction_types=["zoom", "pan", "tooltip"]
            )
            
            return {
                "rendering_quality": quality,
                "performance_metrics": performance,
                "interaction_responsiveness": responsiveness
            }
        
        # Create harness and validate
        harness = ApplicationTestHarness("chart_rendering", validate_visualization)
        harness.add_test_case(test_case)
        results = harness.run_all()
        
        return results

    def test_interactive_features(self):
        """Test interactive visualization features and user interactions."""
        # Define interaction scenarios
        interaction_scenarios = {
            "data_exploration": {
                "features": ["zoom", "pan", "filter"],
                "data_volume": "large",
                "update_frequency": "real-time"
            },
            "comparative_analysis": {
                "features": ["overlay", "sync_views", "annotations"],
                "chart_types": ["line", "scatter", "bar"],
                "linked_views": True
            },
            "custom_visualization": {
                "features": ["custom_colors", "export", "share"],
                "templates": ["clinical", "research", "patient"],
                "customization_level": "advanced"
            }
        }
        
        # Create test case
        test_case = ApplicationTestCase(
            name="interactive_features",
            inputs={
                "visualization_state": self.engine.get_current_state(),
                "interaction_scenarios": interaction_scenarios,
                "user_actions": [
                    {"type": "zoom", "factor": 2.0, "region": "center"},
                    {"type": "filter", "variable": "severity", "range": [5, 10]},
                    {"type": "annotate", "position": [150, 75], "text": "Peak"}
                ]
            },
            expected_outputs={
                "interaction_quality": {
                    "responsiveness": 0.95,
                    "accuracy": 0.9,
                    "smoothness": 0.85
                },
                "visualization_updates": {
                    "update_speed": 0.1,     # seconds
                    "visual_consistency": 0.9,
                    "data_accuracy": 1.0
                },
                "user_experience": {
                    "interaction_success": 0.95,
                    "error_rate": 0.05,
                    "user_satisfaction": 0.9
                }
            },
            tolerance={
                "responsiveness": 0.1,  # ±100ms tolerance
                "accuracy": 0.05,       # ±5% tolerance
                "satisfaction": 0.1     # ±10% tolerance
            },
            metadata={
                "description": "Validate interactive visualization features",
                "interaction_mode": "real-time",
                "performance_target": "60fps"
            }
        )
        
        # Create validation function
        def validate_interactions(
            visualization_state: Dict[str, Any],
            interaction_scenarios: Dict[str, Dict[str, Any]],
            user_actions: List[Dict[str, Any]]
        ) -> Dict[str, Any]:
            # Process user interactions
            interaction_results = self.handler.process_interactions(
                state=visualization_state,
                scenarios=interaction_scenarios,
                actions=user_actions
            )
            
            # Evaluate interaction quality
            quality = self.handler.evaluate_quality(
                results=interaction_results,
                expected_behavior=test_case.metadata
            )
            
            # Measure update performance
            updates = self.engine.measure_updates(
                results=interaction_results,
                state=visualization_state
            )
            
            # Assess user experience
            experience = self.handler.assess_user_experience(
                results=interaction_results,
                actions=user_actions
            )
            
            return {
                "interaction_quality": quality,
                "visualization_updates": updates,
                "user_experience": experience
            }
        
        # Create harness and validate
        harness = ApplicationTestHarness("interactive_features", validate_interactions)
        harness.add_test_case(test_case)
        results = harness.run_all()
        
        return results 