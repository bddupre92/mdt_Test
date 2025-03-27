"""
Expert Adapter for MoE Framework

This module provides adapters for initializing experts and other MoE framework components 
with the correct parameters, resolving compatibility issues.
"""

import os
import sys
import logging
import importlib
from typing import Dict, List, Any, Optional, Tuple, Union

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ExpertInitializer:
    """
    Initializes experts with the correct parameters, handling parameter mapping and defaults.
    """

    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.available_experts = self._get_available_experts()
        
    def _get_available_experts(self) -> Dict[str, Any]:
        """
        Check which expert classes are available in the moe_framework.
        
        Returns:
            Dictionary mapping expert names to their classes
        """
        available_experts = {}
        
        try:
            # Import BaseExpert to check if experts module is available
            from moe_framework.experts.base_expert import BaseExpert
            
            # Try importing each expert type
            expert_modules = {
                'physiological': 'moe_framework.experts.physiological_expert',
                'environmental': 'moe_framework.experts.environmental_expert',
                'behavioral': 'moe_framework.experts.behavioral_expert',
                'medication_history': 'moe_framework.experts.medication_history_expert'
            }
            
            for expert_type, module_path in expert_modules.items():
                try:
                    module = importlib.import_module(module_path)
                    if expert_type == 'physiological':
                        available_experts[expert_type] = module.PhysiologicalExpert
                    elif expert_type == 'environmental':
                        available_experts[expert_type] = module.EnvironmentalExpert
                    elif expert_type == 'behavioral':
                        available_experts[expert_type] = module.BehavioralExpert
                    elif expert_type == 'medication_history':
                        available_experts[expert_type] = module.MedicationHistoryExpert
                        
                    if self.verbose:
                        logger.info(f"Successfully imported {expert_type} expert")
                        
                except (ImportError, AttributeError) as e:
                    if self.verbose:
                        logger.warning(f"Could not import {expert_type} expert: {str(e)}")
                    
        except ImportError as e:
            if self.verbose:
                logger.warning(f"Could not import BaseExpert: {str(e)}")
            
        return available_experts
    
    def initialize_expert(self, expert_config: Dict[str, Any]) -> Optional[Any]:
        """
        Initialize an expert with the correct parameters based on expert type.
        
        Args:
            expert_config: Configuration dictionary for the expert
            
        Returns:
            Initialized expert instance or None if initialization fails
        """
        expert_type = expert_config.get('type')
        if not expert_type or expert_type not in self.available_experts:
            if self.verbose:
                logger.warning(f"Expert type {expert_type} not available")
            return None
        
        # Get expert class
        expert_class = self.available_experts[expert_type]
        
        # Clean the config by removing 'type' key that's not expected by the expert's __init__
        cleaned_config = expert_config.copy()
        cleaned_config.pop('type', None)
        
        # Handle specific parameter mappings for each expert type
        if expert_type == 'physiological':
            # PhysiologicalExpert parameters
            return self._initialize_physiological_expert(expert_class, cleaned_config)
        elif expert_type == 'environmental':
            # EnvironmentalExpert parameters
            return self._initialize_environmental_expert(expert_class, cleaned_config)
        elif expert_type == 'behavioral':
            # BehavioralExpert parameters
            return self._initialize_behavioral_expert(expert_class, cleaned_config)
        elif expert_type == 'medication_history':
            # MedicationHistoryExpert parameters
            return self._initialize_medication_history_expert(expert_class, cleaned_config)
        
        return None
    
    def _initialize_physiological_expert(self, expert_class, config: Dict[str, Any]) -> Any:
        """Initialize a PhysiologicalExpert with the correct parameters."""
        try:
            # Extract required parameters
            vital_cols = config.get('vital_cols', [])
            patient_id_col = config.get('patient_id_col', 'patient_id')
            timestamp_col = config.get('timestamp_col', 'date')
            normalize_vitals = config.get('normalize_vitals', True)
            extract_variability = config.get('extract_variability', True)
            model = config.get('model', None)
            name = config.get('name', "PhysiologicalExpert")
            metadata = config.get('metadata', {})
            
            # Initialize expert
            expert = expert_class(
                vital_cols=vital_cols,
                patient_id_col=patient_id_col,
                timestamp_col=timestamp_col,
                normalize_vitals=normalize_vitals,
                extract_variability=extract_variability,
                model=model,
                name=name,
                metadata=metadata
            )
            
            if self.verbose:
                logger.info(f"Successfully initialized {name}")
                
            return expert
            
        except Exception as e:
            logger.error(f"Failed to initialize physiological expert: {str(e)}")
            return None
    
    def _initialize_environmental_expert(self, expert_class, config: Dict[str, Any]) -> Any:
        """Initialize an EnvironmentalExpert with the correct parameters."""
        try:
            # Extract required parameters
            env_cols = config.get('env_cols', [])
            location_col = config.get('location_col', 'location')
            timestamp_col = config.get('timestamp_col', 'date')
            include_weather = config.get('include_weather', True)
            include_pollution = config.get('include_pollution', True)
            model = config.get('model', None)
            name = config.get('name', "EnvironmentalExpert")
            metadata = config.get('metadata', {})
            
            # Initialize expert
            expert = expert_class(
                env_cols=env_cols,
                location_col=location_col,
                timestamp_col=timestamp_col,
                include_weather=include_weather,
                include_pollution=include_pollution,
                model=model,
                name=name,
                metadata=metadata
            )
            
            if self.verbose:
                logger.info(f"Successfully initialized {name}")
                
            return expert
            
        except Exception as e:
            logger.error(f"Failed to initialize environmental expert: {str(e)}")
            return None
    
    def _initialize_behavioral_expert(self, expert_class, config: Dict[str, Any]) -> Any:
        """Initialize a BehavioralExpert with the correct parameters."""
        try:
            # Extract required parameters
            behavior_cols = config.get('behavior_cols', [])
            patient_id_col = config.get('patient_id_col', 'patient_id')
            timestamp_col = config.get('timestamp_col', 'date')
            include_sleep = config.get('include_sleep', True)
            include_activity = config.get('include_activity', True)
            include_stress = config.get('include_stress', True)
            model = config.get('model', None)
            name = config.get('name', "BehavioralExpert")
            metadata = config.get('metadata', {})
            
            # Initialize expert
            expert = expert_class(
                behavior_cols=behavior_cols,
                patient_id_col=patient_id_col,
                timestamp_col=timestamp_col,
                include_sleep=include_sleep,
                include_activity=include_activity,
                include_stress=include_stress,
                model=model,
                name=name,
                metadata=metadata
            )
            
            if self.verbose:
                logger.info(f"Successfully initialized {name}")
                
            return expert
            
        except Exception as e:
            logger.error(f"Failed to initialize behavioral expert: {str(e)}")
            return None
    
    def _initialize_medication_history_expert(self, expert_class, config: Dict[str, Any]) -> Any:
        """Initialize a MedicationHistoryExpert with the correct parameters."""
        try:
            # Extract required parameters
            medication_cols = config.get('medication_cols', [])
            patient_id_col = config.get('patient_id_col', 'patient_id')
            timestamp_col = config.get('timestamp_col', 'date')
            include_dosage = config.get('include_dosage', True)
            include_frequency = config.get('include_frequency', True)
            include_interactions = config.get('include_interactions', True)
            model = config.get('model', None)
            name = config.get('name', "MedicationHistoryExpert")
            metadata = config.get('metadata', {})
            
            # Initialize expert
            expert = expert_class(
                medication_cols=medication_cols,
                patient_id_col=patient_id_col,
                timestamp_col=timestamp_col,
                include_dosage=include_dosage,
                include_frequency=include_frequency,
                include_interactions=include_interactions,
                model=model,
                name=name,
                metadata=metadata
            )
            
            if self.verbose:
                logger.info(f"Successfully initialized {name}")
                
            return expert
            
        except Exception as e:
            logger.error(f"Failed to initialize medication_history expert: {str(e)}")
            return None


class GatingNetworkInitializer:
    """
    Initializes gating networks with the correct parameters, handling parameter mapping and defaults.
    """
    
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.available_gating_networks = self._get_available_gating_networks()
        
    def _get_available_gating_networks(self) -> Dict[str, Any]:
        """
        Check which gating network classes are available in the moe_framework.
        
        Returns:
            Dictionary mapping gating network names to their classes
        """
        available_networks = {}
        
        try:
            # Import gating module
            from moe_framework.gating import gating_network
            
            # Try importing each gating network type
            try:
                available_networks['quality_aware'] = gating_network.QualityAwareWeighting
                if self.verbose:
                    logger.info("Successfully imported QualityAwareWeighting")
            except AttributeError as e:
                if self.verbose:
                    logger.warning(f"Could not import QualityAwareWeighting: {str(e)}")
                
            try:
                available_networks['adaptive'] = gating_network.AdaptiveWeighting
                if self.verbose:
                    logger.info("Successfully imported AdaptiveWeighting")
            except AttributeError as e:
                if self.verbose:
                    logger.warning(f"Could not import AdaptiveWeighting: {str(e)}")
                
        except ImportError as e:
            if self.verbose:
                logger.warning(f"Could not import gating_network module: {str(e)}")
            
        return available_networks
    
    def initialize_gating_network(self, config: Dict[str, Any], experts: List[Any]) -> Optional[Any]:
        """
        Initialize a gating network with the correct parameters.
        
        Args:
            config: Configuration dictionary for the gating network
            experts: List of expert instances
            
        Returns:
            Initialized gating network instance or None if initialization fails
        """
        network_type = config.get('type', 'quality_aware')
        
        if network_type not in self.available_gating_networks:
            if self.verbose:
                logger.warning(f"Gating network type {network_type} not available")
            return None
        
        # Get gating network class
        network_class = self.available_gating_networks[network_type]
        
        # Clean the config by removing 'type' key
        cleaned_config = config.copy()
        cleaned_config.pop('type', None)
        # Remove 'combination_strategy' if it exists (caused an error in QualityAwareWeighting)
        cleaned_config.pop('combination_strategy', None)
        
        try:
            if network_type == 'quality_aware':
                # QualityAwareWeighting parameters
                return self._initialize_quality_aware_weighting(network_class, cleaned_config, experts)
            elif network_type == 'adaptive':
                # AdaptiveWeighting parameters
                return self._initialize_adaptive_weighting(network_class, cleaned_config, experts)
        except Exception as e:
            logger.error(f"Failed to initialize gating network: {str(e)}")
            return None
        
        return None
    
    def _initialize_quality_aware_weighting(self, network_class, config: Dict[str, Any], experts: List[Any]) -> Any:
        """Initialize a QualityAwareWeighting network with the correct parameters."""
        try:
            # Extract required parameters
            confidence_threshold = config.get('confidence_threshold', 0.7)
            quality_weight = config.get('quality_weight', 0.5)
            
            # Initialize network
            network = network_class(
                experts=experts,
                confidence_threshold=confidence_threshold,
                quality_weight=quality_weight
            )
            
            if self.verbose:
                logger.info("Successfully initialized QualityAwareWeighting")
                
            return network
            
        except Exception as e:
            logger.error(f"Failed to initialize QualityAwareWeighting: {str(e)}")
            
            # Check the actual init parameters to help debug
            import inspect
            sig = inspect.signature(network_class.__init__)
            logger.error(f"QualityAwareWeighting.__init__ accepts parameters: {list(sig.parameters.keys())}")
            
            return None
    
    def _initialize_adaptive_weighting(self, network_class, config: Dict[str, Any], experts: List[Any]) -> Any:
        """Initialize an AdaptiveWeighting network with the correct parameters."""
        try:
            # Extract required parameters
            learning_rate = config.get('learning_rate', 0.01)
            adaptation_rate = config.get('adaptation_rate', 0.1)
            
            # Initialize network
            network = network_class(
                experts=experts,
                learning_rate=learning_rate,
                adaptation_rate=adaptation_rate
            )
            
            if self.verbose:
                logger.info("Successfully initialized AdaptiveWeighting")
                
            return network
            
        except Exception as e:
            logger.error(f"Failed to initialize AdaptiveWeighting: {str(e)}")
            return None


def initialize_moe_pipeline(config: Dict[str, Any] = None, verbose: bool = False) -> Optional[Any]:
    """
    Initialize a MoE pipeline with the correct parameters using the adapter classes.
    
    Args:
        config: Configuration dictionary for the pipeline
        verbose: Whether to display detailed logs
        
    Returns:
        Initialized MoE pipeline instance or None if initialization fails
    """
    if verbose:
        logger.info("Initializing MoE pipeline with adapters...")
    
    try:
        # Import MoEPipeline
        from moe_framework.workflow.moe_pipeline import MoEPipeline
        
        # Create expert initializer
        expert_initializer = ExpertInitializer(verbose=verbose)
        
        # Get default config if None provided
        if config is None:
            config = {
                'experts': [
                    {
                        'type': 'physiological',
                        'vital_cols': ['heart_rate', 'blood_pressure', 'temperature'],
                        'name': 'PhysiologicalExpert'
                    },
                    {
                        'type': 'environmental',
                        'env_cols': ['temperature', 'humidity', 'air_quality'],
                        'name': 'EnvironmentalExpert'
                    },
                    {
                        'type': 'behavioral',
                        'behavior_cols': ['sleep_hours', 'activity_level', 'stress_level'],
                        'name': 'BehavioralExpert'
                    },
                    {
                        'type': 'medication_history',
                        'medication_cols': ['medication_name', 'dosage', 'frequency'],
                        'name': 'MedicationHistoryExpert'
                    }
                ],
                'gating': {
                    'type': 'quality_aware',
                    'confidence_threshold': 0.7,
                    'quality_weight': 0.5
                }
            }
        
        # Initialize experts
        experts = []
        expert_configs = config.get('experts', [])
        
        for expert_config in expert_configs:
            expert = expert_initializer.initialize_expert(expert_config)
            if expert is not None:
                experts.append(expert)
        
        if verbose:
            logger.info(f"Initialized {len(experts)} experts: {[expert.__class__.__name__ for expert in experts]}")
        
        # Create a clean pipeline config without experts and gating configs
        pipeline_config = config.copy()
        pipeline_config.pop('experts', None)
        pipeline_config.pop('gating', None)
        
        # Initialize pipeline with the cleaned config
        pipeline = MoEPipeline(experts=experts, config=pipeline_config, verbose=verbose)
        
        # Initialize gating network separately
        gating_initializer = GatingNetworkInitializer(verbose=verbose)
        gating_config = config.get('gating', {})
        
        gating_network = gating_initializer.initialize_gating_network(gating_config, experts)
        if gating_network is not None:
            # Set the gating network directly
            pipeline.gating_network = gating_network
        
        if verbose:
            logger.info("Successfully initialized MoE pipeline with adapters")
        
        return pipeline
        
    except Exception as e:
        logger.error(f"Failed to initialize MoE pipeline: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return None

# Utility function to get MoE pipeline configuration from file
def get_default_moe_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Get default MoE pipeline configuration.
    
    Args:
        config_path: Optional path to configuration file
        
    Returns:
        Configuration dictionary
    """
    if config_path and os.path.exists(config_path):
        import json
        with open(config_path, 'r') as f:
            return json.load(f)
    
    # Default configuration
    return {
        'experts': [
            {
                'type': 'physiological',
                'vital_cols': ['heart_rate', 'blood_pressure', 'temperature'],
                'name': 'PhysiologicalExpert'
            },
            {
                'type': 'environmental',
                'env_cols': ['temperature', 'humidity', 'air_quality'],
                'name': 'EnvironmentalExpert'
            },
            {
                'type': 'behavioral',
                'behavior_cols': ['sleep_hours', 'activity_level', 'stress_level'],
                'name': 'BehavioralExpert'
            },
            {
                'type': 'medication_history',
                'medication_cols': ['medication_name', 'dosage', 'frequency'],
                'name': 'MedicationHistoryExpert'
            }
        ],
        'gating': {
            'type': 'quality_aware',
            'confidence_threshold': 0.7,
            'quality_weight': 0.5
        }
    } 