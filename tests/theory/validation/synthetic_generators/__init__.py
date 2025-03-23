"""
Synthetic data generators for validation testing.

This package provides generators for creating synthetic test data for various
components of the migraine prediction system.
"""

from tests.theory.validation.synthetic_generators.patient_generators import (
    PatientGenerator,
    LongitudinalDataGenerator,
    PatientProfile
)

from tests.theory.validation.synthetic_generators.trigger_generators import (
    TriggerProfile,
    generate_patient_scenario
)

from tests.theory.validation.synthetic_generators.environmental_generators import (
    generate_environmental_scenario
)

from tests.theory.validation.synthetic_generators.signal_generators import (
    generate_multimodal_stress_response
)

# Define aliases to match expected interface in tests
PatientDataGenerator = PatientGenerator
MigraineEpisodeGenerator = LongitudinalDataGenerator
TriggerProfileGenerator = TriggerProfile
TriggerEventGenerator = generate_patient_scenario
EnvironmentalDataGenerator = generate_environmental_scenario
SensorDataGenerator = generate_multimodal_stress_response

__all__ = [
    'PatientGenerator',
    'LongitudinalDataGenerator',
    'PatientProfile',
    'TriggerProfile',
    'generate_patient_scenario',
    'generate_environmental_scenario',
    'generate_multimodal_stress_response',
    # Aliases
    'PatientDataGenerator',
    'MigraineEpisodeGenerator',
    'TriggerProfileGenerator',
    'TriggerEventGenerator',
    'EnvironmentalDataGenerator',
    'SensorDataGenerator'
] 