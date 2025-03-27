"""
Domain-Specific Preprocessing Operations Component

This module provides Streamlit UI components for configuring domain-specific preprocessing operations
such as medication normalization, symptom extraction, temporal pattern extraction, and comorbidity analysis.
These operations are specifically designed for medical and clinical data related to migraine treatment.
"""

import streamlit as st
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional


def render_domain_operations():
    """Render UI for configuring domain-specific preprocessing operations."""
    st.subheader("Domain-Specific Preprocessing")
    
    # Get current configuration
    config = st.session_state.preprocessing_config
    domain_ops = config.get('domain_operations', {})
    
    # Medication Normalizer
    st.write("---")
    st.write("**Medication Normalization**")
    
    med_config = domain_ops.get('medication_normalizer', {})
    
    med_include = st.checkbox(
        "Include Medication Normalizer",
        value=med_config.get('include', False),
        key="med_include_checkbox"
    )
    
    if med_include:
        med_params = med_config.get('params', {})
        
        col1, col2 = st.columns(2)
        
        with col1:
            med_column = st.text_input(
                "Medication Column",
                value=med_params.get('med_column', 'medication'),
                key="med_column_input"
            )
            med_params['med_column'] = med_column
            
            dosage_column = st.text_input(
                "Dosage Column (optional)",
                value=med_params.get('dosage_column', ''),
                key="dosage_column_input"
            )
            med_params['dosage_column'] = dosage_column if dosage_column else None
            
        with col2:
            normalize_brand_names = st.checkbox(
                "Normalize Brand Names",
                value=med_params.get('normalize_brand_names', True),
                key="normalize_brand_names_checkbox"
            )
            med_params['normalize_brand_names'] = normalize_brand_names
            
            standardize_units = st.checkbox(
                "Standardize Units",
                value=med_params.get('standardize_units', True),
                key="standardize_units_checkbox"
            )
            med_params['standardize_units'] = standardize_units
            
        # Add help text explaining medication normalization
        with st.expander("About Medication Normalization"):
            st.markdown("""
            **Medication Normalization** standardizes medication names, dosages, and units to enable consistent analysis.
            
            Features:
            - Converts brand names to generic names (e.g., "Imitrex" → "sumatriptan")
            - Standardizes dosage units (e.g., "mg", "milligrams" → "mg")
            - Normalizes dosage formats (e.g., "100mg" → "100 mg")
            - Categorizes medications by class (e.g., triptans, NSAIDs)
            
            This processing is essential for accurate analysis of medication effectiveness and patterns.
            """)
            
        # Custom medication mappings
        st.write("**Custom Medication Mappings**")
        
        custom_mappings = med_params.get('custom_mappings', {})
        
        # Display existing mappings
        if custom_mappings:
            st.write("Current custom mappings:")
            mapping_df = pd.DataFrame({
                'Original': list(custom_mappings.keys()),
                'Normalized': list(custom_mappings.values())
            })
            st.dataframe(mapping_df)
            
        # Add new mapping
        col1, col2, col3 = st.columns([2, 2, 1])
        
        with col1:
            original_name = st.text_input(
                "Original Name",
                key="original_name_input"
            )
            
        with col2:
            normalized_name = st.text_input(
                "Normalized Name",
                key="normalized_name_input"
            )
            
        with col3:
            if st.button("Add Mapping") and original_name and normalized_name:
                custom_mappings[original_name] = normalized_name
                st.success(f"Added mapping: {original_name} → {normalized_name}")
                
        med_params['custom_mappings'] = custom_mappings
        
        med_config['params'] = med_params
        
    med_config['include'] = med_include
    domain_ops['medication_normalizer'] = med_config
    
    # Symptom Extractor
    st.write("---")
    st.write("**Symptom Extraction**")
    
    symptom_config = domain_ops.get('symptom_extractor', {})
    
    symptom_include = st.checkbox(
        "Include Symptom Extractor",
        value=symptom_config.get('include', False),
        key="symptom_include_checkbox"
    )
    
    if symptom_include:
        symptom_params = symptom_config.get('params', {})
        
        col1, col2 = st.columns(2)
        
        with col1:
            text_column = st.text_input(
                "Text Column",
                value=symptom_params.get('text_column', 'notes'),
                key="text_column_input"
            )
            symptom_params['text_column'] = text_column
            
            extract_severity = st.checkbox(
                "Extract Severity",
                value=symptom_params.get('extract_severity', True),
                key="extract_severity_checkbox"
            )
            symptom_params['extract_severity'] = extract_severity
            
        with col2:
            extract_duration = st.checkbox(
                "Extract Duration",
                value=symptom_params.get('extract_duration', True),
                key="extract_duration_checkbox"
            )
            symptom_params['extract_duration'] = extract_duration
            
            extract_location = st.checkbox(
                "Extract Location",
                value=symptom_params.get('extract_location', True),
                key="extract_location_checkbox"
            )
            symptom_params['extract_location'] = extract_location
            
        # Add help text explaining symptom extraction
        with st.expander("About Symptom Extraction"):
            st.markdown("""
            **Symptom Extraction** identifies and normalizes symptoms mentioned in clinical notes or patient reports.
            
            Features:
            - Identifies common migraine symptoms (headache, nausea, photophobia, etc.)
            - Extracts severity indicators (mild, moderate, severe)
            - Identifies anatomical locations (frontal, temporal, etc.)
            - Extracts duration information (hours, days)
            
            This processing converts unstructured text data into structured features for analysis.
            """)
            
        # Custom symptom keywords
        st.write("**Custom Symptom Keywords**")
        
        symptom_keywords = symptom_params.get('symptom_keywords', [
            'headache', 'nausea', 'vomiting', 'photophobia', 'phonophobia',
            'aura', 'dizziness', 'fatigue', 'sensitivity'
        ])
        
        keywords_str = st.text_area(
            "Symptom Keywords (one per line)",
            value='\n'.join(symptom_keywords),
            height=100,
            key="symptom_keywords_textarea"
        )
        
        symptom_params['symptom_keywords'] = [kw.strip() for kw in keywords_str.split('\n') if kw.strip()]
        
        symptom_config['params'] = symptom_params
        
    symptom_config['include'] = symptom_include
    domain_ops['symptom_extractor'] = symptom_config
    
    # Temporal Pattern Extractor
    st.write("---")
    st.write("**Temporal Pattern Extraction**")
    
    temporal_config = domain_ops.get('temporal_pattern_extractor', {})
    
    temporal_include = st.checkbox(
        "Include Temporal Pattern Extractor",
        value=temporal_config.get('include', False),
        key="temporal_include_checkbox"
    )
    
    if temporal_include:
        temporal_params = temporal_config.get('params', {})
        
        col1, col2 = st.columns(2)
        
        with col1:
            timestamp_column = st.text_input(
                "Timestamp Column",
                value=temporal_params.get('timestamp_column', 'timestamp'),
                key="timestamp_column_input"
            )
            temporal_params['timestamp_column'] = timestamp_column
            
            patient_id_column = st.text_input(
                "Patient ID Column",
                value=temporal_params.get('patient_id_column', 'patient_id'),
                key="patient_id_column_input"
            )
            temporal_params['patient_id_column'] = patient_id_column
            
        with col2:
            event_column = st.text_input(
                "Event Column (optional)",
                value=temporal_params.get('event_column', ''),
                key="event_column_input"
            )
            temporal_params['event_column'] = event_column if event_column else None
            
            extract_cyclical = st.checkbox(
                "Extract Cyclical Features",
                value=temporal_params.get('extract_cyclical', True),
                key="extract_cyclical_checkbox"
            )
            temporal_params['extract_cyclical'] = extract_cyclical
            
        # Additional temporal features
        st.write("**Temporal Features to Extract**")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            extract_hour = st.checkbox(
                "Hour of Day",
                value=temporal_params.get('extract_hour', True),
                key="extract_hour_checkbox"
            )
            temporal_params['extract_hour'] = extract_hour
            
            extract_day = st.checkbox(
                "Day of Week",
                value=temporal_params.get('extract_day', True),
                key="extract_day_checkbox"
            )
            temporal_params['extract_day'] = extract_day
            
        with col2:
            extract_month = st.checkbox(
                "Month of Year",
                value=temporal_params.get('extract_month', True),
                key="extract_month_checkbox"
            )
            temporal_params['extract_month'] = extract_month
            
            extract_season = st.checkbox(
                "Season",
                value=temporal_params.get('extract_season', True),
                key="extract_season_checkbox"
            )
            temporal_params['extract_season'] = extract_season
            
        with col3:
            extract_frequency = st.checkbox(
                "Event Frequency",
                value=temporal_params.get('extract_frequency', True),
                key="extract_frequency_checkbox"
            )
            temporal_params['extract_frequency'] = extract_frequency
            
            extract_intervals = st.checkbox(
                "Time Intervals",
                value=temporal_params.get('extract_intervals', True),
                key="extract_intervals_checkbox"
            )
            temporal_params['extract_intervals'] = extract_intervals
            
        # Add help text explaining temporal pattern extraction
        with st.expander("About Temporal Pattern Extraction"):
            st.markdown("""
            **Temporal Pattern Extraction** identifies time-based patterns in clinical data.
            
            Features:
            - Cyclical time features (hour, day, month)
            - Seasonal patterns
            - Event frequency and intervals
            - Temporal clustering of events
            
            For migraine data, this can reveal:
            - Time-of-day patterns for migraine onset
            - Seasonal variations
            - Frequency changes over time
            - Patterns in attack intervals
            """)
            
        temporal_config['params'] = temporal_params
        
    temporal_config['include'] = temporal_include
    domain_ops['temporal_pattern_extractor'] = temporal_config
    
    # Comorbidity Analyzer
    st.write("---")
    st.write("**Comorbidity Analysis**")
    
    comorbidity_config = domain_ops.get('comorbidity_analyzer', {})
    
    comorbidity_include = st.checkbox(
        "Include Comorbidity Analyzer",
        value=comorbidity_config.get('include', False),
        key="comorbidity_include_checkbox"
    )
    
    if comorbidity_include:
        comorbidity_params = comorbidity_config.get('params', {})
        
        col1, col2 = st.columns(2)
        
        with col1:
            diagnosis_column = st.text_input(
                "Diagnosis Column",
                value=comorbidity_params.get('diagnosis_column', 'diagnosis'),
                key="diagnosis_column_input"
            )
            comorbidity_params['diagnosis_column'] = diagnosis_column
            
            patient_id_column = st.text_input(
                "Patient ID Column",
                value=comorbidity_params.get('patient_id_column', 'patient_id'),
                key="comorbidity_patient_id_column_input"
            )
            comorbidity_params['patient_id_column'] = patient_id_column
            
        with col2:
            create_indicator_features = st.checkbox(
                "Create Indicator Features",
                value=comorbidity_params.get('create_indicator_features', True),
                key="create_indicator_features_checkbox"
            )
            comorbidity_params['create_indicator_features'] = create_indicator_features
            
            create_count_features = st.checkbox(
                "Create Count Features",
                value=comorbidity_params.get('create_count_features', True),
                key="create_count_features_checkbox"
            )
            comorbidity_params['create_count_features'] = create_count_features
            
        # Comorbidity categories
        st.write("**Comorbidity Categories**")
        
        default_categories = [
            'anxiety', 'depression', 'insomnia', 'hypertension', 
            'allergies', 'asthma', 'epilepsy', 'fibromyalgia'
        ]
        
        comorbidity_categories = comorbidity_params.get('comorbidity_categories', default_categories)
        
        categories_str = st.text_area(
            "Comorbidity Categories (one per line)",
            value='\n'.join(comorbidity_categories),
            height=100,
            key="comorbidity_categories_textarea"
        )
        
        comorbidity_params['comorbidity_categories'] = [cat.strip() for cat in categories_str.split('\n') if cat.strip()]
        
        # Add help text explaining comorbidity analysis
        with st.expander("About Comorbidity Analysis"):
            st.markdown("""
            **Comorbidity Analysis** identifies and analyzes conditions that frequently co-occur with migraines.
            
            Features:
            - Creates indicator variables for common comorbidities
            - Calculates comorbidity counts and ratios
            - Groups related conditions into categories
            - Identifies potential risk factors and correlations
            
            Common migraine comorbidities include:
            - Anxiety and depression
            - Sleep disorders
            - Cardiovascular conditions
            - Other neurological disorders
            - Allergies and asthma
            
            This analysis can reveal important patterns and potential treatment considerations.
            """)
            
        comorbidity_config['params'] = comorbidity_params
        
    comorbidity_config['include'] = comorbidity_include
    domain_ops['comorbidity_analyzer'] = comorbidity_config
    
    # Update the configuration
    config['domain_operations'] = domain_ops
    st.session_state.preprocessing_config = config
