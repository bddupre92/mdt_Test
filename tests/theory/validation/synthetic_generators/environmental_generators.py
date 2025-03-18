"""
Environmental Factors Generator for Migraine Digital Twin Validation.

This module provides generators for creating synthetic environmental data
including weather patterns, air quality, and light/noise exposure.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from datetime import datetime, timedelta

class WeatherGenerator:
    """Generate synthetic weather patterns."""
    
    def __init__(
        self,
        base_temperature: float = 20.0,  # Celsius
        base_humidity: float = 0.5,      # 0-1
        base_pressure: float = 1013.25,  # hPa
        seasonal_variation: bool = True
    ):
        self.base_temperature = base_temperature
        self.base_humidity = base_humidity
        self.base_pressure = base_pressure
        self.seasonal_variation = seasonal_variation
    
    def generate(
        self,
        duration_days: int,
        start_date: Optional[datetime] = None,
        hourly: bool = True
    ) -> Dict[str, np.ndarray]:
        """
        Generate synthetic weather data.
        
        Args:
            duration_days: Number of days to generate
            start_date: Starting date (defaults to today)
            hourly: Whether to generate hourly (True) or daily (False) data
            
        Returns:
            Dictionary containing weather parameters over time
        """
        if start_date is None:
            start_date = datetime.now()
        
        # Generate time points
        points_per_day = 24 if hourly else 1
        num_points = duration_days * points_per_day
        timestamps = [start_date + timedelta(hours=i/points_per_day*24) 
                     for i in range(num_points)]
        
        # Generate base patterns
        time_array = np.arange(num_points)
        
        # Temperature with daily and seasonal variations
        temp = self.base_temperature + \
               5 * np.sin(2*np.pi*time_array/(points_per_day)) + \
               (10 * np.sin(2*np.pi*time_array/(365*points_per_day)) 
                if self.seasonal_variation else 0)
        
        # Humidity with daily variation
        humidity = self.base_humidity + \
                  0.1 * np.sin(2*np.pi*time_array/(points_per_day)) + \
                  0.05 * np.random.randn(num_points)
        humidity = np.clip(humidity, 0, 1)
        
        # Pressure with weather system variations
        pressure = self.base_pressure + \
                  10 * np.sin(2*np.pi*time_array/(7*points_per_day)) + \
                  5 * np.random.randn(num_points)
        
        return {
            'timestamps': timestamps,
            'temperature': temp,
            'humidity': humidity,
            'pressure': pressure
        }

class AirQualityGenerator:
    """Generate synthetic air quality data."""
    
    def __init__(
        self,
        base_aqi: float = 50.0,
        base_pm25: float = 12.0,
        base_no2: float = 20.0
    ):
        self.base_aqi = base_aqi
        self.base_pm25 = base_pm25
        self.base_no2 = base_no2
    
    def generate(
        self,
        duration_days: int,
        start_date: Optional[datetime] = None,
        hourly: bool = True
    ) -> Dict[str, np.ndarray]:
        """
        Generate synthetic air quality data.
        
        Args:
            duration_days: Number of days to generate
            start_date: Starting date (defaults to today)
            hourly: Whether to generate hourly (True) or daily (False) data
            
        Returns:
            Dictionary containing air quality parameters over time
        """
        if start_date is None:
            start_date = datetime.now()
        
        points_per_day = 24 if hourly else 1
        num_points = duration_days * points_per_day
        timestamps = [start_date + timedelta(hours=i/points_per_day*24) 
                     for i in range(num_points)]
        
        # Generate daily patterns with rush hour peaks
        time_array = np.arange(num_points)
        
        # AQI with daily variation and random events
        aqi = self.base_aqi + \
              20 * np.sin(2*np.pi*time_array/(points_per_day)) + \
              10 * np.random.randn(num_points)
        aqi = np.clip(aqi, 0, 500)
        
        # PM2.5 with correlation to AQI
        pm25 = self.base_pm25 + \
               (aqi - self.base_aqi) * 0.2 + \
               2 * np.random.randn(num_points)
        pm25 = np.clip(pm25, 0, 500)
        
        # NO2 with traffic patterns
        no2 = self.base_no2 + \
              15 * np.sin(2*np.pi*time_array/(points_per_day)) + \
              5 * np.random.randn(num_points)
        no2 = np.clip(no2, 0, 200)
        
        return {
            'timestamps': timestamps,
            'aqi': aqi,
            'pm25': pm25,
            'no2': no2
        }

class ExposureGenerator:
    """Generate synthetic light and noise exposure data."""
    
    def __init__(
        self,
        base_light: float = 500.0,  # lux
        base_noise: float = 45.0,   # dB
        sampling_rate: float = 1/60  # One sample per minute
    ):
        self.base_light = base_light
        self.base_noise = base_noise
        self.sampling_rate = sampling_rate
    
    def generate(
        self,
        duration_hours: float,
        activity_events: List[Tuple[float, str, float]] = None
    ) -> Dict[str, np.ndarray]:
        """
        Generate synthetic exposure data with activity-based variations.
        
        Args:
            duration_hours: Duration in hours
            activity_events: List of (time, activity_type, intensity) tuples
            
        Returns:
            Dictionary containing exposure parameters over time
        """
        num_samples = int(duration_hours * 3600 * self.sampling_rate)
        time = np.linspace(0, duration_hours * 3600, num_samples)
        
        # Generate base patterns
        light = np.ones(num_samples) * self.base_light
        noise = np.ones(num_samples) * self.base_noise
        
        # Add natural daily variation
        day_progress = (time % 86400) / 86400  # Time of day (0-1)
        light += 1000 * np.sin(2*np.pi*day_progress)  # Daylight variation
        
        # Add activity-based variations
        if activity_events:
            for event_time, activity_type, intensity in activity_events:
                event_idx = int(event_time * 3600 * self.sampling_rate)
                duration_idx = int(0.5 * 3600 * self.sampling_rate)  # 30-minute effect
                
                if activity_type == 'indoor':
                    light[event_idx:event_idx+duration_idx] *= 0.5
                    noise[event_idx:event_idx+duration_idx] += 20 * intensity
                elif activity_type == 'outdoor':
                    light[event_idx:event_idx+duration_idx] *= 2.0
                    noise[event_idx:event_idx+duration_idx] += 15 * intensity
                elif activity_type == 'screen':
                    light[event_idx:event_idx+duration_idx] += 300 * intensity
                    noise[event_idx:event_idx+duration_idx] += 5 * intensity
        
        # Add random variations
        light += 100 * np.random.randn(num_samples)
        noise += 5 * np.random.randn(num_samples)
        
        # Ensure physical limits
        light = np.clip(light, 0, 100000)  # Max 100k lux (bright sunlight)
        noise = np.clip(noise, 30, 120)     # 30-120 dB range
        
        return {
            'time': time,
            'light': light,
            'noise': noise
        }

def generate_environmental_scenario(
    duration_days: int,
    include_events: bool = True
) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Generate a complete environmental scenario with coordinated variations.
    
    Args:
        duration_days: Number of days to generate
        include_events: Whether to include specific events/activities
        
    Returns:
        Dictionary containing all environmental parameters
    """
    # Initialize generators
    weather_gen = WeatherGenerator(seasonal_variation=True)
    air_gen = AirQualityGenerator()
    exposure_gen = ExposureGenerator()
    
    # Generate base patterns
    weather_data = weather_gen.generate(duration_days, hourly=True)
    air_data = air_gen.generate(duration_days, hourly=True)
    
    # Generate exposure data with activities
    activity_events = []
    if include_events:
        # Add some example activities
        for day in range(duration_days):
            # Morning outdoor activity
            activity_events.append((day * 24 + 8, 'outdoor', 0.7))
            # Midday indoor work
            activity_events.append((day * 24 + 13, 'indoor', 0.8))
            # Evening screen time
            activity_events.append((day * 24 + 20, 'screen', 0.9))
    
    exposure_data = exposure_gen.generate(duration_days * 24, activity_events)
    
    return {
        'weather': weather_data,
        'air_quality': air_data,
        'exposure': exposure_data
    } 