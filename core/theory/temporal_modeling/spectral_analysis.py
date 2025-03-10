"""
Spectral Analysis for Physiological Time Series Data.

This module provides theoretical components for spectral analysis of physiological time series data,
including Fourier transforms, wavelet analysis, power spectral density estimation, and spectral
entropy measures relevant to migraine prediction.
"""

import numpy as np
from scipy import signal
from scipy.fft import fft, fftfreq
from scipy import integrate
import pywt
from typing import Dict, List, Any, Tuple, Optional, Union, Callable

from core.theory.base import TheoryComponent


class SpectralAnalyzer(TheoryComponent):
    """
    Analyzer for theoretical spectral properties in physiological time series.
    
    This class provides methods for analyzing and characterizing the spectral
    properties of physiological time series data relevant to migraine prediction,
    including frequency domain analysis, wavelet transforms, and spectral entropy.
    """
    
    SPECTRAL_METRICS = {
        "bands": {
            "delta": (0.5, 4),       # 0.5-4 Hz (deep sleep, migraine aura)
            "theta": (4, 8),         # 4-8 Hz (drowsiness, stress, some migraine patterns)
            "alpha": (8, 13),        # 8-13 Hz (relaxation, closed eyes)
            "beta": (13, 30),        # 13-30 Hz (active thinking, focus)
            "gamma": (30, 100),      # 30-100 Hz (cognitive processing)
            "ultra_low": (0.0001, 0.5), # Circadian and ultradian rhythms
            "very_low": (0.003, 0.04),  # Blood pressure, thermoregulation
            "low": (0.04, 0.15),     # Sympathetic activity
            "high": (0.15, 0.4)      # Parasympathetic activity
        },
        "methods": ["fft", "wavelet", "welch", "periodogram", "multitaper"],
        "entropy_types": ["spectral", "approximate", "sample", "permutation"]
    }
    
    def __init__(self, data_type: str = "general", description: str = ""):
        """
        Initialize the spectral analyzer with data type.
        
        Args:
            data_type: Type of physiological data (e.g., "eeg", "hrv", "general")
            description: Optional description
        """
        super().__init__(description)
        self.data_type = data_type.lower()
        self.sampling_rate = None
        self.nyquist_freq = None
        
        # Map data types to relevant frequency bands
        self.relevant_bands = self._determine_relevant_bands(data_type)
        
    def _determine_relevant_bands(self, data_type: str) -> List[str]:
        """
        Determine the relevant frequency bands for the given data type.
        
        Args:
            data_type: Type of physiological data
            
        Returns:
            List of relevant band names
        """
        if data_type == "eeg":
            return ["delta", "theta", "alpha", "beta", "gamma"]
        elif data_type in ["ecg", "hrv", "heart"]:
            return ["ultra_low", "very_low", "low", "high"]
        elif data_type in ["emg", "muscle"]:
            return ["alpha", "beta", "gamma"]
        else:
            # For general or unknown data, include all bands
            return list(self.SPECTRAL_METRICS["bands"].keys())
            
    def analyze(self, time_series: np.ndarray, sampling_rate: float, 
                method: str = "fft", window_size: int = None) -> Dict[str, Any]:
        """
        Analyze spectral components of the time series.
        
        Args:
            time_series: Time series data array
            sampling_rate: Sampling rate in Hz
            method: Analysis method ("fft", "wavelet", "welch", etc.)
            window_size: Window size for windowed methods (defaults to N/8)
            
        Returns:
            Dictionary containing spectral analysis results
        """
        self.sampling_rate = sampling_rate
        self.nyquist_freq = sampling_rate / 2
        
        if window_size is None:
            window_size = len(time_series) // 8
            
        # Basic validation
        if len(time_series) < 2:
            raise ValueError("Time series must contain at least 2 data points")
            
        # Call appropriate analysis method
        if method == "fft":
            spectral_data = self._analyze_fft(time_series)
        elif method == "wavelet":
            spectral_data = self._analyze_wavelet(time_series)
        elif method == "welch":
            spectral_data = self._analyze_welch(time_series, window_size)
        elif method == "periodogram":
            spectral_data = self._analyze_periodogram(time_series)
        elif method == "multitaper":
            spectral_data = self._analyze_multitaper(time_series)
        else:
            raise ValueError(f"Unknown method: {method}")
            
        # Add band-specific analysis
        band_powers = self._calculate_band_powers(
            spectral_data["frequencies"],
            spectral_data["power_spectrum"]
        )
        spectral_data["band_powers"] = band_powers
        
        # Calculate entropy measures
        spectral_data["entropy"] = self._calculate_entropy(
            time_series, 
            spectral_data["power_spectrum"]
        )
        
        # Add theoretical insights
        spectral_data["theoretical_insights"] = self._get_theoretical_insights(
            band_powers, 
            spectral_data["entropy"]["spectral_entropy"]
        )
        
        return spectral_data
        
    def _analyze_fft(self, time_series: np.ndarray) -> Dict[str, Any]:
        """
        Perform FFT analysis on time series.
        
        Args:
            time_series: Time series data array
            
        Returns:
            Dictionary with FFT results
        """
        n = len(time_series)
        
        # Apply window to reduce spectral leakage
        window = signal.windows.hann(n)
        windowed_data = time_series * window
        
        # Compute FFT
        yf = fft(windowed_data)
        # Take only first half (positive frequencies)
        yf = yf[:n//2]
        # Compute power spectrum (normalize by window energy)
        power_spectrum = np.abs(yf)**2 / (np.sum(window**2) / 2)
        
        # Frequency axis
        xf = fftfreq(n, 1/self.sampling_rate)[:n//2]
        
        # Theoretical frequency resolution
        freq_resolution = self.sampling_rate / n
        
        return {
            "frequencies": xf,
            "power_spectrum": power_spectrum,
            "amplitude_spectrum": np.abs(yf),
            "phase_spectrum": np.angle(yf),
            "method": "fft",
            "window": "hann",
            "freq_resolution": freq_resolution
        }
    
    def _analyze_wavelet(self, time_series: np.ndarray) -> Dict[str, Any]:
        """
        Perform wavelet analysis on time series.
        
        Args:
            time_series: Time series data array
            
        Returns:
            Dictionary with wavelet analysis results
        """
        # Determine appropriate wavelet scales
        n = len(time_series)
        dt = 1 / self.sampling_rate
        
        # Create logarithmically spaced wavelet scales
        # This range is suitable for most physiological signals
        scales = np.logspace(0.5, np.log10(n//4), num=50)
        
        # Calculate frequencies corresponding to scales for 'morl' wavelet
        wavelet = 'morl'  # Morlet wavelet - good for physiological data
        central_freq = pywt.central_frequency(wavelet)
        frequencies = central_freq / (scales * dt)
        
        # Perform continuous wavelet transform
        coef, freqs = pywt.cwt(time_series, scales, wavelet, dt)
        
        # Calculate power spectrum (mean across time)
        power_spectrum = np.mean(np.abs(coef)**2, axis=1)
        
        # Reorder to be ascending in frequency
        idx = frequencies.argsort()
        frequencies = frequencies[idx]
        power_spectrum = power_spectrum[idx]
        
        return {
            "frequencies": frequencies,
            "power_spectrum": power_spectrum,
            "wavelet_coefficients": coef,
            "wavelet_scales": scales,
            "method": "wavelet",
            "wavelet_type": wavelet,
            "time_frequency_resolution": "adaptive"
        }
        
    def _analyze_welch(self, 
                       time_series: np.ndarray, 
                       window_size: int) -> Dict[str, Any]:
        """
        Perform Welch's method for power spectral density estimation.
        
        Args:
            time_series: Time series data array
            window_size: Window size for segmentation
            
        Returns:
            Dictionary with Welch's method results
        """
        # Use Welch's method for power spectral density
        freqs, psd = signal.welch(
            time_series, 
            fs=self.sampling_rate, 
            nperseg=window_size,
            scaling='density'
        )
        
        return {
            "frequencies": freqs,
            "power_spectrum": psd,
            "method": "welch",
            "nperseg": window_size,
            "window": "hann",  # Default in scipy
            "freq_resolution": self.sampling_rate / window_size
        }
    
    def _analyze_periodogram(self, time_series: np.ndarray) -> Dict[str, Any]:
        """
        Compute periodogram power spectral density estimate.
        
        Args:
            time_series: Time series data array
            
        Returns:
            Dictionary with periodogram results
        """
        n = len(time_series)
        
        # Apply window to reduce spectral leakage
        window = signal.windows.hann(n)
        windowed_data = time_series * window
        
        # Compute periodogram
        freqs, psd = signal.periodogram(
            windowed_data,
            fs=self.sampling_rate,
            window='hann',
            scaling='density'
        )
        
        return {
            "frequencies": freqs,
            "power_spectrum": psd,
            "method": "periodogram",
            "window": "hann",
            "freq_resolution": self.sampling_rate / n
        }
    
    def _analyze_multitaper(self, time_series: np.ndarray) -> Dict[str, Any]:
        """
        Perform multitaper spectral analysis.
        
        Args:
            time_series: Time series data array
            
        Returns:
            Dictionary with multitaper analysis results
        """
        # Determine appropriate NW product and number of tapers
        n = len(time_series)
        nw = 4.0  # Time-bandwidth product
        k = 7     # Number of tapers
        
        try:
            # Multitaper spectral estimation
            freqs, psd, _ = signal.multitaper.psd(
                time_series,
                fs=self.sampling_rate,
                NW=nw,
                k=k
            )
            
            return {
                "frequencies": freqs,
                "power_spectrum": psd,
                "method": "multitaper",
                "time_bandwidth_product": nw,
                "num_tapers": k,
                "freq_resolution": nw * self.sampling_rate / n
            }
        except:
            # Fallback to Welch's method if multitaper fails
            return self._analyze_welch(time_series, n//8)
            
    def _calculate_band_powers(self, 
                              frequencies: np.ndarray, 
                              power_spectrum: np.ndarray) -> Dict[str, float]:
        """
        Calculate power in different frequency bands.
        
        Args:
            frequencies: Frequency array
            power_spectrum: Power spectrum array
            
        Returns:
            Dictionary with band powers and relative powers
        """
        results = {
            "absolute": {},
            "relative": {},
            "peak_frequency": {},
            "central_frequency": {}
        }
        
        total_power = integrate.trapezoid(power_spectrum, frequencies)
        
        # Calculate power in each relevant band
        for band_name in self.relevant_bands:
            if band_name in self.SPECTRAL_METRICS["bands"]:
                low, high = self.SPECTRAL_METRICS["bands"][band_name]
                
                # Skip bands above Nyquist frequency
                if low >= self.nyquist_freq:
                    continue
                    
                # Adjust high frequency to Nyquist if needed
                high = min(high, self.nyquist_freq)
                
                # Find indices corresponding to band
                idx = np.logical_and(frequencies >= low, frequencies <= high)
                
                if np.any(idx):
                    # Calculate absolute power in band
                    band_power = integrate.trapezoid(power_spectrum[idx], frequencies[idx])
                    results["absolute"][band_name] = band_power
                    
                    # Calculate relative power
                    results["relative"][band_name] = band_power / total_power if total_power > 0 else 0
                    
                    # Find peak frequency in band
                    if np.any(power_spectrum[idx]):
                        peak_idx = np.argmax(power_spectrum[idx])
                        results["peak_frequency"][band_name] = frequencies[idx][peak_idx]
                        
                        # Calculate central frequency (weighted average)
                        if band_power > 0:
                            central_freq = np.sum(frequencies[idx] * power_spectrum[idx]) / np.sum(power_spectrum[idx])
                            results["central_frequency"][band_name] = central_freq
        
        # Calculate ratios commonly used in physiological analysis
        if "alpha" in results["absolute"] and "beta" in results["absolute"]:
            if results["absolute"]["beta"] > 0:
                results["ratios"] = {
                    "alpha_beta": results["absolute"]["alpha"] / results["absolute"]["beta"]
                }
                
        # Calculate spectral edge frequencies
        results["spectral_edge"] = {
            "95": self._calculate_spectral_edge(frequencies, power_spectrum, 0.95),
            "90": self._calculate_spectral_edge(frequencies, power_spectrum, 0.90),
            "50": self._calculate_spectral_edge(frequencies, power_spectrum, 0.50)  # Median frequency
        }
        
        return results
        
    def _calculate_spectral_edge(self, 
                                frequencies: np.ndarray, 
                                power_spectrum: np.ndarray, 
                                percentile: float = 0.95) -> float:
        """
        Calculate spectral edge frequency.
        
        Args:
            frequencies: Frequency array
            power_spectrum: Power spectrum array
            percentile: Percentile for spectral edge (0.0-1.0)
            
        Returns:
            Spectral edge frequency
        """
        if len(frequencies) == 0 or np.all(power_spectrum == 0):
            return 0
            
        # Calculate cumulative power
        total_power = np.sum(power_spectrum)
        
        if total_power == 0:
            return 0
            
        cumulative_power = np.cumsum(power_spectrum) / total_power
        
        # Find spectral edge
        edge_idx = np.searchsorted(cumulative_power, percentile)
        
        if edge_idx >= len(frequencies):
            edge_idx = len(frequencies) - 1
            
        return frequencies[edge_idx]
        
    def _calculate_entropy(self, 
                          time_series: np.ndarray, 
                          power_spectrum: np.ndarray) -> Dict[str, float]:
        """
        Calculate various entropy measures.
        
        Args:
            time_series: Time series data array
            power_spectrum: Power spectrum array
            
        Returns:
            Dictionary with entropy measures
        """
        results = {}
        
        # Spectral entropy
        norm_power = power_spectrum / np.sum(power_spectrum) if np.sum(power_spectrum) > 0 else np.zeros_like(power_spectrum)
        
        # Remove zeros to avoid log(0)
        norm_power = norm_power[norm_power > 0]
        
        if len(norm_power) > 0:
            spectral_entropy = -np.sum(norm_power * np.log2(norm_power))
            # Normalize by log2(N)
            spectral_entropy /= np.log2(len(norm_power))
            results["spectral_entropy"] = spectral_entropy
        else:
            results["spectral_entropy"] = 0
            
        # Sample entropy (requires time series)
        # We'll use a simplified calculation with fixed parameters
        try:
            sample_entropy = self._sample_entropy(time_series, m=2, r=0.2*np.std(time_series))
            results["sample_entropy"] = sample_entropy
        except:
            results["sample_entropy"] = None
            
        return results
        
    def _sample_entropy(self, 
                       time_series: np.ndarray, 
                       m: int = 2, 
                       r: float = 0.2) -> float:
        """
        Calculate sample entropy - simplified implementation.
        
        Args:
            time_series: Time series data array
            m: Embedding dimension
            r: Tolerance
            
        Returns:
            Sample entropy value
        """
        N = len(time_series)
        
        if N <= m+1:
            return 0
            
        # Create templates of length m and m+1
        def create_templates(data, m_len):
            templates = np.zeros((N-m_len, m_len))
            for i in range(N-m_len):
                templates[i] = data[i:i+m_len]
            return templates
            
        # Count matches
        def count_matches(templates, r_threshold):
            count = 0
            for i in range(len(templates)):
                # Calculate Chebyshev distances
                distances = np.max(np.abs(templates - templates[i]), axis=1)
                # Count matches excluding self-match
                count += np.sum(distances < r_threshold) - 1
            return count
            
        # Create templates
        temp_m = create_templates(time_series, m)
        temp_m1 = create_templates(time_series, m+1)
        
        # Count matches
        count_m = count_matches(temp_m, r)
        count_m1 = count_matches(temp_m1, r)
        
        # Avoid division by zero
        if count_m == 0:
            return float('inf')
            
        # Calculate sample entropy
        return -np.log(count_m1 / count_m)
        
    def _get_theoretical_insights(self, 
                                 band_powers: Dict[str, Dict[str, float]], 
                                 spectral_entropy: float) -> List[str]:
        """
        Generate theoretical insights based on spectral analysis.
        
        Args:
            band_powers: Band power information
            spectral_entropy: Spectral entropy value
            
        Returns:
            List of theoretical insights
        """
        insights = []
        
        # Insight about spectral complexity
        if spectral_entropy > 0.8:
            insights.append("High spectral entropy indicates complex, possibly chaotic dynamics that may need nonlinear models.")
        elif spectral_entropy < 0.4:
            insights.append("Low spectral entropy suggests regular, predictable patterns that may be modeled with simpler techniques.")
        
        # Insights about dominant frequencies
        relative_powers = band_powers.get("relative", {})
        if relative_powers:
            # Find dominant band
            dominant_band = max(relative_powers.items(), key=lambda x: x[1]) if relative_powers else (None, 0)
            
            if dominant_band[0]:
                band_name, power = dominant_band
                if power > 0.5:
                    insights.append(f"Strong {band_name} band dominance (>50%) suggests examination of corresponding physiological processes.")
                
                # Specific physiological insights
                if band_name == "delta" and self.data_type == "eeg":
                    insights.append("Dominant delta activity may indicate deep sleep or pathological conditions relevant to migraine aura.")
                elif band_name == "alpha" and self.data_type == "eeg":
                    insights.append("Dominant alpha activity suggests relaxed wakefulness; changes may precede migraine onset.")
                elif band_name == "low" and self.data_type in ["hrv", "ecg"]:
                    insights.append("Dominant low-frequency HRV suggests sympathetic nervous system activation, possibly related to stress responses.")
        
        # Insight about temporal modeling approaches
        insights.append(f"The spectral characteristics suggest {'nonstationary' if spectral_entropy > 0.7 else 'relatively stationary'} dynamics, indicating the need for {'adaptive' if spectral_entropy > 0.7 else 'standard'} temporal modeling approaches.")
        
        return insights
        
    def compare_signals(self, 
                       signals: List[np.ndarray],
                       labels: List[str],
                       sampling_rate: float,
                       method: str = "welch") -> Dict[str, Any]:
        """
        Compare multiple physiological signals.
        
        Args:
            signals: List of time series data arrays
            labels: List of labels for each signal
            sampling_rate: Sampling rate in Hz
            method: Analysis method
            
        Returns:
            Dictionary with comparison results
        """
        if len(signals) != len(labels):
            raise ValueError("Number of signals must match number of labels")
            
        comparison_results = {
            "individual_analyses": [],
            "band_power_comparison": {},
            "entropy_comparison": {},
            "coherence": {},
            "theoretical_implications": []
        }
        
        # Analyze each signal
        for i, (signal, label) in enumerate(zip(signals, labels)):
            analysis = self.analyze(signal, sampling_rate, method)
            comparison_results["individual_analyses"].append({
                "label": label,
                "analysis": analysis
            })
            
            # Extract key metrics for comparison
            for band in self.relevant_bands:
                if band in analysis["band_powers"]["relative"]:
                    if band not in comparison_results["band_power_comparison"]:
                        comparison_results["band_power_comparison"][band] = {}
                    comparison_results["band_power_comparison"][band][label] = analysis["band_powers"]["relative"][band]
            
            comparison_results["entropy_comparison"][label] = analysis["entropy"]["spectral_entropy"]
        
        # Calculate pairwise coherence
        if len(signals) > 1:
            for i in range(len(signals)):
                for j in range(i+1, len(signals)):
                    label_pair = f"{labels[i]}_{labels[j]}"
                    
                    # Calculate coherence
                    try:
                        f, Cxy = signal.coherence(signals[i], signals[j], fs=sampling_rate)
                        
                        # Average coherence in relevant bands
                        band_coherence = {}
                        for band in self.relevant_bands:
                            if band in self.SPECTRAL_METRICS["bands"]:
                                low, high = self.SPECTRAL_METRICS["bands"][band]
                                
                                # Skip bands above Nyquist frequency
                                if low >= self.nyquist_freq:
                                    continue
                                    
                                # Adjust high frequency to Nyquist if needed
                                high = min(high, self.nyquist_freq)
                                
                                # Find indices corresponding to band
                                idx = np.logical_and(f >= low, f <= high)
                                
                                if np.any(idx):
                                    band_coherence[band] = np.mean(Cxy[idx])
                        
                        comparison_results["coherence"][label_pair] = {
                            "full_spectrum": {"freqs": f.tolist(), "coherence": Cxy.tolist()},
                            "band_averages": band_coherence
                        }
                    except:
                        comparison_results["coherence"][label_pair] = None
        
        # Generate theoretical implications
        comparison_results["theoretical_implications"] = self._generate_comparison_insights(
            comparison_results["band_power_comparison"],
            comparison_results["entropy_comparison"],
            comparison_results["coherence"]
        )
        
        return comparison_results
    
    def _generate_comparison_insights(self,
                                     band_comparison: Dict[str, Dict[str, float]],
                                     entropy_comparison: Dict[str, float],
                                     coherence: Dict[str, Any]) -> List[str]:
        """
        Generate theoretical insights from signal comparisons.
        
        Args:
            band_comparison: Band power comparison data
            entropy_comparison: Entropy comparison data
            coherence: Coherence data
            
        Returns:
            List of theoretical insights
        """
        insights = []
        
        # Insights about relative entropy
        if len(entropy_comparison) > 1:
            # Find signal with highest/lowest entropy
            max_entropy = max(entropy_comparison.items(), key=lambda x: x[1])
            min_entropy = min(entropy_comparison.items(), key=lambda x: x[1])
            
            if max_entropy[1] > 0.7 and min_entropy[1] < 0.5:
                insights.append(f"Signal '{max_entropy[0]}' shows significantly higher complexity than '{min_entropy[0]}', suggesting different underlying physiological processes that may require separate modeling approaches.")
        
        # Insights about coherence
        if coherence:
            for pair, coh_data in coherence.items():
                if coh_data and "band_averages" in coh_data:
                    # Find band with highest coherence
                    band_coh = coh_data["band_averages"]
                    if band_coh:
                        max_band = max(band_coh.items(), key=lambda x: x[1])
                        if max_band[1] > 0.7:
                            insights.append(f"High coherence (>{max_band[1]:.2f}) in {max_band[0]} band between {pair} suggests strong physiological coupling with potential predictive value for migraine onset.")
                        elif max_band[1] < 0.3:
                            insights.append(f"Low coherence in all bands between {pair} suggests independent processes that should be modeled separately.")
        
        # Theoretical implications for temporal modeling
        insights.append("Spectral differences between signals suggest a multi-channel approach for temporal modeling, integrating the distinct physiological processes while capturing their interactions.")
        
        return insights
    
    def get_formal_definition(self) -> str:
        """
        Get formal mathematical definition of spectral analysis.
        
        Returns:
            String with formal definition
        """
        definition = """
        Spectral Analysis Formal Definition:
        
        For a discrete time series x[n] of length N, the discrete Fourier transform X[k] is defined as:
        
        X[k] = ∑_{n=0}^{N-1} x[n] · e^{-j2πkn/N}, k = 0, 1, ..., N-1
        
        where j is the imaginary unit.
        
        The power spectral density (PSD) P[k] is:
        
        P[k] = |X[k]|²/N
        
        For physiological time series, the spectral content is typically divided into frequency bands, with the power in band [f₁, f₂] calculated as:
        
        P_{band} = ∫_{f₁}^{f₂} P(f) df
        
        For evolutionary spectral analysis with wavelets, the continuous wavelet transform W(a,b) is:
        
        W(a,b) = (1/√a) ∫_{-∞}^{∞} x(t) · ψ*((t-b)/a) dt
        
        where ψ* is the complex conjugate of the mother wavelet, a is the scale parameter, and b is the translation parameter.
        
        Spectral entropy H is defined as:
        
        H = -∑_{k} p[k] · log₂(p[k])
        
        where p[k] = P[k]/∑P[k] is the normalized PSD.
        
        These theoretical formulations provide the foundation for analyzing the spectral properties of physiological signals relevant to migraine prediction.
        """
        
        return definition 