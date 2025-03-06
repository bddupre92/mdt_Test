#!/usr/bin/env python3
"""
generate_synthetic.py
---------------------
Generates a robust, realistic synthetic migraine dataset for multiple patients
over daily records. Includes:
- Weather data (barometric pressure, temperature, humidity) with seasonal variation
- Hormonal cycle for female patients
- Sleep hours, stress level, dietary triggers (caffeine, alcohol, skip_meal, etc.)
- Wearable-like data (heart rate as a proxy)
- Concept drift across different time phases
- Missing values and anomalies
- Binary migraine occurrence (with optional severity)

This script serves as a foundation for testing advanced meta-optimizers
(GA/DE/PSO/GWO/ACO/ES) in feature selection/hyperparameter tuning.
"""

import numpy as np
import pandas as pd

def generate_synthetic_migraine_data(
    n_patients=50,
    n_days=365,
    start_date="2023-01-01",
    pct_female=0.6,
    missing_rate=0.1,
    anomaly_rate=0.01,
    drift_phases=2,
    random_seed=42,
    include_severity=False
):
    """
    Generates a comprehensive synthetic migraine dataset.

    Parameters
    ----------
    n_patients : int
        Number of patients to simulate.
    n_days : int
        Number of daily records per patient (before drift expansions).
    start_date : str
        Start date (YYYY-MM-DD) for time series. We'll generate daily rows from here.
    pct_female : float
        Proportion of patients who are female, used to assign hormone cycles.
    missing_rate : float
        Fraction of data randomly replaced with NaNs (simulating missingness).
    anomaly_rate : float
        Fraction of data replaced with random out-of-range anomalies (simulating device/survey errors).
    drift_phases : int
        Number of concept drift phases to simulate (1 means no drift, 2+ means multiple shifts).
    random_seed : int
        Random seed for reproducibility.
    include_severity : bool
        Whether to include a numeric migraine severity column (1-10 scale) if True.

    Returns
    -------
    df : pd.DataFrame
        A pandas DataFrame with columns:
        ['patient_id', 'date', 'female', 'phase',
         'barometric_pressure', 'temperature', 'humidity',
         'hormone_cycle_day', 'sleep_hours', 'stress_level', 'hrv', 'hr_rest',
         'caffeine_mg', 'alcohol_units', 'skip_meal',
         'migraine_occurred', (optional 'migraine_severity'),
         'medication_usage', 'trigger_chocolate', 'trigger_red_wine', ...]
        Potentially more columns if needed, each row representing a day of data.
    """
    rng = np.random.default_rng(random_seed)

    # Convert start_date to a Pandas Timestamp
    start_dt = pd.to_datetime(start_date)

    # Step 1: Initialize patient-level attributes
    # -------------------------------------------------------------
    # patient_id: 0..n_patients-1
    # female: True/False, assigned based on pct_female
    n_female = int(n_patients * pct_female)
    n_male = n_patients - n_female
    # Shuffle so it's random who is female vs male
    female_flags = np.array([True]*n_female + [False]*n_male)
    rng.shuffle(female_flags)

    # We'll build up a list of data frames, one per patient, then concat
    all_patient_data = []

    # For concept drift, we subdivide n_days into drift_phases segments
    # e.g. if drift_phases=2 and n_days=365, we might do two segments ~ 182 days + 183 days
    # each segment we can shift the patient's triggers or environment
    # We'll define a function to break n_days into segments
    def split_days_into_phases(total_days, phases):
        # example: for phases=2 => [0, 182, 365]
        # for phases=3 => [0, 121, 243, 365]
        # we'll accumulate
        segment_sizes = np.linspace(0, total_days, num=phases+1, dtype=int)
        # segment_sizes will be array([0, 182, 365]) if phases=2
        return segment_sizes

    day_splits = split_days_into_phases(n_days, drift_phases)

    # We'll define some functions to simulate the day-level features:

    def simulate_weather(day_index, total_days, base_pressure=1013, rng=None):
        """
        Return barometric_pressure, temperature, humidity for a given day index
        with seasonal variation.
        """
        # Let's define a simple seasonal cycle for temperature
        # amplitude ~ 10C, baseline ~ 15C
        # day_of_year = day_index % 365
        # fraction_of_year = day_of_year / 365
        fraction_of_year = (day_index % 365) / 365.0
        # Seasonality for temperature: ~ 10 deg amplitude
        seasonal_temp = 15 + 10 * np.sin(2 * np.pi * fraction_of_year)
        # add random daily variation
        temp = seasonal_temp + rng.normal(0, 3)

        # barometric pressure: base +/- small daily random, plus occasional bigger swings
        pressure = base_pressure + rng.normal(0, 5)

        # humidity: 20-90% range, seasonal shift maybe
        # let's do a simple approach: mean ~ 60% + 20% * sin(...) + noise
        seasonal_humid = 60 + 20 * np.sin(2 * np.pi * fraction_of_year)
        humidity = seasonal_humid + rng.normal(0, 10)
        humidity = np.clip(humidity, 0, 100)

        return pressure, temp, humidity

    def simulate_hormone_cycle_day(day_index, cycle_length=28, rng=None):
        """
        Return day_in_cycle in [1..cycle_length].
        We'll just cycle day_in_cycle across time. 
        We won't do actual random variation here, but you could if you want.
        """
        return 1 + (day_index % cycle_length)

    def hormone_migraine_modifier(hormone_cycle_day):
        """
        If day ~ 27..28 or 1..2 => higher chance of migraines if female
        Could define a small multiplier or offset to migraine prob
        """
        # Late luteal phase ~ day 26-28 or day 1-2
        # we'll do a simple piecewise
        if hormone_cycle_day in [26, 27, 28, 1, 2]:
            return 0.3  # means +30% baseline risk
        return 0.0

    def simulate_sleep_hours(rng=None):
        """
        Typical adult range 4..10, mean ~7
        """
        return float(np.clip(rng.normal(7, 1.0), 3, 12))

    def simulate_stress_level(rng=None, prev_stress=None):
        """
        Daily stress 1..10. If we want continuity, we can move from prev_stress
        with some random step. If prev_stress is None, pick random init.
        """
        if prev_stress is None:
            val = rng.uniform(2, 8)
        else:
            # small random walk
            step = rng.normal(0, 1.5)
            val = prev_stress + step
        return float(np.clip(val, 0, 10))

    def simulate_hrv_or_heart_rate(rng=None):
        """
        For HRV, typical 50..150 ms. We'll do a normal around 100 with std ~ 20.
        For heart rate resting, typical 50..100. We'll do normal around 70 with std 10.
        """
        hrv = float(np.clip(rng.normal(100, 20), 30, 200))
        hr_rest = float(np.clip(rng.normal(70, 10), 40, 120))
        return hrv, hr_rest

    def simulate_diet_factors(rng=None):
        """
        Returns caffeine_mg, alcohol_units, skip_meal, plus optional triggers
        e.g. chocolate or red_wine flags
        """
        # caffeine: 0..400 mg
        caffeine = float(np.clip(rng.normal(80, 60), 0, 400))
        # alcohol: 0..3 units typically, though up to 5 in extreme
        # 1 unit ~ 1 standard drink (e.g. 5oz wine). We'll random
        # many days = 0
        # Use some skewed approach: ~70% zero, else 1-5
        if rng.random() < 0.7:
            alcohol = 0.0
        else:
            alcohol = float(np.clip(rng.normal(1.5, 1.0), 0, 5))

        # skip_meal: 0..1, ~15% chance
        skip_meal = int(rng.random() < 0.15)

        # triggers e.g. chocolate, red_wine
        # rarely. Let's say 10% days chocolate, 5% days red_wine
        # in synergy with the above
        choc_flag = int(rng.random() < 0.1)
        # red_wine if also had alcohol, small chance
        # or separate?
        red_wine_flag = 0
        if alcohol > 0.0 and rng.random() < 0.3:
            red_wine_flag = 1

        return caffeine, alcohol, skip_meal, choc_flag, red_wine_flag

    def medication_usage_logic(day_index, rng=None, base_freq=0.05):
        """
        Probability that the patient used acute medication this day.
        Possibly if they had a migraine the previous day or a high-likelihood day?
        We'll do a simple approach: 5% baseline usage, if day_index is
        in a cluster of migraines, usage might be higher.
        For now, let's do random approach with base_freq.
        """
        return int(rng.random() < base_freq)

    # Step 2: concept drift design
    # We'll define a function that returns a dict of 'patient_state' changes
    # e.g. { 'pressure_base': 1013, 'stress_offset': 0.0, 'migraine_base_rate': 0.1, ... }
    # Then after each phase shift, we update these.

    def generate_phase_params(phase_index, total_phases, rng, is_female):
        """
        Each phase we can shift or tweak patient's baseline. 
        For example:
        - If female, maybe hormone sensitivity is high in early phase, less in later.
        - If we want medication usage to reduce migraines in later phase, reduce migraine base rate.
        We'll create some random but structured changes.
        """
        # We define random offsets
        d = {}
        # baseline daily migraine prob for this phase 
        # assume ~ 0.05..0.15 for mild/ moderate. 
        # if is_female, might be slightly higher
        base_prob = rng.uniform(0.03, 0.12)
        if is_female:
            base_prob += 0.02
        # if phase_index>0 maybe we've lowered it or raised it
        # do random +/- 0.02 each phase
        base_prob += rng.normal(0, 0.02)
        base_prob = max(0.01, min(base_prob, 0.2))
        d['migraine_base_prob'] = base_prob

        # pressure_base shift? maybe if we assume they moved location in next phase
        # let's do small shift +/- up to 10 hPa
        d['pressure_base'] = 1013 + rng.integers(-5, 6)  # +/- 5
        # stress_offset 
        d['stress_offset'] = rng.normal(0, 1)

        # medication usage freq might start low, then if patient uses more, risk of MOH?
        # but let's keep it simple: 0.05 baseline + small random
        d['med_usage_prob'] = 0.05 + rng.normal(0, 0.02)
        d['med_usage_prob'] = np.clip(d['med_usage_prob'], 0.0, 0.2)

        # We can also define a 'hormone_sensitivity' for female
        if is_female:
            # how strongly hormone spikes add to migraine prob
            d['hormone_sensitivity'] = rng.uniform(0.2, 0.4)
        else:
            d['hormone_sensitivity'] = 0.0

        # time-of-day pattern or circadian can be simplified
        # but we won't handle sub-daily in this example.

        return d

    def daily_migraine_probability(
        day_features, phase_params, is_female, rng
    ):
        """
        day_features includes:
         - barometric_pressure, temperature, humidity
         - hormone_cycle_day
         - sleep_hours
         - stress_level
         - caffeine_mg, alcohol_units, skip_meal, choc_flag, red_wine_flag
        phase_params includes:
         - migraine_base_prob
         - hormone_sensitivity
         - stress_offset
         ...
        We'll combine them in a simple logistic-ish approach.
        """
        p = phase_params['migraine_base_prob']

        # weather effect if big drop in pressure from previous day?
        # We'll check it if we store 'prev_pressure' or we can store day_features for delta
        # For simplicity we won't do day-to-day delta here unless we keep track
        # but let's do a small penalty if pressure < 1005
        if day_features['barometric_pressure'] < 1005:
            p += 0.02  # small increment

        # temperature extremes (hot or cold) might add 0.01
        if day_features['temperature'] < 0 or day_features['temperature'] > 30:
            p += 0.01

        # humidity high, e.g. >80 => +0.01
        if day_features['humidity'] > 80:
            p += 0.01

        # hormone effect
        if is_female:
            day_in_cycle = day_features['hormone_cycle_day']
            # if day_in_cycle in [26,27,28,1,2], we add hormone_sensitivity
            if day_in_cycle in [26,27,28,1,2]:
                p += phase_params['hormone_sensitivity']

        # sleep effect: if <5 => +0.05
        if day_features['sleep_hours'] < 5:
            p += 0.05

        # stress effect: if stress_level>7 => +0.04
        stress_adj = day_features['stress_level'] + phase_params['stress_offset']
        if stress_adj > 7:
            p += 0.04

        # diet triggers
        # caffeine >300 => +0.02
        if day_features['caffeine_mg'] > 300:
            p += 0.02
        # alcohol >2 => +0.03, or if red_wine_flag => +0.04
        if day_features['alcohol_units'] > 2:
            p += 0.03
        if day_features['red_wine_flag'] == 1:
            p += 0.01  # stacked
        # skip_meal => +0.02
        if day_features['skip_meal'] == 1:
            p += 0.02
        # chocolate => +0.01
        if day_features['choc_flag'] == 1:
            p += 0.01

        # We can saturate p so it doesn't exceed ~0.8
        p = max(0, min(p, 0.8))
        return p

    # Step 3: Build data for each patient
    # -------------------------------------------------------------
    for pid in range(n_patients):
        is_female = female_flags[pid]
        # define # of total days. We might keep n_days * drift_phases or we do separate segments
        # Here we do n_days total, but it's splitted across phases
        # we'll accumulate rows in a list
        patient_rows = []
        # Generate the random param sets for each phase
        phase_param_list = [
            generate_phase_params(i, drift_phases, rng, is_female)
            for i in range(drift_phases)
        ]

        current_stress = float(rng.uniform(2, 8))  # init
        # to simulate cyc, we track day_in_cycle separately
        # we'll do day_in_cycle from 1..28 repeating for female, else 0 for male
        day_in_cycle = rng.integers(1, 29) if is_female else 0

        # Now for each phase, fill the day range
        for ph in range(drift_phases):
            start_day = day_splits[ph]
            end_day = day_splits[ph+1]
            phase_len = end_day - start_day
            # get param
            phase_params_ = phase_param_list[ph]

            for day_ix in range(phase_len):
                global_day_index = start_day + day_ix
                date_day = start_dt + pd.Timedelta(days=int(global_day_index))

                # simulate weather
                baro, temp, hum = simulate_weather(
                    global_day_index,
                    n_days,
                    base_pressure=phase_params_['pressure_base'],
                    rng=rng
                )
                # simulate hormone if female
                if is_female:
                    day_in_cycle = 1 + ((day_in_cycle) % 28)  # cycle it
                # simulate sleep
                sleep_h = simulate_sleep_hours(rng=rng)
                # stress (random walk around current)
                current_stress = simulate_stress_level(rng=rng, prev_stress=current_stress)
                # hrv, hr_rest
                hrv, hr_rest = simulate_hrv_or_heart_rate(rng=rng)
                # diet triggers
                caff, alc, skipmeal, choc, redwine = simulate_diet_factors(rng=rng)
                # medication usage
                med_use = medication_usage_logic(global_day_index, rng=rng, base_freq=phase_params_['med_usage_prob'])

                # combine into dict
                day_features = {
                    'barometric_pressure': baro,
                    'temperature': temp,
                    'humidity': hum,
                    'hormone_cycle_day': day_in_cycle if is_female else 0,
                    'sleep_hours': sleep_h,
                    'stress_level': current_stress,
                    'hrv': hrv,
                    'hr_rest': hr_rest,
                    'caffeine_mg': caff,
                    'alcohol_units': alc,
                    'skip_meal': skipmeal,
                    'choc_flag': choc,
                    'red_wine_flag': redwine,
                }

                # compute migraine prob
                mig_prob = daily_migraine_probability(day_features, phase_params_, is_female, rng)
                mig_occurred = int(rng.random() < mig_prob)

                # optional severity if migraine occurred
                mig_sev = None
                if include_severity and mig_occurred == 1:
                    # 1..10, let's pick ~ average 6
                    # if hormone triggered, might be higher
                    base_sev = rng.normal(6, 2)
                    if is_female and day_in_cycle in [26,27,28,1,2]:
                        base_sev += 1  # ~1 higher if hormonal
                    # clip 1..10
                    mig_sev = int(np.clip(base_sev, 1, 10))
                elif include_severity:
                    # no migraine => severity=0 or np.nan
                    mig_sev = 0

                row = {
                    'patient_id': pid,
                    'date': date_day,
                    'female': 1 if is_female else 0,
                    'phase': ph,
                    'barometric_pressure': baro,
                    'temperature': temp,
                    'humidity': hum,
                    'hormone_cycle_day': day_in_cycle if is_female else np.nan,
                    'sleep_hours': sleep_h,
                    'stress_level': current_stress,
                    'hrv': hrv,
                    'hr_rest': hr_rest,
                    'caffeine_mg': caff,
                    'alcohol_units': alc,
                    'skip_meal': skipmeal,
                    'trigger_chocolate': choc,
                    'trigger_red_wine': redwine,
                    'medication_usage': med_use,
                    'migraine_occurred': mig_occurred
                }
                if include_severity:
                    row['migraine_severity'] = mig_sev

                patient_rows.append(row)

        # build a df
        pat_df = pd.DataFrame(patient_rows)
        all_patient_data.append(pat_df)

    df = pd.concat(all_patient_data, ignore_index=True)

    # Step 4: Introduce missing data
    # -------------------------------------------------------------
    # For each numeric or category column except 'patient_id' and 'date', we'll randomly set some portion to NaN
    rng = np.random.default_rng(random_seed+123)  # a new seed offset for missing
    all_cols = [c for c in df.columns if c not in ['patient_id','date','migraine_occurred','migraine_severity']]
    # also might skip 'female','phase' from missing? we can skip phase/female from missing
    skip_missing_cols = {'patient_id','date','phase','female','migraine_occurred'}
    for c in df.columns:
        if c in skip_missing_cols:
            continue
        # create mask
        mask = rng.random(len(df)) < missing_rate
        df.loc[mask, c] = np.nan

    # Step 5: Introduce anomalies
    # -------------------------------------------------------------
    # We'll pick anomaly_rate fraction of numeric entries and replace with out-of-range values
    rng = np.random.default_rng(random_seed+999)
    numeric_cols = [c for c in df.columns if c not in skip_missing_cols and df[c].dtype in [np.float64, np.int64]]
    for c in numeric_cols:
        mask = rng.random(len(df)) < anomaly_rate
        # define out-of-range based on the col
        # e.g. barometric_pressure ~ 1013, anomaly => 2000 or -100
        # we'll do a random sign
        # let's find min and max of normal range
        # simpler approach: if c is 'barometric_pressure', let's do random 2000..2500 or negative
        # We'll do an approach: random pick from uniform(2.0*mean, 3.0*mean) or negative
        # but let's handle columns individually if needed.
        # For simplicity, do a huge outlier e.g. mean +- 5 std
        col_vals = df[c].dropna()
        if len(col_vals) < 10:
            continue
        mean_ = col_vals.mean()
        std_ = col_vals.std() if col_vals.std() else 1.0
        # anomaly could be mean + 5 std or mean - 5 std
        # we'll random pick sign
        anomaly_values = []
        for i in range(len(df)):
            # only apply if mask[i] = True
            anomaly_values.append(None)  # default
        for idx in df[mask].index:
            sign = rng.choice([-1,1])
            val = mean_ + sign*5*std_ + rng.normal(0, 2*std_)
            anomaly_values[idx] = val

        # now fill them in
        df.loc[mask, c] = anomaly_values

    # final shuffle or sort
    df.sort_values(by=['patient_id','date'], inplace=True)
    df.reset_index(drop=True, inplace=True)

    return df

def main():
    # Example usage
    df = generate_synthetic_migraine_data(
        n_patients=20,
        n_days=180,
        start_date="2024-01-01",
        pct_female=0.6,
        missing_rate=0.08,
        anomaly_rate=0.005,
        drift_phases=2,
        random_seed=42,
        include_severity=True
    )
    print("Generated synthetic migraine data with shape:", df.shape)
    print(df.head(20))

    # Save to CSV
    df.to_csv("synthetic_migraine_data.csv", index=False)
    print("Saved to synthetic_migraine_data.csv")

if __name__ == "__main__":
    main()
