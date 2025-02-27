#!/usr/bin/env python3
"""
Populate the database with test data for drift detection visualization.
"""
import sys
import os
from pathlib import Path
import logging
import argparse
from datetime import datetime, timedelta
import random
import json

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.core.models.database import User, Prediction
from app.core.database import get_db
from sqlalchemy.orm import Session
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from app.core.config.settings import Settings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def get_database_session():
    """Get a database session."""
    settings = Settings()
    engine = create_engine(settings.DATABASE_URL or "sqlite:///./test.db")
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    return SessionLocal()

def get_user(db: Session, username: str):
    """Get a user by username."""
    return db.query(User).filter(User.username == username).first()

def create_prediction_entries(db: Session, user_id: int, features_list, timestamps, probabilities, actual_values):
    """Create prediction entries in the database."""
    entries = []
    
    for i, (features, timestamp, probability, actual) in enumerate(
        zip(features_list, timestamps, probabilities, actual_values)
    ):
        # Create prediction entry
        prediction_entry = Prediction(
            user_id=user_id,
            features=features,
            probability=float(probability),
            prediction=bool(probability > 0.5),  # Convert to boolean based on threshold
            actual=bool(actual) if actual is not None else None,
            created_at=timestamp,
            drift_detected=False  # Will be updated by drift detection API
        )
        entries.append(prediction_entry)
        
    # Add all entries to the database
    db.add_all(entries)
    db.commit()
    
    logger.info(f"Created {len(entries)} prediction entries")
    return entries

def main():
    parser = argparse.ArgumentParser(description="Populate database with test data for drift detection")
    parser.add_argument("--username", default="testuser", help="Username to associate data with")
    parser.add_argument("--days", type=int, default=90, help="Number of days of data to generate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--drift-points", type=int, default=3, help="Number of drift points to introduce")
    
    args = parser.parse_args()
    
    # Initialize generator
    from app.core.data.test_data_generator import TestDataGenerator
    generator = TestDataGenerator(seed=args.seed)
    
    # Get database session
    db = get_database_session()
    
    try:
        # Get user
        user = get_user(db, args.username)
        if not user:
            logger.error(f"User {args.username} not found")
            return
        
        logger.info(f"Generating {args.days} days of data for user {user.username} (ID: {user.id})")
        
        # Generate data with drift
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=args.days)
        
        # Generate timestamps
        timestamps = []
        current_date = start_date
        while current_date <= end_date:
            timestamps.append(current_date)
            # Add random hours to make timestamps more realistic
            hours_increment = random.randint(2, 8)
            current_date = current_date + timedelta(hours=hours_increment)
        
        # Determine drift points
        n_samples = len(timestamps)
        drift_indices = []
        
        if args.drift_points > 0 and n_samples > 50:
            # Ensure drift points are spaced out
            min_spacing = n_samples // (args.drift_points + 1)
            
            for i in range(args.drift_points):
                # Add some randomness to drift point positions
                base_idx = (i + 1) * min_spacing
                jitter = random.randint(-min_spacing//4, min_spacing//4)
                drift_idx = max(30, min(n_samples - 20, base_idx + jitter))
                drift_indices.append(drift_idx)
        
        logger.info(f"Introducing drift at indices: {drift_indices}")
        
        # Generate feature data
        features_list = []
        probabilities = []
        actual_values = []
        
        current_drift_idx = 0
        current_drift_factor = 0.0
        
        for i in range(n_samples):
            # Check if we need to introduce drift
            if current_drift_idx < len(drift_indices) and i >= drift_indices[current_drift_idx]:
                # Introduce drift by shifting feature distributions
                logger.info(f"Introducing drift at sample {i}")
                current_drift_factor = random.uniform(0.5, 1.5)
                current_drift_idx += 1
            
            # Generate features with or without drift
            include_drift = current_drift_factor > 0
            record = generator.generate_single_record(
                include_drift=include_drift,
                drift_factor=current_drift_factor,
                time_index=i
            )
            features_list.append(record)
            
            # Calculate migraine probability
            probability = generator.calculate_migraine_probability(record)
            probabilities.append(probability)
            
            # Generate actual value (binary outcome)
            # Higher probability means higher chance of migraine
            threshold = 0.5
            actual = 1 if probability > threshold and random.random() < probability else 0
            actual_values.append(actual)
        
        # Create entries in database
        create_prediction_entries(db, user.id, features_list, timestamps, probabilities, actual_values)
        
        logger.info(f"Successfully populated database with {n_samples} samples and {len(drift_indices)} drift points")
        
    finally:
        db.close()

if __name__ == "__main__":
    main()
