
import pandas as pd
import numpy as np


def synthesize_data(df, n_synthetic_samples, noise_level=0.05):
    """
    Generate synthetic data by adding Gaussian noise to existing data points.

    Parameters:
    - df: A pandas DataFrame containing the original data.
    - n_synthetic_samples: Number of synthetic data points to generate.
    - noise_level: A float representing the noise level as a fraction of each feature's standard deviation.

    Returns:
    - A new DataFrame containing the original data and the synthesized data.
    """
    synthetic_data = []
    n_samples, n_features = df.shape
    
    # Ensure noise_level is positive and not too large
    noise_level = max(min(noise_level, 1.0), 0.01)
    
    for _ in range(n_synthetic_samples):
        # Randomly choose a sample to be the base for synthesis
        base_sample_idx = np.random.randint(0, n_samples)
        base_sample = df.iloc[base_sample_idx].values
        
        # Generate synthetic sample by adding Gaussian noise
        noise = np.random.normal(loc=0.0, scale=noise_level * df.std().values, size=n_features)
        synthetic_sample = base_sample + noise
        
        synthetic_data.append(synthetic_sample)
    
    # Create DataFrame from synthetic data and concatenate with original data
    synthetic_df = pd.DataFrame(synthetic_data, columns=df.columns)
    augmented_df = pd.concat([df, synthetic_df], ignore_index=True)
    
    return augmented_df