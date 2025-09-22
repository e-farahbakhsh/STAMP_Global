'''
Feature Analysis

Author: Ehsan Farahbakhsh
Contact email: e.farahbakhsh@sydney.edu.au
Date last modified: 05/08/2025
'''


import numpy as np
import pandas as pd


def downsample(
        df,
        n_target,
        n_bins_age=50,
        n_bins_spatial=10,
        random_state=None,
    ):
    
    # Determine target sample size
    if len(df) < n_target:
        print("No downsampling required!")
        return df
    
    if not isinstance(random_state, np.random.Generator):
        random_state = np.random.default_rng(random_state)
    
    df = df.copy()
    
    # Bin by age (quantile bins)
    df['age_bin'] = pd.qcut(df['age (Ma)'], q=n_bins_age, duplicates='drop')
    
    # Within each age bin, spatially bin and sample
    sampled_indices = []
    
    for age_bin, age_group in df.groupby('age_bin'):
        age_group = age_group.copy()
    
        # Spatial binning within this age group
        age_group['lon_bin'] = pd.cut(age_group['lon'], bins=n_bins_spatial)
        age_group['lat_bin'] = pd.cut(age_group['lat'], bins=n_bins_spatial)
    
        # Group by spatial bins
        spatial_groups = age_group.groupby(['lon_bin', 'lat_bin'])
    
        # Evenly sample from spatial bins in this age bin
        n_total_bins = len(spatial_groups)
        if n_total_bins == 0:
            continue
        samples_per_bin = max(1, (n_target // n_bins_age) // n_total_bins)
    
        for _, spatial_group in spatial_groups:
            n_samples = min(samples_per_bin, len(spatial_group))
            sampled_indices.extend(random_state.choice(spatial_group.index, size=n_samples, replace=False))
    
    # Return downsampled result
    sampled_df = df.loc[sampled_indices]
    sampled_df = sampled_df.drop(columns=['age_bin', 'lon_bin', 'lat_bin'], errors='ignore')

    return sampled_df
    

def analyze_correlations(corr_matrix, threshold=0.8):
    
    if isinstance(corr_matrix, str):
        corr_matrix = pd.read_csv(corr_matrix, index_col=0)
    else:
        corr_matrix = pd.DataFrame(corr_matrix)
    
    # Dictionary to store correlations
    correlations = {}
    
    for column in corr_matrix.columns:
        positive_corr = []
        negative_corr = []
        feature = corr_matrix[column]
        
        for i in range(feature.shape[0]):
            if abs(feature.iloc[i]) >= threshold and feature.index[i] != column:
                if feature.iloc[i] > 0:
                    positive_corr.append((feature.index[i], feature.iloc[i]))
                else:
                    negative_corr.append((feature.index[i], feature.iloc[i]))
        
        if positive_corr or negative_corr:
            correlations[column] = {
                'positive': sorted(positive_corr, key=lambda x: x[1], reverse=True),
                'negative': sorted(negative_corr, key=lambda x: x[1])
            }
    
    return correlations


def generate_report(correlations, threshold):
    
    print(f"Correlation Analysis Report (Threshold: {threshold})")
    print("=" * 50)
    
    for feature, corr in correlations.items():
        print(f"\nFeature: {feature}")
        print("-" * 30)
        
        if corr['positive']:
            print("Positive Correlations:")
            for c, value in corr['positive']:
                print(f"  {c}: {value:.3f}")
        
        if corr['negative']:
            print("Negative Correlations:")
            for c, value in corr['negative']:
                print(f"  {c}: {value:.3f}")
    
    print("\nTotal features with strong correlations:", len(correlations))
