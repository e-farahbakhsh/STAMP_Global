'''
Feature Analysis

Author: Ehsan Farahbakhsh
Contact email: e.farahbakhsh@sydney.edu.au
Date last modified: 04/05/2026
'''


import numpy as np
import pandas as pd


def downsample(
        df,
        n_target,
        n_bins_age=50,
        n_bins_spatial=10,
        random_state=42,
    ):
    
    """
    Downsample a dataset to a target number of samples while
    preserving age and spatial diversity.

    This function reduces the size of an input DataFrame by:
    1. Binning the data into quantile-based age bins (`age (Ma)` column).
    2. Within each age bin, further dividing points into longitude and latitude bins.
    3. Sampling approximately equal numbers of points from each spatial bin, 
       proportional to the overall target sample size.

    Parameters
    ----------
    df : pandas.DataFrame
        Input data containing at least 'age (Ma)', 'lon', and 'lat' columns.
    n_target : int
        Desired number of samples in the downsampled dataset.
    n_bins_age : int, default=50
        Number of quantile-based bins to divide the age axis into.
    n_bins_spatial : int, default=10
        Number of equally sized bins for both longitude and latitude within each age bin.
    random_state : int, numpy.random.Generator, or None, default=None
        Random seed or generator for reproducible sampling.

    Returns
    -------
    pandas.DataFrame
        Downsampled DataFrame containing a balanced subset of the input data.
        Extra binning columns ('age_bin', 'lon_bin', 'lat_bin') are removed before returning.

    Notes
    -----
    - If the dataset has fewer rows than `n_target`, the original DataFrame is returned unchanged.
    - Sampling ensures that each spatial bin contributes at least one point (if available),
      which helps retain geographic coverage across all ages.
    """
    
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
    
    for age_bin, age_group in df.groupby('age_bin', observed=True):
        age_group = age_group.copy()
    
        # Spatial binning within this age group
        age_group['lon_bin'] = pd.cut(age_group['lon'], bins=n_bins_spatial)
        age_group['lat_bin'] = pd.cut(age_group['lat'], bins=n_bins_spatial)
    
        # Group by spatial bins
        spatial_groups = age_group.groupby(['lon_bin', 'lat_bin'], observed=True)

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
    
    if len(sampled_df) > n_target:
        print(f"To preserve age and spatial diversity, considering current n_bins_age and n_bins_spatial, the number of samples must be at least {len(sampled_df)} which is over n_target={n_target}.")

    return sampled_df
    

def analyze_correlations(corr_matrix, threshold=0.8):
    
    """
    Identify strong positive and negative correlations between features.

    This function scans through a correlation matrix and extracts feature pairs
    whose correlation magnitude exceeds a given threshold. It separates strong
    correlations into positive and negative groups and returns them in a
    structured dictionary.

    Parameters
    ----------
    corr_matrix : str, pandas.DataFrame, or array-like
        Input correlation matrix. Can be:
        - A file path (CSV) containing a correlation matrix,
        - A pandas DataFrame, or
        - Any array-like structure convertible to a DataFrame.
    threshold : float, default=0.8
        Minimum absolute correlation value to consider a relationship significant.

    Returns
    -------
    dict
        Dictionary mapping each feature (column name) to its correlations:
        {
            'feature_name': {
                'positive': [(other_feature, corr_value), ...],
                'negative': [(other_feature, corr_value), ...]
            },
            ...
        }
        - 'positive' contains positively correlated features sorted by
          descending correlation strength.
        - 'negative' contains negatively correlated features sorted by
          ascending correlation strength.

    Notes
    -----
    - Self-correlations (feature with itself) are excluded.
    - Features with no correlations above the threshold are omitted from the result.
    - Useful for identifying redundant variables, feature groups, or candidates
      for dimensionality reduction.
    """
    
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
