import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import sys
import os

# Add root directory to path to import functions
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)

# Import all functions from functions.py
from functions import *

def load_data():
    """Load the aluminium data from CSV file"""
    data_path = os.path.join(parent_dir, 'Data', 'aluminium_pre_inputs.csv')
    return pd.read_csv(data_path)

def calculate_correlation(df, shift_val, feature_cols):
    """Calculate correlation between features and volatility target for a given shift value"""
    y = df[f'{pred_value_to_char(shift_val)}_vol'][252:]
    # Use .loc[] instead of .iloc[] for column names, and fix the indexing
    X_shifted = df.iloc[252-shift_val:-shift_val][feature_cols]
    correlation_with_y = X_shifted.corrwith(y)
    return correlation_with_y

def get_top_features(corr_results, expiry_dates, cols):
    """Extract top features from correlation results"""
    top_features_set = set()
    
    for shift_val in expiry_dates:
        top5 = corr_results[f'{pred_value_to_char(shift_val)}_exp'].abs().sort_values(ascending=False).head(5).index
        top_features_set.update(top5)
    
    return sorted(list(top_features_set))

def plot_correlation_bars(corr_results_abs, features, expiry_dates, corr_results):
    """Create horizontal bar chart showing feature correlations with ranks"""
    y = np.arange(len(features))
    height = 0.18
    
    plt.figure(figsize=(10, 8))
    
    for i, shift_val in enumerate(expiry_dates):
        bars = plt.barh(
            y + i*height, 
            corr_results_abs[f'{pred_value_to_char(shift_val)}_exp'], 
            height=height, 
            label=f'{pred_value_to_char(shift_val)}_exp'
        )
        
        # Get ranks based on ALL features (not just top features)
        full_ranks = corr_results[f'{pred_value_to_char(shift_val)}_exp'].abs().rank(ascending=False).astype(int)
        for bar, feature in zip(bars, features):
            rank = full_ranks[feature]
            plt.text(
                bar.get_width() + 0.01,
                bar.get_y() + bar.get_height()/2,
                str(rank),
                va='center',
                fontsize=9,
                color='black'
            )
    
    plt.yticks(y + 1.5*height, features)
    plt.xlabel("Absolute Correlation")
    plt.title("Top Feature Correlations with Ranks (from all features) for Different Expiry Dates")
    plt.legend()
    plt.tight_layout()
    
    # Save the plot without showing
    plot_filename = "top_feature_correlations_with_ranks.png"
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    plt.close()  # Close the figure to free memory
    print(f"Plot saved as: {plot_filename}")

def plot_correlation_distributions(corr_results, expiry_dates):
    """Create histograms showing distribution of correlations for each expiry date"""
    bins = np.arange(0, 1.05, 0.05)
    fig, axes = plt.subplots(2, 2, figsize=(12, 10), sharey=True)
    axes = axes.flatten()
    stats_dict = {}
    
    for i, shift_val in enumerate(expiry_dates):
        data = corr_results[f'{pred_value_to_char(shift_val)}_exp'].abs()
        mean_val = data.mean()
        median_val = data.median()
        
        stats_dict[shift_val] = (mean_val, median_val)
        
        axes[i].hist(
            data,
            bins=bins,
            color='skyblue',
            edgecolor='black'
        )
        
        axes[i].axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.3f}')
        axes[i].axvline(median_val, color='purple', linestyle='-', linewidth=2, label=f'Median: {median_val:.3f}')
        
        axes[i].set_title(f'{pred_value_to_char(shift_val)}_exp', fontsize=12)
        axes[i].set_xlabel(f"Absolute Correlation with {pred_value_to_char(shift_val)}_vol")
        if i % 2 == 0:
            axes[i].set_ylabel("Count of Features")
        axes[i].grid(axis='y', linestyle='--', alpha=0.7)
        axes[i].legend()
    
    plt.suptitle("Distribution of Feature Correlations by Expiry Date", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    # Save the plot without showing
    plot_filename = "correlation_distributions_by_expiry.png"
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    plt.close()  # Close the figure to free memory
    print(f"Plot saved as: {plot_filename}")
    
    return stats_dict

def print_statistics(stats_dict):
    """Print correlation statistics for each expiry date"""
    for shift_val, (mean_val, median_val) in stats_dict.items():
        print(f"Expiry {shift_val}: Mean = {mean_val:.4f}, Median = {median_val:.4f}")

def export_results(corr_results, expiry_dates):
    """Export correlation results to CSV and JSON files"""
    # Export to CSV
    csv_filename = "all_feature_correlations.csv"
    corr_results.to_csv(csv_filename)
    print(f"CSV file saved as: {csv_filename}")
    
    # Export absolute correlations to CSV
    abs_csv_filename = "absolute_feature_correlations.csv"
    corr_results.abs().to_csv(abs_csv_filename)
    print(f"Absolute correlations CSV saved as: {abs_csv_filename}")
    
    # Export to JSON with sorted features
    sorted_features_dict = {}
    
    for shift_val in expiry_dates:
        sorted_corr = corr_results[f'{pred_value_to_char(shift_val)}_exp'].abs().sort_values(ascending=False)
        sorted_features_dict[f'{pred_value_to_char(shift_val)}_exp'] = {
            "features": sorted_corr.index.tolist(),
            "abs_correlation": sorted_corr.values.tolist()
        }
    
    json_filename = "sorted_feature_correlations.json"
    with open(json_filename, "w") as f:
        json.dump(sorted_features_dict, f, indent=4)
    
    print(f"JSON file saved as: {json_filename}")
    print("All files exported successfully to the current directory.")

def main():
    """Main function to run the feature correlation analysis"""
    # Load data
    df = load_data()
    
    # Define parameters
    expiry_dates = [5, 22, 66, 252]
    
    # Get feature columns (excluding target columns and date column)
    cols = [col for col in df.columns if not col.endswith('_vol') and col != 'date']
    
    # Initialize results dataframe
    corr_results = pd.DataFrame(index=cols)
    
    # Calculate correlations for each expiry date
    for shift_val in expiry_dates:
        corr_results[f'{pred_value_to_char(shift_val)}_exp'] = calculate_correlation(df, shift_val, cols)
    
    # Get top features
    top_features = get_top_features(corr_results, expiry_dates, cols)
    corr_results_abs = corr_results.abs().loc[top_features]
    
    # Create visualizations
    plot_correlation_bars(corr_results_abs, top_features, expiry_dates, corr_results)
    stats_dict = plot_correlation_distributions(corr_results, expiry_dates)
    
    # Print statistics
    print_statistics(stats_dict)
    
    # Export results
    export_results(corr_results, expiry_dates)

if __name__ == "__main__":
    main()