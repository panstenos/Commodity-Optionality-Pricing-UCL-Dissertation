"""
Add GARCH(1,1) Predictions to Aluminium Pre-Inputs Dataset

This script loads GARCH predictions from JSON and adds them as properly aligned columns
to the aluminium_pre_inputs.csv dataset.
"""

import pandas as pd
import numpy as np
import json
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.abspath('..'))
from functions import load_data, pred_char_to_value

def load_garch_predictions(json_file="arch_all_results/garch11_predictions.json"):
    """
    Load GARCH predictions from JSON file.
    
    Parameters:
    -----------
    json_file : str
        Path to the JSON file containing predictions
    
    Returns:
    --------
    predictions_dict : dict
        Dictionary with horizon keys and prediction arrays
    """
    print(f"Loading GARCH predictions from: {json_file}")
    
    with open(json_file, 'r') as f:
        predictions_dict = json.load(f)
    
    # Convert lists back to numpy arrays
    for horizon in predictions_dict:
        predictions_dict[horizon] = np.array(predictions_dict[horizon])
    
    print("Loaded predictions for horizons:", list(predictions_dict.keys()))
    for horizon, preds in predictions_dict.items():
        print(f"  {horizon}: {len(preds)} predictions")
    
    return predictions_dict

def add_garch_predictions_to_dataset(input_file="../Data/aluminium_pre_inputs.csv", 
                                   output_file="../Data/aluminium_pre_inputs_with_garch.csv",
                                   predictions_json="arch_all_results/garch11_predictions.json"):
    """
    Add GARCH predictions as new columns to the dataset with proper alignment.
    
    Parameters:
    -----------
    input_file : str
        Path to the original dataset
    output_file : str
        Path to save the enhanced dataset
    predictions_json : str
        Path to the JSON file containing GARCH predictions
    """
    print("=== Adding GARCH Predictions to Dataset ===")
    
    # Load original dataset
    print(f"Loading original dataset: {input_file}")
    df = load_data(input_file)
    print(f"Original dataset shape: {df.shape}")
    
    # Load GARCH predictions
    garch_predictions = load_garch_predictions(predictions_json)
    
    # Create a copy of the dataframe
    df_enhanced = df.copy()
    
    # Add GARCH prediction columns with proper alignment
    for horizon, predictions in garch_predictions.items():
        horizon_days = pred_char_to_value(horizon)
        context_window = 2 * horizon_days  # c = 2h
        
        print(f"\nAdding {horizon} GARCH predictions:")
        print(f"  Horizon: {horizon_days} days")
        print(f"  Context window: {context_window} days")
        print(f"  Predictions: {len(predictions)} values")
        
        # Create column name
        garch_col_name = f'{horizon}_garch_pred'
        
        # Initialize column with NaN values
        df_enhanced[garch_col_name] = np.nan
        
        # Align predictions with the dataset
        # Predictions start at index context_window and go to the end
        start_idx = context_window
        end_idx = start_idx + len(predictions)
        
        print(f"  Alignment: predictions[{len(predictions)}] -> rows[{start_idx}:{end_idx}]")
        
        # Ensure we don't exceed the dataframe length
        if end_idx > len(df_enhanced):
            # Truncate predictions if necessary
            max_predictions = len(df_enhanced) - start_idx
            predictions = predictions[:max_predictions]
            end_idx = start_idx + max_predictions
            print(f"  Warning: Truncated predictions to {max_predictions} to fit dataset")
        
        # Insert predictions at the correct positions
        df_enhanced.loc[df_enhanced.index[start_idx:end_idx], garch_col_name] = predictions
        
        # Count non-null values
        non_null_count = df_enhanced[garch_col_name].notna().sum()
        print(f"  Added {non_null_count} predictions to column '{garch_col_name}'")
    
    # Save enhanced dataset
    print(f"\nSaving enhanced dataset: {output_file}")
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Save with index=True to preserve the date index
    df_enhanced.to_csv(output_file, index=True)
    
    print(f"Enhanced dataset shape: {df_enhanced.shape}")
    print(f"New columns added: {[f'{h}_garch_pred' for h in garch_predictions.keys()]}")
    
    # Show sample of the new columns
    print("\nSample of GARCH prediction columns:")
    garch_cols = [f'{h}_garch_pred' for h in garch_predictions.keys()]
    sample_data = df_enhanced[garch_cols].dropna().head()
    print(sample_data)
    
    return df_enhanced

def verify_alignment(df_enhanced, predictions_json="arch_all_results/garch11_predictions.json"):
    """
    Verify that the GARCH predictions are properly aligned with the dataset.
    
    Parameters:
    -----------
    df_enhanced : pandas.DataFrame
        Enhanced dataframe with GARCH predictions
    predictions_json : str
        Path to the original predictions JSON file
    """
    print("\n=== Verifying Alignment ===")
    
    # Load original predictions for verification
    garch_predictions = load_garch_predictions(predictions_json)
    
    for horizon, original_predictions in garch_predictions.items():
        horizon_days = pred_char_to_value(horizon)
        context_window = 2 * horizon_days
        garch_col_name = f'{horizon}_garch_pred'
        
        # Get the predictions from the dataframe
        df_predictions = df_enhanced[garch_col_name].dropna().values
        
        # Compare lengths
        print(f"{horizon} alignment check:")
        print(f"  Original predictions: {len(original_predictions)}")
        print(f"  DataFrame predictions: {len(df_predictions)}")
        
        # Check if they match
        if len(original_predictions) == len(df_predictions):
            # Check if values match (allowing for small floating point differences)
            if np.allclose(original_predictions, df_predictions, rtol=1e-10):
                print(f"  ✓ Alignment verified: predictions match exactly")
            else:
                print(f"  ✗ Warning: predictions don't match exactly")
                print(f"    Max difference: {np.max(np.abs(original_predictions - df_predictions))}")
        else:
            print(f"  ✗ Warning: length mismatch")
        
        # Show first and last few values for manual verification
        print(f"  First 3 original: {original_predictions[:3]}")
        print(f"  First 3 dataframe: {df_predictions[:3]}")
        print(f"  Last 3 original: {original_predictions[-3:]}")
        print(f"  Last 3 dataframe: {df_predictions[-3:]}")
        print()

if __name__ == "__main__":
    # Add GARCH predictions to the dataset
    enhanced_df = add_garch_predictions_to_dataset()
    
    # Verify alignment
    verify_alignment(enhanced_df)
    
    print("\n=== Process Complete ===")
    print("Enhanced dataset saved with GARCH predictions properly aligned!")
    print("New columns: 1w_garch_pred, 1m_garch_pred, 3m_garch_pred, 1y_garch_pred")
