import torch
import torch.nn as nn
import torch.nn.functional as F
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
import sys
import os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, parent_dir)
neural_nets_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, neural_nets_dir)
from functions import pred_value_to_char, load_data
import pandas as pd
from NN_functions import preprocess_dataframe, get_dataloaders, create_model, train_model, plot_train_test_predictions, evaluate_and_print_metrics, train_one_epoch, validate_one_epoch

data_path = os.path.join(parent_dir, 'Data', 'aluminium_pre_inputs.csv')
df = load_data(data_path)
df['al_lme_prices_abs_log_returns'] = abs(df['al_lme_prices_log_returns'])

window = 10
expiry = 5
batch_size = 64

X_raw, y_raw, input_size = preprocess_dataframe(df['al_lme_prices_abs_log_returns'], df[f'{pred_value_to_char(expiry)}_vol'], window, expiry)

trainloader, testloader, y_scaler, X_train, X_test, y_train, y_test = get_dataloaders(X_raw, y_raw, window, expiry, batch_size)

hidden_size = 64
number_layers = 3
output_size = 1

# Create LSTM model (default - model_type='LSTM' or omitted)
model = create_model(input_size, hidden_size, number_layers, output_size, activation_fn=nn.ReLU())

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_function = nn.MSELoss()
n_epochs = 50

train_model(model, optimizer, loss_function, trainloader, testloader, n_epochs)

plot_train_test_predictions(model, X_train, y_train, X_test, y_test, y_scaler, window, device, f'{pred_value_to_char(expiry)}_vol')
evaluate_and_print_metrics(model, X_test, y_test, y_scaler, window, device)

# =============================================================================
# VMD-LSTM TRAINING WITH SAME PARAMETERS
# =============================================================================
print("\n" + "=" * 60)
print("VMD-LSTM TRAINING (Same parameters as LSTM above)")
print("=" * 60)

# Use the same parameters as the LSTM training above
expiry = 5
window = expiry*2
batch_size = 64

X_raw, y_raw, input_size = preprocess_dataframe(df[f'{pred_value_to_char(expiry)}_vol'], df[f'{pred_value_to_char(expiry)}_vol'], window, expiry)

trainloader, testloader, y_scaler, X_train, X_test, y_train, y_test = get_dataloaders(X_raw, y_raw, window, expiry, batch_size)

hidden_size = 64
number_layers = 3
output_size = 1

# Create VMD-LSTM model with same parameters (with debug enabled)
model = create_model(input_size, hidden_size, number_layers, output_size, activation_fn=nn.ReLU(), model_type='VMD_LSTM', debug=True)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_function = nn.MSELoss()
n_epochs = 50

train_model(model, optimizer, loss_function, trainloader, testloader, n_epochs)

plot_train_test_predictions(model, X_train, y_train, X_test, y_test, y_scaler, window, device, f'{pred_value_to_char(expiry)}_vol')
evaluate_and_print_metrics(model, X_test, y_test, y_scaler, window, device)

# =============================================================================
# MODEL USAGE EXAMPLES AND COMPARISON
# =============================================================================
# 
# You can now choose between two different architectures:
# 
# 1. LSTM (default): Standard LSTM with stacked layers
#    - Use: create_model(..., model_type='LSTM') or omit model_type parameter
#    - Pros: Simple, fast training, good for basic time series
#    - Cons: May overfit, limited feature interaction
# 
# 2. VMD-LSTM: Enhanced LSTM with Variational Mode Decomposition
#    - Use: create_model(..., model_type='VMD_LSTM')
#    - Pros: Decomposes time series into IMFs, handles non-stationary data, better pattern recognition
#    - Cons: More complex, requires tuning of VMD parameters
# 
# =============================================================================

# Example: Quick comparison of both models
print("\n" + "=" * 60)
print("QUICK MODEL COMPARISON: LSTM vs VMD-LSTM")
print("=" * 60)

# Test both models with the same parameters for comparison
expiry = 5
window = 10
batch_size = 32
hidden_size = 32
number_layers = 2
output_size = 1

X_raw, y_raw, input_size = preprocess_dataframe(df['al_lme_prices_abs_log_returns'], df[f'{pred_value_to_char(expiry)}_vol'], window, expiry)
trainloader, testloader, y_scaler, X_train, X_test, y_train, y_test = get_dataloaders(X_raw, y_raw, window, expiry, batch_size)

# Test LSTM
print("\n--- Testing LSTM Model ---")
lstm_model = create_model(input_size, hidden_size, number_layers, output_size, activation_fn=nn.ReLU(), model_type='LSTM')
lstm_optimizer = torch.optim.Adam(lstm_model.parameters(), lr=0.001)
lstm_loss_function = nn.MSELoss()

# Quick training for comparison (fewer epochs)
print("Training LSTM model...")
for epoch in range(1, 11):  # Just 10 epochs for quick comparison
    print(f"Epoch {epoch}/10")
    train_one_epoch(lstm_model, lstm_optimizer, lstm_loss_function, trainloader)
    validate_one_epoch(lstm_model, lstm_loss_function, testloader)

# Test VMD-LSTM
print("\n--- Testing VMD-LSTM Model ---")
vmd_model = create_model(input_size, hidden_size, number_layers, output_size, activation_fn=nn.ReLU(), model_type='VMD_LSTM', debug=True)
vmd_optimizer = torch.optim.Adam(vmd_model.parameters(), lr=0.001)
vmd_loss_function = nn.MSELoss()

# Quick training for comparison (fewer epochs)
print("Training VMD-LSTM model...")
for epoch in range(1, 11):  # Just 10 epochs for quick comparison
    print(f"Epoch {epoch}/10")
    train_one_epoch(vmd_model, vmd_optimizer, vmd_loss_function, trainloader)
    validate_one_epoch(vmd_model, vmd_loss_function, testloader)

print("\n" + "=" * 60)
print("MODEL COMPARISON COMPLETE!")
print("=" * 60)
print("You can now compare the performance of both models.")
print("VMD-LSTM typically provides better pattern recognition and handles non-stationary data,")
print("while LSTM is simpler and may train faster.")
print("=" * 60)

# =============================================================================
# USAGE EXAMPLES:
# =============================================================================
# 
# # For LSTM (default):
# model = create_model(input_size, hidden_size, num_layers, output_size, activation_fn=nn.ReLU())
# # OR explicitly:
# model = create_model(input_size, hidden_size, num_layers, output_size, activation_fn=nn.ReLU(), model_type='LSTM')
# 
# # For VMD-LSTM:
# model = create_model(input_size, hidden_size, num_layers, output_size, activation_fn=nn.ReLU(), model_type='VMD_LSTM')
# 
# # For STL-LSTM (Seasonal-Trend-LSTM):
# model = create_model(input_size, hidden_size, num_layers, output_size, activation_fn=nn.ReLU(), 
#                     model_type='STL_LSTM', seasonal_period=12, trend_window=7)
# 
# =============================================================================

# Example: STL-LSTM Training
print("\n" + "=" * 60)
print("STL-LSTM TRAINING EXAMPLE")
print("=" * 60)

# Use appropriate parameters for STL-LSTM
expiry = 5
window = 20  # Longer window for better seasonal detection
batch_size = 32
hidden_size = 64
number_layers = 2
output_size = 1

X_raw, y_raw, input_size = preprocess_dataframe(df['al_lme_prices_abs_log_returns'], df[f'{pred_value_to_char(expiry)}_vol'], window, expiry)

trainloader, testloader, y_scaler, X_train, X_test, y_train, y_test = get_dataloaders(X_raw, y_raw, window, expiry, batch_size)

# Create STL-LSTM model with seasonal and trend parameters
model = create_model(input_size, hidden_size, number_layers, output_size, 
                    activation_fn=nn.ReLU(), model_type='STL_LSTM', 
                    seasonal_period=5, trend_window=7)  # Adjust based on your data

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_function = nn.MSELoss()
n_epochs = 30  # Fewer epochs for demonstration

# Quick training for STL-LSTM
print("Training STL-LSTM model...")
for epoch in range(1, n_epochs + 1):
    print(f"Epoch {epoch}/{n_epochs}")
    train_one_epoch(model, optimizer, loss_function, trainloader)
    validate_one_epoch(model, loss_function, testloader)

print("STL-LSTM training completed!")

# =============================================================================

# Example: VMD-LSTM with Custom Loss Function (Prevents Flat Predictions)
print("\n" + "=" * 60)
print("VMD-LSTM WITH ANTI-FLAT LOSS FUNCTIONS")
print("=" * 60)

# Use appropriate parameters for VMD-LSTM
expiry = 5
window = 20  # Longer window for better VMD decomposition
batch_size = 32
hidden_size = 64
number_layers = 2
output_size = 1

X_raw, y_raw, input_size = preprocess_dataframe(df['al_lme_prices_abs_log_returns'], df[f'{pred_value_to_char(expiry)}_vol'], window, expiry)

trainloader, testloader, y_scaler, X_train, X_test, y_train, y_test = get_dataloaders(X_raw, y_raw, window, expiry, batch_size)

# Create VMD-LSTM model
model = create_model(input_size, hidden_size, number_layers, output_size, 
                    activation_fn=nn.ReLU(), model_type='VMD_LSTM', 
                    num_modes=3, alpha=1000, debug=True)  # Reduced alpha for stability

# Option 1: Use enhanced VMDLSTMLoss
print("Training with Enhanced VMDLSTMLoss...")
loss_function1 = VMDLSTMLoss(alpha=0.3, beta=0.2, gamma=0.1)
optimizer1 = torch.optim.Adam(model.parameters(), lr=0.001)
n_epochs = 20

for epoch in range(1, n_epochs + 1):
    print(f"Epoch {epoch}/{n_epochs}")
    train_one_epoch(model, optimizer1, loss_function1, trainloader)
    validate_one_epoch(model, loss_function1, testloader)

print("Enhanced VMDLSTMLoss training completed!")

# Option 2: Use AntiFlatLoss for even more aggressive flatness prevention
print("\nTraining with AntiFlatLoss...")
loss_function2 = AntiFlatLoss(base_loss=nn.MSELoss(), flatness_weight=2.0)
optimizer2 = torch.optim.Adam(model.parameters(), lr=0.0005)  # Lower learning rate
n_epochs = 15

for epoch in range(1, n_epochs + 1):
    print(f"Epoch {epoch}/{n_epochs}")
    train_one_epoch(model, optimizer2, loss_function2, trainloader)
    validate_one_epoch(model, loss_function2, testloader)

print("AntiFlatLoss training completed!")

# Test the model to see if predictions are no longer flat
print("\nTesting model predictions...")
model.eval()
with torch.no_grad():
    test_batch = next(iter(testloader))
    x_test, y_test = test_batch
    predictions = model(x_test.to(device))
    
    print(f"Prediction shape: {predictions.shape}")
    print(f"Prediction variance: {torch.var(predictions).item():.6f}")
    print(f"Target variance: {torch.var(y_test).item():.6f}")
    print(f"Prediction range: {torch.max(predictions).item() - torch.min(predictions).item():.6f}")
    print(f"Target range: {torch.max(y_test).item() - torch.min(y_test).item():.6f}")

print("VMD-LSTM training with anti-flat losses completed!")

# =============================================================================
# MULTI-DIMENSIONAL INPUT EXAMPLE
# =============================================================================
print("\n" + "=" * 60)
print("MULTI-DIMENSIONAL INPUT EXAMPLE")
print("=" * 60)

# Example: Using multiple features as input
# This demonstrates how to use multiple columns as features
print("Example: Multi-dimensional input with multiple features")
print("Input features: ['al_lme_prices_abs_log_returns', 'vix']")

# Use appropriate parameters for multi-dimensional input
expiry = 5
window = 20
batch_size = 32
hidden_size = 64
number_layers = 2
output_size = 1

# Create multi-dimensional input with multiple features
# Note: You can add more features by adding more columns to the list
multi_feature_input = df[['al_lme_prices_abs_log_returns', 'vix']]

# Prepare data with multi-dimensional input
X_raw, y_raw, input_size = preprocess_dataframe(
    multi_feature_input,  # Multi-dimensional input
    df[f'{pred_value_to_char(expiry)}_vol'],  # Target variable
    window, expiry
)

print(f"Multi-dimensional input shape: {X_raw.shape}")
print(f"Number of input features: {input_size}")
print(f"Expected input shape: (sequences, {window}, {input_size})")

trainloader, testloader, y_scaler, X_train, X_test, y_train, y_test = get_dataloaders(X_raw, y_raw, window, expiry, batch_size)

# Create model with correct input_size
model = create_model(input_size, hidden_size, number_layers, output_size, 
                    activation_fn=nn.ReLU(), model_type='LSTM')

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_function = nn.MSELoss()
n_epochs = 10  # Fewer epochs for demonstration

# Quick training for multi-dimensional input
print("Training LSTM model with multi-dimensional input...")
for epoch in range(1, n_epochs + 1):
    print(f"Epoch {epoch}/{n_epochs}")
    train_one_epoch(model, optimizer, loss_function, trainloader)
    validate_one_epoch(model, loss_function, testloader)

print("Multi-dimensional input training completed!")

# Test the model to verify it works with multi-dimensional input
print("\nTesting multi-dimensional input model...")
model.eval()
with torch.no_grad():
    test_batch = next(iter(testloader))
    x_test, y_test = test_batch
    print(f"Test input shape: {x_test.shape}")
    print(f"Expected: (batch_size, {window}, {input_size})")
    
    predictions = model(x_test.to(device))
    print(f"Prediction shape: {predictions.shape}")
    print(f"Target shape: {y_test.shape}")

print("Multi-dimensional input example completed successfully!")

# =============================================================================
# USAGE GUIDE FOR MULTI-DIMENSIONAL INPUT
# =============================================================================
print("\n" + "=" * 60)
print("MULTI-DIMENSIONAL INPUT USAGE GUIDE")
print("=" * 60)
print("To use multiple features as input:")
print("1. Select multiple columns: df[['feature1', 'feature2', 'feature3']]")
print("2. Pass to preprocess_dataframe()")
print("3. The function automatically detects the number of features")
print("4. All models automatically adapt to the input dimensions")
print("5. No manual input_size calculation needed!")
print("=" * 60)
# =============================================================================

