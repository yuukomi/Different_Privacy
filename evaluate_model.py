import pandas as pd
import numpy as np
import sys
from Different_Privacy.data_preprocessing import preprocess_data

def evaluate_from_csv(csv_path, model_type="Basic Linear", show_data=True):
    """
    Evaluate model accuracy on data from a CSV file
    
    Parameters:
    -----------
    csv_path : str
        Path to the CSV file
    model_type : str
        Type of model to use ('Basic Linear', 'Quadratic', 'Interaction', or 'Hybrid')
    show_data : bool
        Whether to show sample data information
        
    Returns:
    --------
    tuple
        (MSE, RMSE, MAE) metrics
    """
    try:
        # Load data
        print(f"\nLoading data from: {csv_path}")
        df = pd.read_csv(csv_path)
        
        if show_data:
            print("\n" + "="*50)
            print("DATA OVERVIEW")
            print("="*50)
            print(f"Number of samples: {len(df)}")
            print(f"Number of features: {len(df.columns)}")
            print(f"Features: {', '.join(df.columns.tolist())}")
            
            print("\nFirst 5 rows:")
            print(df.head())
            
            print("\nData types:")
            print(df.dtypes)
            
            print("\nMissing values:")
            missing_values = df.isnull().sum()
            if missing_values.sum() > 0:
                print(missing_values[missing_values > 0])
            else:
                print("No missing values found")
            
            print("\nBasic statistics:")
            print(df.describe())
        
        # Preprocess data
        print(f"\nPreprocessing data...")
        X, y, _ = preprocess_data(df)
        
        if show_data:
            print(f"\nAfter preprocessing:")
            print(f"Feature matrix shape: {X.shape}")
            print(f"Target vector shape: {y.shape}")
            print(f"Feature names: {X.columns.tolist()}")
        
        # Import models and evaluation functions
        from linear_models import LinearRegression, QuadraticRegression, InteractionRegression, HybridRegression, mean_squared_error, mean_absolute_error
        
        models = {
            "Basic Linear": LinearRegression(),
            "Quadratic": QuadraticRegression(),
            "Interaction": InteractionRegression(),
            "Hybrid": HybridRegression()
        }
        
        if model_type not in models:
            print(f"Warning: Model type '{model_type}' not found. Available models: {list(models.keys())}")
            print("Using 'Basic Linear' as default.")
            model_type = "Basic Linear"
        
        model = models[model_type]
        
        # Train model
        print(f"\n" + "="*50)
        print(f"TRAINING {model_type.upper()} MODEL")
        print("="*50)
        
        model.fit(X, y)
        
        # Make predictions
        y_pred = model.predict(X)
        
        # Calculate metrics
        mse = mean_squared_error(y, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y, y_pred)
        
        # Calculate additional metrics
        r2 = 1 - (np.sum((y - y_pred) ** 2) / np.sum((y - np.mean(y)) ** 2))
        mape = np.mean(np.abs((y - y_pred) / (y + 1e-8))) * 100
        
        # Print results
        print(f"\n" + "="*50)
        print("MODEL PERFORMANCE METRICS")
        print("="*50)
        print(f"Mean Squared Error (MSE): {mse:.2f}")
        print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
        print(f"Mean Absolute Error (MAE): {mae:.2f}")
        print(f"R-squared (R²): {r2:.4f}")
        print(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")
        
        if show_data:
            print(f"\n" + "="*50)
            print("PREDICTION EXAMPLES")
            print("="*50)
            
            # Show first 10 predictions
            n_examples = min(10, len(y))
            comparison = pd.DataFrame({
                'Actual': y.iloc[:n_examples].values,
                'Predicted': y_pred[:n_examples],
                'Absolute Error': np.abs(y.iloc[:n_examples].values - y_pred[:n_examples]),
                'Percentage Error': np.abs((y.iloc[:n_examples].values - y_pred[:n_examples]) / (y.iloc[:n_examples].values + 1e-8)) * 100
            })
            
            print(comparison.round(2))
            
            # Show prediction distribution
            print(f"\n" + "="*50)
            print("PREDICTION DISTRIBUTION")
            print("="*50)
            print(f"Actual values - Min: {y.min():.2f}, Max: {y.max():.2f}, Mean: {y.mean():.2f}")
            print(f"Predicted values - Min: {y_pred.min():.2f}, Max: {y_pred.max():.2f}, Mean: {y_pred.mean():.2f}")
            
            # Show error distribution
            errors = np.abs(y - y_pred)
            print(f"\nError distribution:")
            print(f"Min error: {errors.min():.2f}")
            print(f"Max error: {errors.max():.2f}")
            print(f"Mean error: {errors.mean():.2f}")
            print(f"Median error: {np.median(errors):.2f}")
            
            # Show accuracy within different ranges
            error_ranges = [100, 500, 1000, 2000, 5000]
            print(f"\nAccuracy within error ranges:")
            for error_range in error_ranges:
                within_range = np.sum(errors <= error_range)
                percentage = (within_range / len(errors)) * 100
                print(f"Within ±{error_range}: {within_range}/{len(errors)} ({percentage:.1f}%)")
        
        return mse, rmse, mae
        
    except FileNotFoundError:
        print(f"Error: File not found at {csv_path}")
        print("Please check the file path and ensure the file exists.")
        return None
    except Exception as e:
        print(f"Error during evaluation: {str(e)}")
        print("Please check your data format and try again.")
        return None

if __name__ == "__main__":
    # Command line interface
    if len(sys.argv) < 2:
        print("Usage: python evaluate_model.py <csv_path> [model_type] [show_data]")
        print("Example: python evaluate_model.py data.csv 'Hybrid' True")
        sys.exit(1)
    
    csv_path = sys.argv[1]
    model_type = sys.argv[2] if len(sys.argv) > 2 else "basic_linear"
    show_data = bool(int(sys.argv[3])) if len(sys.argv) > 3 else True
    
    results = evaluate_from_csv(csv_path, model_type, show_data)
    
    if results:
        print("\nEvaluation completed successfully!")
