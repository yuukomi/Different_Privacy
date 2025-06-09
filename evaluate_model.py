import pandas as pd
import numpy as np
from data_preprocessing import preprocess_data
from linear_models import LinearRegression, QuadraticRegression, InteractionRegression, HybridRegression

def evaluate_from_csv(csv_path, model_type="basic_linear", show_data=True):
    """
    Evaluate model accuracy on data from a CSV file
    
    Parameters:
    -----------
    csv_path : str
        Path to the CSV file
    model_type : str
        Type of model to use ('basic_linear', 'quadratic', 'interaction', or 'hybrid')
    show_data : bool
        Whether to show sample data information
        
    Returns:
    --------
    tuple
        (MSE, RMSE, MAE) metrics
    """
    try:
        # Load and preprocess data
        print(f"\nLoading data from: {csv_path}")
        df = pd.read_csv(csv_path)
        
        if show_data:
            print("\nData Overview:")
            print(f"Number of samples: {len(df)}")
            print(f"Features: {', '.join(df.columns.tolist())}")
            print("\nFirst few samples:")
            print(df.head())
            print("\nData Summary:")
            print(df.describe())
        
        X, y, _ = preprocess_data(df)
        
        if show_data:
            print("\nPreprocessed Features:")
            print(X.head())
            print("\nTarget Values:")
            print(y.head())
        
        # Select model
        models = {
            "basic_linear": LinearRegression(),
            "quadratic": QuadraticRegression(),
            "interaction": InteractionRegression(),
            "hybrid": HybridRegression()
        }
        
        if model_type not in models:
            raise ValueError(f"Invalid model type. Choose from: {list(models.keys())}")
            
        model = models[model_type]
        
        # Train and evaluate
        print(f"\nTraining {model_type} model...")
        model.fit(X, y)
        y_pred = model.predict(X)
        
        # Calculate metrics
        mse = np.mean((y - y_pred) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(y - y_pred))
        
        # Print results
        print("\nModel Performance Metrics:")
        print(f"Mean Squared Error: {mse:.2f}")
        print(f"Root Mean Squared Error: {rmse:.2f}")
        print(f"Mean Absolute Error: {mae:.2f}")
        
        if show_data:
            print("\nPrediction Examples:")
            comparison = pd.DataFrame({
                'Actual Price': y[:5],
                'Predicted Price': y_pred[:5],
                'Difference': np.abs(y[:5] - y_pred[:5])
            })
            print(comparison)
        
        return mse, rmse, mae
        
    except FileNotFoundError:
        print(f"Error: File not found at {csv_path}")
        return None
    except Exception as e:
        print(f"Error: {str(e)}")
        return None

if __name__ == "__main__":
    # Example usage
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python evaluate_model.py <csv_path> [model_type] [show_data]")
        print("Model types: basic_linear, quadratic, interaction, hybrid")
        print("show_data: 0 or 1 (default: 1)")
        sys.exit(1)
        
    csv_path = sys.argv[1]
    model_type = sys.argv[2] if len(sys.argv) > 2 else "basic_linear"
    show_data = bool(int(sys.argv[3])) if len(sys.argv) > 3 else True
    
    results = evaluate_from_csv(csv_path, model_type, show_data)
    
    if results:
        print("\nEvaluation completed successfully!") 